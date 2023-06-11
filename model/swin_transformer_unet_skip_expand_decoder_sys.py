import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import torch.utils.checkpoint as checkpoint
import math
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from model.decoder_blocks import Decoder_Attention
from model.decoder_blocks import Decoder_Block
from model.module import ResnetBlock
from model.regression import Regression

from model.utils import padding, unpadding, SoftEncodeAB, CIELAB, AnnealedMeanDecodeQ

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5, device='cuda'):
        prior = torch.from_numpy(cielab.gamut.prior)

        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def __call__(self, ab_actual):
        return self.weights[ab_actual.argmax(dim=1, keepdim=True)].to(ab_actual.device)

class RebalanceLoss(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, data_input, weights):
        ctx.save_for_backward(weights)

        return data_input.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors

        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        grad_input = grad_output * weights

        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return grad_input, None


def window_partition(x, window_size):
    """
    将feature_map按照window_size划分为一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将一个个window还原成一个feature_map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
    return block


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww] 绝对位置索引
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        # print("type(q):", type(q))
        # print("type(self.scale):", type(self.scale))

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        # attn = attn * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # if self.shift_size > 0:
        #     # calculate attention mask for SW-MSA
        #     H, W = self.input_resolution  # 64
        #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        #     # 保证Hp和Wp是window_size的整数倍
        #     # Hp = int(np.ceil(H / self.window_size)) * self.window_size
        #     # Wp = int(np.ceil(W / self.window_size)) * self.window_size
        #     # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        #     # img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        #
        #     h_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     w_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1
        #
        #     mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw] 将每个window展平
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1] 关键:广播机制
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # else:
        #     attn_mask = None
        #
        # self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, window_size*window_size, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."



        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B H/2*W/2 2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C -> B, 2*H*2*W, C//2
        """
        H, W = self.input_resolution
        x = self.expand(x)  # B, L, C
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)  # B, 2*H*2*W, C/2
        x= self.norm(x)

        return x

class SkipPatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim//2)

    def forward(self, x):
        """
        x: B, H*W, C -> B, 2*H*2*W, C//2
        """
        H, W = self.input_resolution
        # x_linear = self.linear(x)
        x = self.expand(x)  # B, L, C  2*C_ori
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)  # B, 2*H*2*W, C_ori/2
        x = self.norm(x)

        return x

class SkipPatchConv(nn.Module):
    def __init__(self, indim, outdim):
        super.__init__()
        self.patchconv = nn.Sequential(
                conv3x3_bn_relu(indim, indim),
                nn.ConvTranspose2d(indim, outdim, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(True),
        )
        self.norm = nn.LayerNorm(outdim)

    def forward(self, x):
        x = self.patchconv(x)
        x = x + self.norm(x)
        return x

class SkipColorConv(nn.Module):
    def __init__(self, indim, outdim):
        super.__init__()
        self.colorconv = nn.Sequential(
                conv3x3_bn_relu(indim, indim),
                nn.ConvTranspose2d(indim, outdim, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(True),
        )
        self.norm = nn.LayerNorm(outdim)

    def forward(self, x):
        x = self.colorconv(x)
        x = x + self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= x + self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # 64(img_size=256) or 56(img_size=224)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]  广播机制
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # 多尺度
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1)//2, (W + 1)//2  # 保证是偶数
        return x, H, W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # 64
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  #[B, HW, C]
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim, q_to_ab=None, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s
        self.q_to_ab = q_to_ab      # [313, 2]

    def forward(self, x, H, W):
        B, N, C = x.shape           # [B, 313, C]
        if self.q_to_ab is not None:        # color pos.
            cnn_feat = torch.zeros(B, H, W, C).to(x.device)    # [b, 23, 23, c]
            bin = 10
            torch_ab = torch.from_numpy(self.q_to_ab).to(x.device)
            # new_ab = (torch_ab + 110) // bin        # [313, 2]
            new_ab = torch.div(torch_ab + 110, bin, rounding_mode='floor')
            cnn_feat[:, new_ab[:, 0].long(), new_ab[:, 1].long(), :] = x      # [B, N, C]

            conv_cnn_feat = self.proj(cnn_feat.permute(0, 3, 1, 2))     # [B, C, 23, 23]
            conv_cnn_feat = conv_cnn_feat.permute(0, 2, 3, 1)       # [B, 23, 23, C]
            x_pos = torch.zeros_like(x)
            x_pos[:, :, :] = conv_cnn_feat[:, new_ab[:, 0].long(), new_ab[:, 1].long(), :]     # [B, N, C]
            x = x + x_pos
        else:       # patch pos.
            feat_token = x
            cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat
            x = x.flatten(2).transpose(1, 2)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class SwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,  # 128
                 depths=[2, 2, 2, 2],
                 depths_decoder=[1, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 final_upsample="expand_first",
                 cls=313,
                 with_regression=False):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution  # 64
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        #color token
        self.patch_size = 16
        self.patch_size_8 = 8
        self.patch_size_4 = 4
        self.scale = embed_dim ** -0.5
        self.cls = cls
        self.cielab = CIELAB()
        self.q_to_ab = self.cielab.q_to_ab
        self.color_embed_dim = 6*embed_dim
        self.cls_emb = nn.Parameter(torch.randn(1, cls, self.color_embed_dim))  # [1, 313, dim]
        self.pos_color = PosCNN(self.color_embed_dim, self.color_embed_dim, self.q_to_ab)
        self.pos_patch = PosCNN(self.color_embed_dim, self.color_embed_dim)


        # self.per_block = nn.ModuleList(
        #     [Decoder_Block(d_model, d_model//64, d_model*4, drop_path_rate, dpr[i]) for i in range(2)]  # 2
        # )

        d_model = [6*embed_dim, 4*embed_dim, 3*embed_dim]
        self.blocks = nn.ModuleList()
        for i in range(3):
            per_blocks = nn.ModuleList()
            for i_block in range(2):
                per_blocks.append(Decoder_Block(d_model[i], d_model[i]//64, d_model[i]*4, 0, dpr[i]))
            blocks = nn.Module()
            blocks.per_blocks = per_blocks
            self.blocks.append(blocks)

        self.concat_down_4C = nn.Linear(4*embed_dim, 2*embed_dim)
        self.concat_down_2C = nn.Linear(2*embed_dim, embed_dim)
        self.concat_down_C = nn.Linear(embed_dim, embed_dim)

        self.expand_4C = SkipPatchExpand(input_resolution=(16, 16), dim=6*embed_dim, dim_scale=2, norm_layer=norm_layer)
        self.color_expand_4C = nn.Linear(6*embed_dim, 4*embed_dim)

        self.expand_2C = SkipPatchExpand(input_resolution=(32, 32), dim=4*embed_dim, dim_scale=2, norm_layer=norm_layer)
        self.color_expand_2C = nn.Linear(4 * embed_dim, 3*embed_dim)

        self.proj_patch_other = nn.Parameter(self.scale * torch.randn(embed_dim*4, embed_dim*4))
        self.proj_classes_other = nn.Parameter(self.scale * torch.randn(embed_dim*4, embed_dim*4))

        self.proj_patch = nn.Parameter(self.scale * torch.randn(3*embed_dim, 3*embed_dim))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(3*embed_dim, 3*embed_dim))
        self.mask_norm = nn.LayerNorm(self.cls)

        self.default_cielab = CIELAB()
        self.encode_ab = SoftEncodeAB(self.default_cielab)
        self.decode_q = AnnealedMeanDecodeQ(self.default_cielab, T=0.38)
        self.class_rebal_lambda = 0.5
        self.get_class_weights = GetClassWeights(self.default_cielab,
                                                 lambda_=self.class_rebal_lambda)
        self.rebalance_loss = RebalanceLoss.apply

        self.outfeature_norm = nn.Conv2d(embed_dim, 2, kernel_size=3, stride=1, padding=1, bias=True)

        # decoder attention block
        dpr_attn = [x.item() for x in torch.linspace(0, drop_path_rate, 3)]
        # 4C
        self.concat_linear_4C_x = nn.Linear(embed_dim*8, embed_dim*4)
        self.concat_linear_4C_down_features = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.concat_linear_4C_regression_patches = nn.Linear(embed_dim * 8, embed_dim * 4)

        self.norm1_4C = nn.LayerNorm(embed_dim*4)
        self.norm2_4C = nn.LayerNorm(embed_dim*4)
        self.mlp_4C = FeedForward(embed_dim*4, embed_dim*16, 0.1)
        self.drop_path_4C = DropPath(dpr_attn[0]) if dpr_attn[0] > 0.0 else nn.Identity()

        # 2C
        self.concat_linear_2C = nn.Linear(embed_dim * 4, embed_dim * 2)

        self.norm1_2C = nn.LayerNorm(embed_dim * 6)
        self.norm2_2C = nn.LayerNorm(embed_dim * 6)
        self.mlp_2C = FeedForward(embed_dim * 6, embed_dim*12, 0.1)
        self.drop_path_2C = DropPath(dpr_attn[1]) if dpr_attn[1] > 0.0 else nn.Identity()

        # C
        self.concat_linear_C = nn.Linear(embed_dim * 2, embed_dim)

        self.norm1_C = nn.LayerNorm(7*embed_dim)
        self.norm2_C = nn.LayerNorm(7*embed_dim)
        self.mlp_C = FeedForward(7*embed_dim, embed_dim*14, 0.1)
        self.drop_path_C = DropPath(dpr_attn[2]) if dpr_attn[2] > 0.0 else nn.Identity()

        # with_regression
        self.with_regression = with_regression

        self.norm_x_color = nn.LayerNorm(embed_dim*4)

        d_model = embed_dim*4

        self.upsampler = nn.Sequential(
            conv3x3_bn_relu(d_model, d_model),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            conv3x3_bn_relu(d_model, d_model),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            conv3x3_bn_relu(d_model, d_model),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            conv3x3_bn_relu(d_model, d_model),
            nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
        )

        self.modelout = nn.Conv2d(d_model, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.upsampler_l1 = nn.Sequential(
            conv3x3_bn_relu(d_model, d_model),
            nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            conv3x3_bn_relu(d_model // 2, d_model // 2),
            nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            conv3x3_bn_relu(d_model // 4, d_model // 4),
            nn.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            conv3x3_bn_relu(d_model // 8, d_model // 8),
            nn.ConvTranspose2d(d_model // 8, d_model // 8, kernel_size=4, stride=2, padding=1, bias=True),
            conv3x3(d_model // 8, 2)
        )

        self.tanh = nn.Tanh()

        self.conv_layers = nn.Conv2d(3*embed_dim, 3*embed_dim, kernel_size=3, stride=1, padding=1)
        self.color_linear = nn.Linear(3*embed_dim, 3*embed_dim)



        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # build decoder layers
        self.layer_up_0 = PatchExpand(input_resolution=(8, 8), dim=8*embed_dim, dim_scale=2, norm_layer=norm_layer)


        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(3*self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)
        self.up_final = nn.Sequential(
            conv3x3_bn_relu(3*embed_dim, 3*embed_dim),
            nn.ConvTranspose2d(3*embed_dim, 3*embed_dim, kernel_size=4, stride=2, padding=1, bias=True),
            conv3x3_bn_relu(3*embed_dim, 3*embed_dim),
            nn.ConvTranspose2d(3*embed_dim, 3*embed_dim, kernel_size=4, stride=2, padding=1, bias=True),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # B L C
  
        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample, input_mask):
        # 计算三层color attention用到的mask
        process_mask = self.calculate_mask(input_mask)
        process_mask_32 = self.calculate_mask_32(input_mask)
        process_mask_64 = self.calculate_mask_64(input_mask)

        for inx in range(4):
            if inx == 0:
                # patch expand
                x = self.layer_up_0(x)
            if inx == 1:
                # concat forward_down_features

                x = torch.cat([x, self.concat_down_4C(x_downsample[3 - inx])], -1)  # on 6C

                # linear
                # x = self.concat_linear_4C_x(x)  # on

                # x = self.concat_back_dim[inx](x)  # 8C
                # h/16 加入color tokens [B, N+313, 4C]
                pos_h, pos_w = 23, 23
                cls_emb = self.cls_emb.expand(x.size(0), -1, -1)  # B, 313, C
                cls_emb = self.pos_color(cls_emb, pos_h, pos_w)  # cpvt for color tokens.
                GS = int(math.sqrt(x.size(1)))  # 16
                x = self.pos_patch(x, GS, GS)  # cpvt for patch tokens.
                x = torch.cat((x, cls_emb), 1)  # B, 16*16+313, 6C

                # color transformer 计算attn
                # print(x.shape)  # [B, 16*16+313, 768]
                for i_block in range(2):
                    x = self.blocks[inx-1].per_blocks[i_block](x, mask=process_mask, without_colorattn=False)

                # split x, c_token
                x, color_token = x[:, :-self.cls, :], x[:, -self.cls:, :]

                # skip expand
                x = self.expand_4C(x)  # [B, 16*16, 6C] -> [B, 32*32, 3C]
                color_token = self.color_expand_4C(color_token)  # [6C] -> [4C]

            if inx == 2:
                # concat forward_down_features
                x = torch.cat([x, self.concat_down_2C(x_downsample[3 - inx])], -1)  # on 4C
                # linear
                # x = self.concat_linear_2C(x)  # on
                # x = self.concat_back_dim[inx](x)
                # h/8 concat color tokens
                x = torch.cat((x, color_token), 1)
                # attn
                for i_block in range(2):
                    x = self.blocks[inx - 1].per_blocks[i_block](x, mask=process_mask_32, without_colorattn=False)
                # split x, c_token
                x, color_token = x[:, :-self.cls, :], x[:, -self.cls:, :]
                # expand
                x = self.expand_2C(x)  # [B, 32, 32, 4C] -> [B, 64, 64, 2C]
                color_token = self.color_expand_2C(color_token)  # [4C] -> [3C]

            if inx == 3:
                # concat forward_down_features

                x = torch.cat([x, self.concat_down_C(x_downsample[3 - inx])], -1)  # on 3C

                # linear
                # x = self.concat_linear_C(x)  # on
                # h/4 concat color tokens
                x = torch.cat((x, color_token), 1)  # 3C
                # attn*2
                for i_block in range(2):
                    x = self.blocks[inx - 1].per_blocks[i_block](x, mask=process_mask_64, without_colorattn=False)
                # split x, c_token
                x, color_token = x[:, :-self.cls, :], x[:, -self.cls:, :]
                patches_64 = x  # [B, 64, 64, 3*C]
                # expand [B, h, w, C] [B, 313, C]
                # patches = self.expand_C(x)  # [B, h/4, w/4, C]  [B, 64, 64, C]
                # color_token = self.color_expand_C(color_token)  # [B, 313, C]


            # else: #
            #     x = torch.cat([patches,x_downsample[3-inx]],-1)
            #     x = self.concat_back_dim[inx](x)
            #     x = layer_up(x)

        # x = self.norm_up(x)  # B L 3C
        B, _, C = x.shape
        # patch, color = x[:, :-313], x[:, -313:]
        patch_h = patch_w = int(math.sqrt(x.size(1)))  # 64
        patch = x.contiguous().view(B, patch_h, patch_w, C).permute(0, 3, 1, 2)  # [B, 3*192, h, w]
        patch = self.conv_layers(patch).contiguous()  # conv after per transformer block for patch.
        color = self.color_linear(color_token)  # linear after per transformer blocks for color.
        patch = patch.view(B, C, patch_h * patch_w).transpose(1, 2)
        x = torch.cat((patch, color), dim=1)
        x = self.norm_up(x)

        x, color_token = x[:, :-self.cls, :], x[:, -self.cls:, :]
  
        return x, color_token, patches_64

    def up_x4(self, x):
        H, W = self.patches_resolution  # H//patch_size = 256/4 = 64
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = x.contiguous().view(B, H, W, 3*self.embed_dim).permute(0, 3, 1, 2)
            x = self.up_final(x)  # B,3C,256,256

            # x = self.up(x)  # 这个需要加吗

            # x = x.view(B,4*H,4*W,-1)  # [B, 256, 256, C]
            # x = x.permute(0,3,1,2) #B,C,H,W
        return x

    def forward(self, x, img_size, gt_ab, input_mask=None):
        H, W = img_size
        x, x_downsample = self.forward_features(x)
        down_features = self.layer_up_0(x)  # [B, 16*16, 4*C]


        x, color_token, patches_64 = self.forward_up_features(x,x_downsample, input_mask)

        x = self.up_x4(x)  # [B, 64*64, C]->[B, C, 256, 256]
        B, C, H_x, W_x = x.shape
        x = x.view(B, 3*self.embed_dim, H_x * W_x).transpose(1, 2).contiguous()  # [B, 256*256, 3C]
        patches = x
        # print("x.shape",x.shape)

        x = x @ self.proj_patch
        color_token = color_token @ self.proj_classes

        x = x / x.norm(dim=-1, keepdim=True)
        color_token = color_token/color_token.norm(dim=-1, keepdim=True)

        masks = x @ color_token.transpose(1, 2)  # [B, N, 313]
        masks = self.mask_norm(masks)

        new_mask = input_mask

        masks = masks.masked_fill(new_mask == 0, -float('inf'))

        masks = rearrange(masks, "b (h w) n -> b n h w", h=H)  # [B, 313, 256, 256]

        # regression
        # if self.with_regression:
            # print("regression:true")



        q_pred = masks  # multi-scaled, [B, 313, 256, 256]
        # print("q_pred:", q_pred)
        q_actual = self.encode_ab(gt_ab)
        # print("q_actual:", q_actual)
        # rebalancing
        color_weights = self.get_class_weights(q_actual)
        q_pred = self.rebalance_loss(q_pred, color_weights)
        ab_pred = self.decode_q(q_pred)  # softmax to [0, 1]

        # patch16 upsampler
        # concat forward_down_features
        down_features = torch.cat([down_features, x_downsample[2]], -1)
        # linear
        down_features = self.concat_linear_4C_down_features(down_features)
        out_features = down_features.contiguous().view(B, 16, 16, 4*self.embed_dim).permute(0, 3, 1, 2)
        out_features = self.upsampler_l1(out_features)
        out_features = self.tanh(out_features)

        return ab_pred, q_pred, q_actual, out_features


    def inference(self, x, img_size, gt_ab, input_mask=None):
        H, W = img_size
        x, x_downsample = self.forward_features(x)
        down_features = self.layers_up[0](x)  # [B, 16*16, 4*C]

        x, color_token, patches_64 = self.forward_up_features(x, x_downsample, input_mask)

        x = self.up_x4(x)  # [B, 64*64, C]->[B, C, 256, 256]
        B, C, H_x, W_x = x.shape
        x = x.view(B, 7 * self.embed_dim, H_x * W_x).transpose(1, 2).contiguous()  # [B, 256*256, C]
        patches = x
        # print("x.shape",x.shape)

        x = x @ self.proj_patch
        color_token = color_token @ self.proj_classes

        x = x / x.norm(dim=-1, keepdim=True)
        color_token = color_token / color_token.norm(dim=-1, keepdim=True)

        masks = x @ color_token.transpose(1, 2)  # [B, N, 313]
        masks = self.mask_norm(masks)

        new_mask = input_mask

        masks = masks.masked_fill(new_mask == 0, -float('inf'))

        masks = rearrange(masks, "b (h w) n -> b n h w", h=H)  # [B, 313, 256, 256]

        # regression
        # if self.with_regression:
        # print("regression:true")

        q_pred = masks  # multi-scaled, [B, 313, 256, 256]
        # print("q_pred:", q_pred)
        q_actual = self.encode_ab(gt_ab)
        # print("q_actual:", q_actual)
        # rebalancing
        # color_weights = self.get_class_weights(q_actual)
        # q_pred = self.rebalance_loss(q_pred, color_weights)
        ab_pred = self.decode_q(q_pred)  # softmax to [0, 1]

        # patch16 upsampler
        # concat forward_down_features
        down_features = torch.cat([down_features, x_downsample[2]], -1)
        # linear
        down_features = self.concat_linear_4C_down_features(down_features)
        out_features = down_features.contiguous().view(B, 16, 16, 4 * self.embed_dim).permute(0, 3, 1, 2)
        out_features = self.upsampler_l1(out_features)
        out_features = self.tanh(out_features)

        if self.with_regression:
            return ab_pred, q_pred, q_actual, out_features
        else:
            return ab_pred, q_pred, q_actual, out_features

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

    def calculate_mask(self, mask):
        # mask: [B, 256x256, 313]-> [B, 16x16+313, 16x16+313]
        B, N, n_cls = mask.size()
        H = W = int(math.sqrt(N))       # H=W=256
        process_mask = mask.view(B, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size, n_cls)  # [B, 16, 16, 16, 16, 313]
        # permute -> [B, 16, 16, 16, 16, 313]
        # view -> [B, 256, 256, 313]
        process_mask = process_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, (H//self.patch_size) * (W//self.patch_size), self.patch_size*self.patch_size, n_cls)
        process_mask = torch.sum(process_mask, dim=2)  # [B, 256, 313]
        mask_t = process_mask.transpose(1, 2)   # [B, 313, 16x16]
        mask_p = torch.ones((B, H//self.patch_size* W//self.patch_size, H//self.patch_size* W//self.patch_size)).to(process_mask.device)  # [B, 256, 256]
        mask_c = torch.ones(B, n_cls, n_cls).to(process_mask.device)  # [B, 313, 313]
        mask_p = torch.cat((mask_p, process_mask), dim=-1)   # [B, 256, 256] +  [B, 256, 313]->[B, 16x16, 16x16+313]
        mask_c = torch.cat((mask_t, mask_c), dim=-1)    # [B, 313, 16x16] + [B, 313, 313]->[B, 313, 16x16+313]
        process_mask = torch.cat((mask_p, mask_c), dim=1)       # [B, 16x16+313, 16x16+313]
        return process_mask

    def calculate_mask_32(self, mask):
        # mask: [B, 256x256, 313]-> [B, 32x32+313, 32x32+313]
        B, N, n_cls = mask.size()
        H = W = int(math.sqrt(N))       # H=W=256
        process_mask = mask.view(B, H//self.patch_size_8, self.patch_size_8, W//self.patch_size_8, self.patch_size_8, n_cls)  # [B, 16, 16, 16, 16, 313]
        # permute -> [B, 32, 8, 32, 8, 313]
        # view -> [B, 32*32, 8*8, 313]
        process_mask = process_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, (H//self.patch_size_8) * (W//self.patch_size_8), self.patch_size_8*self.patch_size_8, n_cls)
        process_mask = torch.sum(process_mask, dim=2)  # [B, 32*32, 313]
        mask_t = process_mask.transpose(1, 2)   # [B, 313, 16x16]
        mask_p = torch.ones((B, H//self.patch_size_8* W//self.patch_size_8, H//self.patch_size_8* W//self.patch_size_8)).to(process_mask.device)  # [B, 32*32, 32*32]
        mask_c = torch.ones(B, n_cls, n_cls).to(process_mask.device)  # [B, 313, 313]
        mask_p = torch.cat((mask_p, process_mask), dim=-1)   # [B, 256, 256] +  [B, 256, 313]->[B, 16x16, 16x16+313]
        mask_c = torch.cat((mask_t, mask_c), dim=-1)    # [B, 313, 16x16] + [B, 313, 313]->[B, 313, 16x16+313]
        process_mask_32 = torch.cat((mask_p, mask_c), dim=1)       # [B, 16x16+313, 16x16+313]
        return process_mask_32

    def calculate_mask_64(self, mask):
        # mask: [B, 256x256, 313]-> [B, 16x16+313, 16x16+313]
        B, N, n_cls = mask.size()
        H = W = int(math.sqrt(N))       # H=W=256
        process_mask = mask.view(B, H//self.patch_size_4, self.patch_size_4, W//self.patch_size_4, self.patch_size_4, n_cls)  # [B, 16, 16, 16, 16, 313]
        # permute -> [B, 16, 16, 16, 16, 313]
        # view -> [B, 256, 256, 313]
        process_mask = process_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, (H//self.patch_size_4) * (W//self.patch_size_4), self.patch_size_4*self.patch_size_4, n_cls)
        process_mask = torch.sum(process_mask, dim=2)  # [B, 256, 313]
        mask_t = process_mask.transpose(1, 2)   # [B, 313, 16x16]
        mask_p = torch.ones((B, H//self.patch_size_4* W//self.patch_size_4, H//self.patch_size_4* W//self.patch_size_4)).to(process_mask.device)  # [B, 256, 256]
        mask_c = torch.ones(B, n_cls, n_cls).to(process_mask.device)  # [B, 313, 313]
        mask_p = torch.cat((mask_p, process_mask), dim=-1)   # [B, 256, 256] +  [B, 256, 313]->[B, 16x16, 16x16+313]
        mask_c = torch.cat((mask_t, mask_c), dim=-1)    # [B, 313, 16x16] + [B, 313, 313]->[B, 313, 16x16+313]
        process_mask_64 = torch.cat((mask_p, mask_c), dim=1)       # [B, 16x16+313, 16x16+313]
        return process_mask_64
