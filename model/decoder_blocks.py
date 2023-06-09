import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from timm.models.layers import DropPath

from timm.models.layers import trunc_normal_

# from model.blocks import Block, FeedForward, Decoder_Block, Decoder_Block_Color, Multiscale_Block
from model.utils import init_weights, CIELAB
from engine import functional_conv2d

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

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class Decoder_Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads  # 3
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, without_colorattn=False):
        if not without_colorattn:
            B, N, C = x.shape       # x: [B, 16*16+313, C]
            # print("--------------------------")
            # print("x.C:", C)
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.heads, C // self.heads)
                .permute(2, 0, 3, 1, 4)     # [3, B, self.heads, N, C//heads]
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )
            # q,k,v: [B, heads, 16*16+313, C//heads] , heads = 3

            attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, heads, 16*16+313, 16*16+313]

            if mask is not None:        # add_mask == True
                expand_mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)     # [B, heads, 16*16+313, 16*16+313]
                attn = attn.masked_fill(expand_mask == 0, -float('inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            assert without_colorattn is True
            B, N, C = x.shape
            n_cls = 313
            p_num = N - n_cls
            patch_tokens, color_tokens = x[:, :-n_cls, :], x[:, p_num:, :]
            qkv = (self.qkv(patch_tokens).reshape(B, p_num, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4))
            q, k, v = (qkv[0], qkv[1], qkv[2])
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            patches = (attn @ v).transpose(1, 2).reshape(B, p_num, C)
            patches = self.proj(patches)
            patches = self.proj_drop(patches)
            x = torch.cat((patches, color_tokens), dim=1)       # [B, N, C]

        return x, attn


class Decoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Decoder_Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False, without_colorattn=False):
        y, attn = self.attn(self.norm1(x), mask, without_colorattn=without_colorattn)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x