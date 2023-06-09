# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

from model.correct import Correct

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self,
                 config,
                 img_size=224,
                 num_classes=21843,
                 zero_head=True,
                 vis=False,
                 with_regression=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.with_regression = with_regression
        embed_dim = config["model"]["swin"]["embed_dim"]
        print("with_regression", with_regression)

        self.swin_unet = SwinTransformerSys(img_size=config["data"]["img_size"],
                                            patch_size=config["model"]["swin"]["patch_size"],
                                            in_chans=config["model"]["swin"]["in_chans"],
                                            num_classes=self.num_classes,
                                            embed_dim=config["model"]["swin"]["embed_dim"],
                                            depths=config["model"]["swin"]["depths"],
                                            num_heads=config["model"]["swin"]["num_heads"],
                                            window_size=config["model"]["swin"]["window_size"],
                                            mlp_ratio=config["model"]["swin"]["mlp_ratio"],
                                            qkv_bias=config["model"]["swin"]["qkv_bias"],
                                            qk_scale=config["model"]["swin"]["qk_scale"],
                                            drop_rate=config["model"]["drop_rate"],
                                            drop_path_rate=config["model"]["drop_path_rate"],
                                            ape=config["model"]["swin"]["ape"],
                                            patch_norm=config["model"]["swin"]["patch_norm"],
                                            use_checkpoint=config["train"]["use_checkpoint"],
                                            with_regression=self.with_regression)
        # self.correct = Correct(indim=1024, hidden_dim=1024)

    def normalize_l(self, l, to):
        # follow Real-Time/ CIC
        normalized = (l-50)/100.
        return normalized

    def forward(self, l, gt_ab, input_mask=None):
        im = self.normalize_l(l, (-1, 1))  # [-1, 1]
        if im.size()[1] == 1:
            im = im.repeat(1, 3, 1, 1)
        H, W = im.size(2), im.size(3)

        if self.with_regression:
            ab_pred, q_pred, q_actual, out_feature = self.swin_unet(im, (H, W), gt_ab, input_mask)
            correct_ab = self.correct(ab_pred, l)

            return ab_pred, q_pred, q_actual, out_feature, correct_ab
        else:
            ab_pred, q_pred, q_actual, out_feature = self.swin_unet(im, (H, W), gt_ab, input_mask)
            return ab_pred, q_pred, q_actual, out_feature


    def inference(self, l, gt_ab, input_mask=None):
        im = self.normalize_l(l, (-1, 1))  # [-1, 1]
        if im.size()[1] == 1:
            im = im.repeat(1, 3, 1, 1)
        H, W = im.size(2), im.size(3)

        if self.with_regression:
            ab_pred, q_pred, q_actual, out_feature = self.swin_unet.inference(im, (H, W), gt_ab, input_mask)
            correct_ab = self.correct(ab_pred, l)

            return ab_pred, q_pred, q_actual, out_feature, correct_ab
        else:
            ab_pred, q_pred, q_actual, out_feature = self.swin_unet.inference(im, (H, W), gt_ab, input_mask)
            return ab_pred, q_pred, q_actual, out_feature





    def load_from(self, config):
        pretrained_path = config["model"]["pretrain_ckpt"]
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            # if "model"  not in pretrained_dict:
            #     print("---start load pretrained modle by splitting---")
            #     pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            #     for k in list(pretrained_dict.keys()):
            #         if "output" in k:
            #             print("delete key:{}".format(k))
            #             del pretrained_dict[k]
            #     msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
            #     # print(msg)
            #     return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            print("swinunet.state_dict")
            # print(self.swin_unet.state_dict())
            full_dict = copy.deepcopy(pretrained_dict)
            # for k, v in pretrained_dict.items():
            #     if "layers." in k:
            #         current_layer_num = 3-int(k[7:8])
            #         current_k = "layers_up." + str(current_layer_num) + k[8:]
            #         full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

    def load(self, args):
        pretrained_path = args.pretrain_ckpt  # config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            # print("pretrained_dict.keys()==",pretrained_dict.keys())  #dict_keys(['model'])
            # 用dict.keys()输出字典元素所有的键

            if "model" not in pretrained_dict:  # 正常情况下都有，不执行这里
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        # print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return

            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of SwinTransformer encoder---")
            # print("pretrained_dict['model']==",pretrained_dict.keys())
            # 此时pretrained_dict包含swin_transformer的全部权重keys-------------------------------
            # 对于tiny其中的[2,2,6,2]encoder，有layers.0.blocks.0.到layers.2.blocks.5.到layers.3.blocks.2.
            model_dict = self.swin_unet.state_dict()
            # 此时model_dict包含swin_unet的全部权重-------------------------------
            # 接下来应该是一一对应赋值
            # print("model_dict==", model_dict.keys())

            full_dict = copy.deepcopy(pretrained_dict)
            # copy.deepcopy() 深拷贝=寻常意义的复制。
            # 将被复制对象完全再复制一遍作为独立的新个体单独存在。
            # 改变原有被复制对象不会对已经复制出来的新对象产生影响。
            # 此时full_dict包含swin_transformer的全部权重keys-------------------------------
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    # print("k==",k,k[7:8])
                    # k == layers.0.blocks.0.norm1.weight，包含预训练权重的全部
                    # k[7:8] == layers之后的number
                    current_layer_num = 3 - int(k[7:8])
                    # print("current_layer_num==",current_layer_num)
                    # current_layer_num现有layer=原有layer总数4-现有layer
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    # 将（tiny）encoder[2，2，6，2]对称地映射出一个decoder权重
                    # print("current_k==",current_k)
                    full_dict.update({current_k: v})
                    # 将映射出的decoder,和原本的encoder的所有权重都存在full_dict里
            # print("full_dict(all)==", full_dict.keys())

            for k in list(full_dict.keys()):
                # 遍历full_dict，如果和model_dict（swin_unet）的key相同，就判断尺寸是否匹配，如果不匹配就删除
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:'{}'; pretrain_shape:'{}'; swinunet_shape:'{}'".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
            # print("full_dict(after del)==", full_dict.keys())
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # 正式执行加载权重。strict=False 表示忽略不匹配的网络层参数
            # 除了.layers_up.1/2/3.upsample没加载预训练权重，都加载了
            # print(msg)
        else:
            print("none pretrain_ckpt to load")
 