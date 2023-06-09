import math
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class Regression(nn.Module):
    '''after upsampler Tli[B, H*W, in_dim]->[B, indim, H, W]->Correct_ab[B, H, W, 2] indim=1024, hidden_dim=2048, out_dim=2
    '''
    def __init__(self, indim, hidden_dim, dropout=0.1, out_dim=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.indim = indim
        self.hidden_dim = hidden_dim

        model1 = [nn.Conv2d(indim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),]
        model1 += [nn.BatchNorm2d(hidden_dim),]
        model1 += [nn.GELU(),]
        model1 += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True),]
        model1 += [nn.GELU(),]

        model2 = [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.BatchNorm2d(hidden_dim), ]
        model2 += [nn.GELU(), ]
        model2 += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.GELU(), ]

        model3 = [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.BatchNorm2d(hidden_dim), ]
        model3 += [nn.GELU(), ]
        model3 += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.GELU(), ]

        model_out = [nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_out = nn.Sequential(*model_out)


    def forward(self, x):  # x:[B, 1024, 256, 256]
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model_out(x)
        x = torch.tanh(x)  # [-1, 1] ???
        d = x[:, 0, :, :]  # d->[-7, 7]  (7 == 5*sqrt(2))
        theta = x[:, 1, :, :]  # theta->[-pi, pi]
        d = torch.unsqueeze(d, 1)
        theta = torch.unsqueeze(theta, 1)
        x = torch.cat((3.5+d*3.5, theta*math.pi), dim=1)  # ????
        return x


# 计算真实值Iab_t和预测值Tab的角度theta和距离d
'''Iab_t:[B, 2, 256, 256] Tab:[B, 2, 256, 256] -> ang_dis_t:[B, C, d, theta]->[B, 2, 256, 256]'''
def ang_dis_count(Iab_t, Tab):  # 前为真实 后为预测值
    coo_rel = torch.sub(Iab_t, Tab)
    size = coo_rel.size
    # print(coo_rel.size)

    coo_rel_a, coo_rel_b = coo_rel[:, 0, ], coo_rel[:, 1, ]
    # d = torch.sqrt(coo_rel_a**2 + coo_rel_b**2)
    d = torch.sqrt(torch.square(coo_rel_a) + torch.square(coo_rel_b))
    d = torch.clamp(d, min=0)
    # d = torch.clamp(d, min=0, max=7)
    theta = torch.atan2(coo_rel_b, coo_rel_a)  # 前面除后面 要是分母为0呢??
    d = torch.unsqueeze(d, 1)
    theta = torch.unsqueeze(theta, 1)
    ang_dis_t = torch.cat((d, theta), dim=1)

    return ang_dis_t





def angle(v1, v2):
    dx1 = v1[2] - v1[0]  # 后减前
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def distance(v1, v2):
    d = np.abs(np.sqrt((v1[2]-v2[2])**2 + (v1[3]-v2[3])**2))
    return d

def dis_loss(I_pre, I_ab_t, T_ab):
    '''input_tensor_size is 2*H*W
    '''
    assert I_pre.size == I_ab_t.size == T_ab.size
    B, C, H, W = I_pre.size  # ?????
    loss_ang_dis = 0
    for i in range(H):
        for j in range(W):
            c0 = T_ab[0, i, j]
            c1 = T_ab[1, i, j]
            a0 = I_ab_t[0, i, j]
            a1 = I_ab_t[1, i, j]
            b0 = I_pre[0, i, j]
            b1 = I_pre[1, i, j]
            CA = [c0, c1, a0, a1]
            CB = [c0, c1, b0, b1]
            ang1 = angle(CA, CB)
            ang_pi = (ang1/180)*math.pi
            d = distance(CA, CB)
            loss_ang_dis += ang_pi + d
    return loss_ang_dis

# 将预测ab_pred图像和矫正矩阵ab_correct相加得到最后的预测图像ab_final
def final_ab(ab_pred, ab_correct):
    ab_final = ab_pred + ab_correct
    return  ab_final




