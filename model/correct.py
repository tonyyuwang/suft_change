import math
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from skimage import color, io

class Correct(nn.Module):
    def __init__(self, indim, hidden_dim, dropout=0.1, out_dim=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.indim = indim
        self.hidden_dim = hidden_dim//4  # hidden_dim=1024

        model = [nn.Conv2d(in_channels=1, out_channels=self.hidden_dim // 2, kernel_size=3, stride=1, padding=1,
                            bias=True), ]

        model1 = [nn.Conv2d(in_channels=1, out_channels=self.hidden_dim//2, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.BatchNorm2d(self.hidden_dim//2), ]
        model1 += [nn.GELU(), ]
        model1 += [nn.Conv2d(self.hidden_dim//2, self.hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.BatchNorm2d(self.hidden_dim), ]
        model1 += [nn.GELU(), ]
        model1 += [nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.GELU(), ]

        model2 = [nn.Conv2d(in_channels=1, out_channels=self.hidden_dim//2, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.BatchNorm2d(self.hidden_dim//2), ]
        model2 += [nn.GELU(), ]
        model2 += [nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.BatchNorm2d(self.hidden_dim), ]
        model2 += [nn.GELU(), ]
        model2 += [nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.GELU(), ]

        model3 = [nn.Conv2d(self.hidden_dim*2, self.hidden_dim*2, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.BatchNorm2d(self.hidden_dim*2), ]
        model3 += [nn.GELU(), ]
        model3 += [nn.Conv2d(self.hidden_dim*2, self.hidden_dim*2, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.GELU(), ]

        model_out = [nn.Conv2d(self.hidden_dim*2, out_dim, kernel_size=3, stride=1, padding=1, bias=True), ]

        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model_out = nn.Sequential(*model_out)

    def forward(self, pred_colors, img_l):
        img_lab = torch.cat((img_l, pred_colors), dim=1)
        img_lab_np = img_lab.cpu()
        # print(img_lab_np.shape)  # (4, 3, 256, 256)
        a = torch.zeros(1, 3, 256, 256)
        # print('a.shape', a.shape)
        for j in range(img_lab_np.size(0)):
            img_rgb = lab_to_rgb(img_lab_np[j].numpy().transpose(1, 2, 0))
            img_lab = torch.tensor(rgb_to_lab(img_rgb).transpose(2, 0, 1))
            img_lab = torch.unsqueeze(img_lab, 0)
            # print('img_lab.shape', img_lab.shape)  # [3, 256, 256]
            a = torch.cat((a, img_lab), dim=0)
        # print('a.shape', a.shape)
        img_l_convert = a[1:, :1, ]
        # print("img_l_convert.shape", img_l_convert.shape)  # [4, 1, 256, 256]
        # cpu to CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_l_convert = img_l_convert.to(device)

        img_l_convert = torch.sub(img_l, img_l_convert)

        pred_d = self.model1(img_l_convert)

        pred_colors_a, pred_colors_b = pred_colors[:, 0, ], pred_colors[:, 1, ]
        theta = torch.atan2(pred_colors_b, pred_colors_a)
        theta = torch.unsqueeze(theta, dim=1)
        # print("theta.shape", theta.shape)
        pred_theta = self.model2(theta)
        # print('pred_theta.shape', pred_theta.shape)

        correct_ab = torch.cat((pred_d, pred_theta), dim=1)
        correct_ab = self.model3(correct_ab)
        correct_ab = self.model_out(correct_ab)

        x = torch.tanh(correct_ab)  # [-1, 1] ???
        d = x[:, 0, :, :]  # d->[-7, 7]  (7 == 5*sqrt(2))
        theta = x[:, 1, :, :]  # theta->[-pi, pi]
        d = torch.unsqueeze(d, 1)
        theta = torch.unsqueeze(theta, 1)
        x = torch.cat((2.5 + d * 2.5, theta * math.pi), dim=1)  # ????
        # print("x", x)
        return x

        # lab_to_rgb




def lab_to_rgb(img):
    # tensor to np
    assert img.dtype == np.float32
    return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)

def rgb_to_lab(img):
    assert img.dtype == np.uint8
    return color.rgb2lab(img).astype(np.float32)
