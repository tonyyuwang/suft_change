import torch
import math

a = torch.randn(2, 256, 192)
b = torch.randn(3, 313, 192)
c = torch.cat((a, b),dim=2)
d = c[:, :-313]
e = c[:, -313:]
print(d.shape)
print(e.shape)