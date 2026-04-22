import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MPCE"]

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
   
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.depthwise = Conv(in_channels, in_channels, k=kernel_size, s=stride, p=padding, g=in_channels)
        self.pointwise = Conv(in_channels, out_channels, k=1, s=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class MPCE(nn.Module):

    def __init__(self, c1, r=4, *args, **kwargs):
        super().__init__()
        self.conv1 = Conv(c1, c1//4, k=1, s=1, p=0)
        self.conv2 = Conv(c1//2, c1, k=1, s=1, p=0)
        self.conv3 = Conv(c1//4, c1//2, k=1, s=1, p=0)
        self.conv4 = Conv(c1, c1//4, k=1, s=1, p=0)
        self.conv5 = Conv(c1//4, c1//2, k=3, s=1, p=1)
        self.dw3 = DepthwiseSeparableConv(c1//4, c1//4, 3, 1, 1)
        self.dw5 = DepthwiseSeparableConv(c1//4, c1//4, 5, 2, 1)
      
        self.w = nn.Parameter(torch.ones(2)) 

    def forward(self, x):
        x_first = self.conv1(x)
        x1 = self.dw3(x_first)
        x2 = self.dw5(x_first)

 
        w = torch.softmax(self.w, dim=0)

        x_fuse = w[0] * x1 + w[1] * x2 + x_first
        x11 = self.conv3(x_fuse)
        x12 = self.conv5(self.conv4(x))
        x111 = x11 + x12
        xx = self.conv2(x111)
        return xx

