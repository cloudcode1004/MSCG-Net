import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("MKSP-CA",)

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

class CoordAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.inp = inp
        self.reduction = reduction
        mid = max(8, inp // reduction)
        # channel reduction conv
        self.conv1 = nn.Conv2d(inp, mid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU()
        # for height and width
        self.conv_h = nn.Conv2d(mid, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mid, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, h, w = x.size()
        # pooling: Hx1 and 1xW
        x_h = F.adaptive_avg_pool2d(x, (h, 1))           # (n, c, h, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  # (n, c, w, 1)
        # concat along spatial dim
        y = torch.cat([x_h, x_w], dim=2)                # (n, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_h * a_w

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        # depthwise using Conv wrapper (groups=in_channels)
        self.depthwise = Conv(in_channels, in_channels, k=kernel_size, s=stride, p=padding, g=in_channels)
        self.pointwise = Conv(in_channels, out_channels, k=1, s=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Block(nn.Module):
    def __init__(self, c, kernel_size, padding):
        super().__init__()
        # 保证 hidden >=1
        hidden = max(1, int(c * 3 // 4))
        self.conv1 = Conv(c, hidden, k=1, s=1, p=0, act=True)
        self.conv2 = Conv(hidden, c, k=3, s=1, p=1, act=True)
        self.conv3 = DepthwiseSeparableConv(c, c, kernel_size=kernel_size, padding=padding, stride=1)
    def forward(self, x):
        # residual: x + conv3(conv2(conv1(x)))
        return x + self.conv3(self.conv2(self.conv1(x)))

class MKSP-CA(nn.Module):
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, k=1, s=1, act=True)

        self.small_conv = Conv(c1, c2 // 2, k=1, s=1, act=True)
   
        self.small_enhance = nn.Conv2d(c2 // 2, c2 // 2, kernel_size=3, stride=1, padding=1, groups=c2 // 2, bias=False)
        self.small_enh_bn = nn.BatchNorm2d(c2 // 2)
        self.small_act = nn.SiLU()
    
        self.small_attn = CoordAttention(c2 // 2, reduction=16)

      
        self.Mconv1 = Conv(c2, c2 // 2, k=1, s=1, act=True)
        self.Mconv2 = Conv(c2, c2 // 2, k=1, s=1, act=True)

        self.block1 = Block(c2 // 4, kernel_size=3, padding=1)
        self.block2 = Block(c2 // 4, kernel_size=5, padding=2)
        self.block3 = Block(c2 // 4, kernel_size=7, padding=3)
        self.block4 = Block(c2 // 4, kernel_size=11, padding=5)

 
        self.fuse = Conv(c2, c2, k=1, s=1, act=True)

     
        self.final_attn = CoordAttention(c2, reduction=16)

     
        self.conv2 = Conv(c2, c2, k=1, s=1, act=True)

    def forward(self, x):
     
        x_end = self.conv1(x)

 
        small = self.small_conv(x)                  
        small = self.small_act(self.small_enh_bn(self.small_enhance(small)))

        small = self.small_attn(small)

     
        x1_first = self.Mconv1(x_end)  
        x2_first = self.Mconv2(x_end) 


        x11, x12 = torch.split(x1_first, x1_first.shape[1] // 2, dim=1)  
        x21, x22 = torch.split(x2_first, x2_first.shape[1] // 2, dim=1)

        xx11 = self.block1(x11)
        xx12 = self.block2(x12 + xx11)
        x1_add = xx11 + xx12

        xx21 = self.block3(x21)
        xx22 = self.block4(x22 + xx21)
        x2_add = xx21 + xx22

       
        merged = torch.cat((small, x1_add, x2_add), dim=1)  

       
        fused = self.fuse(merged)        
        fused = self.final_attn(fused)   

        out = self.conv2(fused)          
  
        return out + x_end
