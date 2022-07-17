import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import torchinfo
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import Permute
from torchvision.ops.stochastic_depth import StochasticDepth


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


class SelfAtten(nn.Module):
    def __init__(self, dim, dim_scale=1):
        super().__init__()
        self.q_linear = nn.Linear(dim, dim // dim_scale)
        self.k_linear = nn.Linear(dim, dim // dim_scale)
        self.v_linear = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, x):
        q = self.q_linear(x)
        k = self.q_linear(x)
        v = self.v_linear(x)
        x = torch.bmm(q, k.transpose(-1, -2))
        x = self.sigmoid(x / (self.dim ** 0.5))
        x = torch.bmm(x, v)
        return x


class ConvPermute(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.permute1 = Permute([0, 3, 1, 2])
        self.permute2 = Permute([0, 2, 3, 1])

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.permute1(x)
        x = self.conv(x)
        return self.permute2(x)


class AxialAtten(nn.Module):
    def __init__(self, height, width, dim_scale=1):
        super().__init__()
        self.height_atten = SelfAtten(dim=height, dim_scale=dim_scale)
        self.width_atten = SelfAtten(dim=width, dim_scale=dim_scale)
        self.height_weights = nn.Parameter(torch.zeros(1))
        self.width_weights = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, H, W, C = x.shape
        xh = x.permute(0, 2, 3, 1).view(B, -1, H)
        xh = self.height_atten(xh)
        xh = xh.view(B, W, C, H).permute(0, 3, 1, 2)
        x = (xh * self.height_weights) + x
        xw = x.permute(0, 1, 3, 2).reshape(B, -1, W)
        xw = self.width_atten(xw)
        xw = xw.view(B, H, C, W).permute(0, 1, 3, 2)
        x = (xw * self.width_weights) + x
        return x


class RA_RA(nn.Module):
    def __init__(self, height, width, num_classes, dim_scale=1):
        super().__init__()
        self.axial_atten = AxialAtten(height, width, dim_scale)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = ConvPermute(num_classes, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature, predict):
        feature = feature + self.axial_atten(feature)
        predict = 1 - self.sigmoid(predict)
        predict = self.conv1(predict)
        return predict * feature


class CNBlock(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation, layer_scale=1e-6, stochastic_depth_prob=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class ConvNextBlock(nn.Module):
    def __init__(self, dim, kernel_size, padding, dilation):
        super().__init__()
        self.conv = CNBlock(dim=dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.permute1 = Permute([0, 3, 1, 2])
        self.permute2 = Permute([0, 2, 3, 1])

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.permute1(x)
        x = self.conv(x)
        return self.permute2(x)


class CFPNexT(nn.Module):
    def __init__(self, dim, kernel_size=5, padding_list=[2, 4, 6, 8], dilation_list=[1, 2, 3, 4]):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[0], dilation=dilation_list[0]),
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[0], dilation=dilation_list[0]))

        self.conv2 = nn.Sequential(
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[1], dilation=dilation_list[1]),
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[1], dilation=dilation_list[1]))

        self.conv3 = nn.Sequential(
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[2], dilation=dilation_list[2]),
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[2], dilation=dilation_list[2]))

        self.conv4 = nn.Sequential(
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[3], dilation=dilation_list[3]),
            ConvNextBlock(dim=dim, kernel_size=kernel_size, padding=padding_list[3], dilation=dilation_list[3]))
        self.head = ConvPermute(dim * 4, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x2 = x2 + x1
        x3 = x3 + x2
        x4 = x4 + x3
        x_ = torch.cat([x1, x2, x3, x4], dim=-1)
        x_ = self.head(x_)
        return x + x_
