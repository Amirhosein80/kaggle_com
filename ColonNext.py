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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features
        self.encoder = IntermediateLayerGetter(self.encoder, return_layers={
            "3": "feature2",  # 40
            "5": "feature3",  # 20
            "7": "feature4"})  # 10

    def forward(self, image):
        return self.encoder(image)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_gelu=False):
        super().__init__()
        self.use_gelu = use_gelu
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])
        if self.use_gelu:
            self.gelu = nn.GELU()

    def forward(self, x):
        x = self.permute1(x)
        x = self.linear(x)
        if self.use_gelu:
            x = self.gelu(x)
        x = self.permute2(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, features, ):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.permute1(x)
        x = self.norm(x)
        x = self.permute2(x)
        return x


class CNBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3, dilation=1, layer_scale=1e-6, stochastic_depth_prob=0.1):
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

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.permute1(x)
        x = self.gelu(x)
        x = self.permute2(x)
        return x

class CFP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cfp1_1 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1), GELU())
        self.cfp1_2 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1), GELU())
        self.cfp1_3 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1), GELU())

        self.cfp2_1 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3), GELU())
        self.cfp2_2 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3), GELU())
        self.cfp2_3 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3), GELU())

        self.cfp3_1 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=5, dilation=5), GELU())
        self.cfp3_2 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=5, dilation=5), GELU())
        self.cfp3_3 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=5, dilation=5), GELU())

        self.cfp4_1 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=9, dilation=9), GELU())
        self.cfp4_2 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=9, dilation=9), GELU())
        self.cfp4_3 = nn.Sequential(LayerNormalization(dim),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=9, dilation=9), GELU())

        self.cat1 = Linear(dim * 3, dim)
        self.cat2 = Linear(dim * 3, dim)
        self.cat3 = Linear(dim * 3, dim)
        self.cat4 = Linear(dim * 3, dim)
        self.cat = Linear(dim * 4, dim)

    def forward(self, x):
        x1_1 = self.cfp1_1(x)
        x1_2 = self.cfp1_2(x1_1)
        x1_3 = self.cfp1_3(x1_2)
        x1 = self.cat1(torch.cat([x1_1, x1_2, x1_3], dim=1))
        del x1_1, x1_2, x1_3

        x2_1 = self.cfp2_1(x)
        x2_2 = self.cfp2_2(x2_1)
        x2_3 = self.cfp2_3(x2_2)
        x2 = self.cat2(torch.cat([x2_1, x2_2, x2_3], dim=1))
        x2 += x1
        del x2_1, x2_2, x2_3

        x3_1 = self.cfp3_1(x)
        x3_2 = self.cfp3_2(x3_1)
        x3_3 = self.cfp3_3(x3_2)
        x3 = self.cat3(torch.cat([x3_1, x3_2, x3_3], dim=1))
        x3 += x2
        del x3_1, x3_2, x3_3

        x4_1 = self.cfp4_1(x)
        x4_2 = self.cfp4_2(x4_1)
        x4_3 = self.cfp4_3(x4_2)
        x4 = self.cat4(torch.cat([x4_1, x4_2, x4_3], dim=1))
        x4 += x3
        del x4_1, x4_2, x4_3

        return x + self.cat(torch.cat([x1, x2, x3, x4], dim=1))


class SelfAtten(nn.Module):
    def __init__(self, dim, dim_scale=2):
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


class AxialAtten(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, dim_scale=8):
        super().__init__()
        self.height_atten = SelfAtten(dim=height, dim_scale=dim_scale)
        self.width_atten = SelfAtten(dim=width, dim_scale=dim_scale)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels, bias=True),
            LayerNormalization(in_channels),
            Linear(in_channels, out_channels, use_gelu=True))
        self.conv_weight = nn.Sequential(LayerNormalization(out_channels),
                                         Linear(out_channels, out_channels // 2, use_gelu=True),
                                         Linear(out_channels // 2, out_channels, use_gelu=False),
                                         nn.Sigmoid())
        self.height_weights = nn.Parameter(torch.zeros(1))
        self.width_weights = nn.Parameter(torch.zeros(1))
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.conv(x)
        x = x * self.conv_weight(x)
        x = self.permute1(x)
        B, H, W, C = x.shape
        x_ = x.clone()
        xh = x.permute(0, 2, 3, 1).view(B, -1, H)
        xh = self.height_atten(xh)
        xh = xh.view(B, W, C, H).permute(0, 3, 1, 2)
        x = (xh * self.height_weights) + x
        xw = xh.permute(0, 1, 3, 2).reshape(B, -1, W)
        xw = self.width_atten(xw)
        xw = xw.view(B, H, C, W).permute(0, 1, 3, 2)
        x = (xw * self.width_weights) + x
        x = x + x_
        x = self.permute2(x)
        del xh, xw, x_
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels_list, num_classes=4):
        super().__init__()
        self.dim = in_channels_list[0]

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rfb2 = Linear(in_channels_list[1], self.dim)
        self.rfb3 = Linear(in_channels_list[2], self.dim)
        self.rfb4 = Linear(in_channels_list[3], self.dim)

        self.cnblock1_1 = CNBlock(self.dim)
        self.cnblock1_2 = CNBlock(self.dim)
        self.cnblock1_3 = CNBlock(self.dim)
        self.cnblock1_4 = CNBlock(self.dim)
        self.cnblock1_5 = CNBlock(self.dim * 2)

        self.cnblock2_1 = CNBlock(self.dim * 2)
        self.cnblock2_2 = CNBlock(self.dim * 3)

        self.cnblock3 = CNBlock(self.dim * 3)
        self.head = Linear(self.dim * 3, num_classes)

    def forward(self, inputs):
        features = OrderedDict()
        x3 = self.rfb2(inputs["feature2"])
        x2 = self.rfb3(inputs["feature3"])
        x1 = self.rfb4(inputs["feature4"])

        x2_1 = self.cnblock1_1(self.up2(x1)) * x2
        x3_1 = self.cnblock1_2(self.up2(self.up2(x1))) * self.cnblock1_3(self.up2(x2)) * x3

        x2_2 = torch.cat((x2_1, self.cnblock1_4(self.up2(x1))), 1)
        x2_2 = self.cnblock2_1(x2_2)

        x3_2 = torch.cat((x3_1, self.cnblock1_5(self.up2(x2_2))), 1)
        x3_2 = self.cnblock2_2(x3_2)

        x = self.cnblock3(x3_2)
        x = self.head(x)

        features["f2"] = x3
        features["f3"] = x2
        features["f4"] = x1
        features["x"] = x

        return features


class Refinement(nn.Module):
    def __init__(self, dim, num_classes, heights=[32, 16, 8], widths=[32, 16, 8]):
        super().__init__()
        self.cfp4 = CFP(dim=dim)
        self.aa_4 = AxialAtten(dim, dim, height=heights[2], width=widths[2])
        self.post4 = Linear(num_classes, 1)
        self.head4 = nn.Sequential(CNBlock(dim), Linear(dim, num_classes))

        self.cfp3 = CFP(dim=dim)
        self.aa_3 = AxialAtten(dim, dim, height=heights[1], width=widths[1])
        self.post3 = Linear(num_classes, 1)
        self.head3 = nn.Sequential(CNBlock(dim), Linear(dim, num_classes))

        self.cfp2 = CFP(dim=dim)
        self.aa_2 = AxialAtten(dim, dim, height=heights[0], width=widths[0])
        self.post2 = Linear(num_classes, 1)
        self.head2 = nn.Sequential(CNBlock(dim), Linear(dim, num_classes))

        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        outputs = OrderedDict()
        x4 = self.cfp4(features["f4"])
        x4 = self.aa_4(x4)
        x4 *= -1 + self.sigmoid(self.post4(F.interpolate(features["x"], scale_factor=0.25, mode='bilinear')))
        x4 = self.head4(x4)
        x4 += F.interpolate(features["x"], scale_factor=0.25, mode='bilinear')

        x3 = self.cfp3(features["f3"])
        x3 = self.aa_3(x3)
        x3 *= -1 + self.sigmoid(self.post3(F.interpolate(x4, scale_factor=2, mode='bilinear')))
        x3 = self.head3(x3)
        x3 += F.interpolate(x4, scale_factor=2, mode='bilinear')

        x2 = self.cfp2(features["f2"])
        x2 = self.aa_2(x2)
        x2 *= -1 + self.sigmoid(self.post2(F.interpolate(x3, scale_factor=2, mode='bilinear')))
        x2 = self.head2(x2)
        x2 += F.interpolate(x3, scale_factor=2, mode='bilinear')

        outputs["out0"] = F.interpolate(features["x"],scale_factor=8,mode='bilinear')
        outputs["out1"] = F.interpolate(x2, scale_factor=8, mode='bilinear')
        outputs["out2"] = F.interpolate(x3, scale_factor=16, mode='bilinear')
        outputs["out3"] = F.interpolate(x4, scale_factor=32, mode='bilinear')


        return outputs


class ColonNext(nn.Module):
    def __init__(self, num_classes=4, heights=[32, 16, 8], widths=[32, 16, 8]):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(in_channels_list=[96, 192, 384, 768], num_classes=num_classes)
        self.refine = Refinement(dim=96, num_classes=num_classes, heights=heights, widths=widths)

    def forward(self, images):
        f = self.encoder(images)
        f = self.decoder(f)
        return self.refine(f)


if __name__ == "__main__":
    img = torch.randn(2, 3, 320, 320)
    un = ColonNext(num_classes=4, heights=[40, 20, 10], widths=[40, 20, 10])
    out = un(img)
    print(count_parameters(un))
    print([(k, v.shape) for k, v in out.items()])
    # print(un(img).shape)
    # print([v.shape for v in un(img)])
    print(torchinfo.summary(un, (16, 3, 320, 320), device="cpu",
                            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
                            row_settings=["var_names"], ))
