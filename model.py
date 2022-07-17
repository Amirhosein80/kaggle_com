import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import torchinfo
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import Permute
from model_utils import *


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features
        self.encoder = IntermediateLayerGetter(self.encoder, return_layers={"1": "feature0",
                                                                            "3": "feature1",
                                                                            "5": "feature2",
                                                                            "7": "bottelneck"})
        self.permute = Permute([0, 2, 3, 1])
    def forward(self, image):
        features = self.encoder(image)
        features["feature0"] = self.permute(features["feature0"])
        features["feature1"] = self.permute(features["feature1"])
        features["feature2"] = self.permute(features["feature2"])
        features["bottelneck"] = self.permute(features["bottelneck"])
        return features


class Decoder(nn.Module):
    def __init__(self, channels_list=[96, 192, 384, 768], dim=96, num_classes=3):
        super().__init__()
        self.conv1 = ConvNextBlock(channels_list[0], kernel_size=7, padding=3, dilation=1)
        self.conv2 = ConvNextBlock(channels_list[1], kernel_size=7, padding=3, dilation=1)
        self.conv3 = ConvNextBlock(channels_list[2], kernel_size=7, padding=3, dilation=1)

        self.upconv3 = ConvPermute(channels_list[3], channels_list[2], kernel_size=1, stride=1, padding=0, bias=True)
        self.upconv2 = ConvPermute(channels_list[2], channels_list[1], kernel_size=1, stride=1, padding=0, bias=True)
        self.upconv1 = ConvPermute(channels_list[1], channels_list[0], kernel_size=1, stride=1, padding=0, bias=True)

        self.dcconv3 = ConvPermute(channels_list[2] * 2, channels_list[2], kernel_size=3, stride=1, padding=1,
                                   bias=True)
        self.dcconv2 = ConvPermute(channels_list[1] * 2, channels_list[1], kernel_size=3, stride=1, padding=1,
                                   bias=True)
        self.dcconv1 = ConvPermute(channels_list[0] * 2, channels_list[0], kernel_size=3, stride=1, padding=1,
                                   bias=True)

        self.conv33 = ConvNextBlock(channels_list[2], kernel_size=7, padding=3, dilation=1)
        self.conv22 = ConvNextBlock(channels_list[1], kernel_size=7, padding=3, dilation=1)
        self.conv11 = ConvNextBlock(channels_list[0], kernel_size=7, padding=3, dilation=1)

        self.head3 = nn.Sequential(ConvNextBlock(channels_list[2], kernel_size=7, padding=3, dilation=1),
                                   nn.Linear(channels_list[2], dim))
        self.head2 = nn.Sequential(ConvNextBlock(channels_list[1], kernel_size=7, padding=3, dilation=1),
                                   nn.Linear(channels_list[1], dim))
        self.head1 = nn.Sequential(ConvNextBlock(channels_list[0], kernel_size=7, padding=3, dilation=1),
                                   nn.Linear(channels_list[0], dim))

        self.head = nn.Sequential(ConvPermute(dim * 3, dim, kernel_size=3, stride=1, padding=1, bias=True),
                                  ConvNextBlock(dim, kernel_size=7, padding=3, dilation=1),
                                  nn.Linear(dim, num_classes))

        self.permute1 = Permute([0, 3, 1, 2])
        self.permute2 = Permute([0, 2, 3, 1])

    def forward(self, features):
        feats = OrderedDict()
        x = self.permute1(features["bottelneck"])  # 8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.permute2(x)
        x = self.upconv3(x)
        x = torch.cat([x, self.conv3(features["feature2"])], dim=-1)
        x = self.dcconv3(x)
        x = self.conv33(x)

        x3 = x.clone()
        x3 = self.head3(x3)
        feats["feature3"] = x3
        x3 = self.permute1(x3)
        x3 = F.interpolate(x3, scale_factor=8, mode="bilinear", align_corners=False)
        x3 = self.permute2(x3)

        x = self.permute1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.permute2(x)
        x = self.upconv2(x)
        x = torch.cat([x, self.conv2(features["feature1"])], dim=-1)
        x = self.dcconv2(x)
        x = self.conv22(x)

        x2 = x.clone()
        x2 = self.head2(x2)
        feats["feature2"] = x2
        x2 = self.permute1(x2)
        x2 = F.interpolate(x2, scale_factor=4, mode="bilinear", align_corners=False)
        x2 = self.permute2(x2)

        x = self.permute1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.permute2(x)
        x = self.upconv1(x)
        x = torch.cat([x, self.conv1(features["feature0"])], dim=-1)
        x = self.dcconv1(x)
        x = self.conv11(x)

        x1 = x.clone()
        x1 = self.head1(x1)
        feats["feature1"] = x1
        x1 = self.permute1(x1)
        x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.permute2(x1)

        # print(x3.shape, x2.shape, x1.shape)
        x = torch.cat([x1, x2, x3], dim=-1)

        x = self.head(x)
        feats["pred"] = x
        targets = self.permute1(x)
        targets = F.interpolate(targets, scale_factor=2, mode="bilinear", align_corners=False)
        return targets, feats


class Refinement(nn.Module):
    def __init__(self, dim=96, num_classes=3, height_list=[64, 32, 16], width_list=[64, 32, 16]):
        super().__init__()
        self.ra_ra3 = RA_RA(height=height_list[2], width=width_list[2], num_classes=num_classes)
        self.ra_ra2 = RA_RA(height=height_list[1], width=width_list[1], num_classes=num_classes)
        self.ra_ra1 = RA_RA(height=height_list[0], width=width_list[0], num_classes=num_classes)

        self.ra3_head = nn.Sequential(ConvNextBlock(dim=dim, kernel_size=7, padding=3, dilation=1),
                                      nn.Linear(dim, num_classes))
        self.ra2_head = nn.Sequential(ConvNextBlock(dim=dim, kernel_size=7, padding=3, dilation=1),
                                      nn.Linear(dim, num_classes))
        self.ra1_head = nn.Sequential(ConvNextBlock(dim=dim, kernel_size=7, padding=3, dilation=1),
                                      nn.Linear(dim, num_classes))
        self.cfp3 = CFPNexT(dim=dim, dilation_list=[1, 3, 5, 9], padding_list=[2, 6, 10, 18])
        self.cfp2 = CFPNexT(dim=dim, dilation_list=[1, 3, 5, 9], padding_list=[2, 6, 10, 18])
        self.cfp1 = CFPNexT(dim=dim, dilation_list=[1, 3, 5, 9], padding_list=[2, 6, 10, 18])
        self.permute1 = Permute([0, 3, 1, 2])
        self.permute2 = Permute([0, 2, 3, 1])

    def forward(self, features):
        preds = OrderedDict()

        x = features["pred"]
        x = self.permute1(x)
        x = F.interpolate(x, scale_factor=0.125, mode="bilinear", align_corners=False)
        x = self.permute2(x)

        x3 = self.ra_ra3(self.cfp3(features["feature3"]), x)
        x3 = self.ra3_head(x3)
        x3 = x3 + x
        x3 = self.permute1(x3)
        preds["pred3"] = F.interpolate(x3, scale_factor=16, mode="bilinear", align_corners=False)

        # x3 = self.permute1(x3)
        x = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.permute2(x)

        x2 = self.ra_ra2(self.cfp2(features["feature2"]), x)
        x2 = self.ra2_head(x2)
        x2 = x2 + x

        x2 = self.permute1(x2)
        preds["pred2"] = F.interpolate(x2, scale_factor=8, mode="bilinear", align_corners=False)

        # x = self.permute1(x)
        x = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.permute2(x)

        x1 = self.ra_ra1(self.cfp1(features["feature1"]), x)
        x1 = self.ra1_head(x1)
        x1 = x1 + x
        x1 = self.permute1(x1)
        preds["pred1"] = F.interpolate(x1, scale_factor=4, mode="bilinear", align_corners=False)
        return preds


class SwinFormer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=num_classes)
        self.refine = Refinement(num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x, f = self.decoder(x)
        p = self.refine(f)
        p["dpred"] = x
        return p


if __name__ == "__main__":
    torch.cuda.empty_cache()
    img = torch.randn(1, 3, 256, 256)
    sf = SwinFormer(4)
    out = sf(img)
    print(count_parameters(sf))
    print([(k, v.shape) for k, v in out.items()])
    # print(sf(img).shape)
    # print([v.shape for v in sf(img)])
    print(torchinfo.summary(sf, (4, 3, 256, 256), device="cpu",
                            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
                            row_settings=["var_names"], ))
