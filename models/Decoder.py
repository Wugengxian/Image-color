import torch.nn as nn
from typing import List
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual
from functools import partial
from models.layers import AttentionBlock, ConvBlock
import torch


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)

        self.attn1 = AttentionBlock(112, 160, 160)
        self.upsample1 = nn.Upsample(mode="bilinear", scale_factor=2)
        inverted_residual_setting1 = [bneck_conf(272, 3, 960, 112, True, "HS", 1, 1),
                                        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                                        bneck_conf(112, 3, 480, 80, True, "HS", 1, 1),
                                        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                                        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                                        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1)]
        layers1: List[nn.Module] = []
        for cnf in inverted_residual_setting1:
            layers1.append(InvertedResidual(cnf, norm_layer))
        self.upConv1 = nn.Sequential(*layers1)

        self.attn2 = AttentionBlock(40, 80, 80)
        self.upsample2 = nn.Upsample(mode="bilinear", scale_factor=2)
        inverted_residual_setting2 = [bneck_conf(120, 3, 240, 40, False, "HS", 1, 1),
                                        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                                        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1)]
        layers2: List[nn.Module] = []
        for cnf in inverted_residual_setting2:
            layers2.append(InvertedResidual(cnf, norm_layer))
        self.upConv2 = nn.Sequential(*layers2)

        self.attn3 = AttentionBlock(24, 40, 40)
        self.upsample3 = nn.Upsample(mode="bilinear", scale_factor=2)
        inverted_residual_setting3 = [bneck_conf(64, 3, 72, 24, True, "RE", 1, 1),
                                        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1)]
        layers3: List[nn.Module] = []
        for cnf in inverted_residual_setting3:
            layers3.append(InvertedResidual(cnf, norm_layer))
        self.upConv3 = nn.Sequential(*layers3)

        self.attn4 = AttentionBlock(16, 24, 24)
        self.upsample4 = nn.Upsample(mode="bilinear", scale_factor=2)
        inverted_residual_setting4 = [bneck_conf(40, 3, 64, 16, False, "RE", 1, 1),
                                        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1)]
        layers4: List[nn.Module] = []
        for cnf in inverted_residual_setting4:
            layers4.append(InvertedResidual(cnf, norm_layer))
        self.upConv4 = nn.Sequential(*layers4)

        self.upsample5 = nn.Upsample(mode="bilinear", scale_factor=2)
        self.T = ConvBlock(16, 2, 1, stride=1, padding=0, activation=nn.Sigmoid)

    def forward(self, c1, c2, c3, c4, c5):
        o1 = self.attn1(c2, c1)
        o1 = self.upsample1(o1)
        o1 = torch.cat([o1, c2], dim=1)
        o1 = self.upConv1(o1)

        o2 = self.attn2(c3, o1)
        o2 = self.upsample2(o2)
        o2 = torch.cat([o2, c3], dim=1)
        o2 = self.upConv2(o2)

        o3 = self.attn3(c4, o2)
        o3 = self.upsample3(o3)
        o3 = torch.cat([o3, c4], dim=1)
        o3 = self.upConv3(o3)

        o4 = self.attn4(c5, o3)
        o4 = self.upsample4(o4)
        o4 = torch.cat([o4, c5], dim=1)
        o4 = self.upConv4(o4)

        o5 = self.upsample5(o4)
        o5 = self.T(o5)
        # this is grid
        return o5