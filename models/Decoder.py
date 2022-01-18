import torch.nn as nn
from typing import List
from torchvision.models.mobilenetv3 import InvertedResidualConfig, InvertedResidual
from functools import partial
from models.layers import AttentionBlock, ConvBlock
import numpy as np
import torch

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

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
        self.T = ConvBlock(16, 2, 5, stride=1, padding=2, activation=nn.Tanh)

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
        o5 = self.T(o5) * 0.5
        # this is grid
        return o5
    
    def requires_grad(self, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        for param in self.parameters():
            param.requires_grad = requires_grad