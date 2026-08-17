"""v2.0 CNN family: standard convolutions, multi-branch blocks, SE attention.

All models in this module are trained from scratch (no pretrained weights),
keep the dual-head I/O contract, and are larger than the v1.0 MobileNetV3
backbone:

  - ``v2_convnet_s``: standard 3x3 conv ResNet-style blocks + SE (~13M params)
  - ``v2_convnet_l``: wider/deeper standard conv blocks + SE (~64M params)
  - ``v2_multibranch``: Inception-style multi-branch blocks + SE (~15M params)

Latency is intentionally sacrificed for accuracy.
"""

import torch
import torch.nn as nn

from models.v2.blocks import ConvBnAct, SEBlock, DualHead


class ConvBlock(nn.Module):
    """ResNet-style block: two standard 3x3 convs + SE + residual."""

    def __init__(self, in_channels, out_channels, stride=1, se_ratio=16):
        super().__init__()
        self.conv1 = ConvBnAct(in_channels, out_channels, 3, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.se = SEBlock(out_channels, reduction=se_ratio)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.se(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)


class _Branches(nn.Module):
    """Inception-style multi-branch: 1x1, 3x3, 5x5, pool+1x1."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        half = max(out_channels // 4, 8)

        def _conv_bn_act(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, s, p, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.branch1 = _conv_bn_act(in_channels, half, 1, stride, 0)
        self.branch2 = nn.Sequential(
            _conv_bn_act(in_channels, half, 1, 1, 0),
            _conv_bn_act(half, half, 3, stride, 1),
        )
        self.branch3 = nn.Sequential(
            _conv_bn_act(in_channels, half, 1, 1, 0),
            _conv_bn_act(half, half, 5, stride, 2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride, 1),
            _conv_bn_act(in_channels, half, 1, 1, 0),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x),
             self.branch3(x), self.branch4(x)], dim=1
        )


class MultiBranchBlock(nn.Module):
    """Multi-branch block + SE attention + residual."""

    def __init__(self, in_channels, out_channels, stride=1, se_ratio=16):
        super().__init__()
        self.branches = _Branches(in_channels, out_channels, stride)
        self.se = SEBlock(out_channels, reduction=se_ratio)
        self.act = nn.GELU()
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.se(self.branches(x))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)


class ConvNet(nn.Module):
    """Configurable standard-conv CNN with SE attention and dual heads."""

    def __init__(self, widths, depths, stem_channels=48, se_ratio=16,
                 num_handedness=2, num_presence=2, dropout=0.1):
        super().__init__()
        stages = []
        in_ch = stem_channels
        for i, (width, depth) in enumerate(zip(widths, depths)):
            blocks = []
            for j in range(depth):
                stride = 2 if j == 0 else 1
                blocks.append(
                    ConvBlock(in_ch, width, stride=stride, se_ratio=se_ratio)
                )
                in_ch = width
            stages.append(nn.Sequential(*blocks))
        self.stem = ConvBnAct(1, stem_channels, 3, stride=2)
        self.stages = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.heads = DualHead(in_ch, num_handedness, num_presence, dropout)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x).flatten(1)
        return self.heads(x)


class MultiBranchNet(nn.Module):
    """Configurable Inception-style multi-branch CNN with SE and dual heads."""

    def __init__(self, widths, depths, stem_channels=48, se_ratio=16,
                 num_handedness=2, num_presence=2, dropout=0.1):
        super().__init__()
        stages = []
        in_ch = stem_channels
        for i, (width, depth) in enumerate(zip(widths, depths)):
            blocks = []
            for j in range(depth):
                stride = 2 if j == 0 else 1
                blocks.append(
                    MultiBranchBlock(in_ch, width, stride=stride,
                                     se_ratio=se_ratio)
                )
                in_ch = width
            stages.append(nn.Sequential(*blocks))
        self.stem = ConvBnAct(1, stem_channels, 3, stride=2)
        self.stages = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.heads = DualHead(in_ch, num_handedness, num_presence, dropout)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x).flatten(1)
        return self.heads(x)


def v2_convnet_s(pretrained=False, num_handedness=2, num_presence=2):
    """Standard-conv CNN, ~8M params (trained from scratch)."""
    return ConvNet(
        widths=[64, 128, 256, 384], depths=[2, 3, 4, 3], stem_channels=48,
        num_handedness=num_handedness, num_presence=num_presence,
    )


def v2_convnet_l(pretrained=False, num_handedness=2, num_presence=2):
    """Standard-conv CNN, ~36M params (trained from scratch)."""
    return ConvNet(
        widths=[96, 192, 384, 768], depths=[3, 4, 8, 4], stem_channels=64,
        num_handedness=num_handedness, num_presence=num_presence,
    )


def v2_multibranch(pretrained=False, num_handedness=2, num_presence=2):
    """Inception-style multi-branch CNN, ~15M params (trained from scratch)."""
    return MultiBranchNet(
        widths=[128, 256, 512, 768], depths=[2, 3, 4, 3], stem_channels=64,
        num_handedness=num_handedness, num_presence=num_presence,
    )
