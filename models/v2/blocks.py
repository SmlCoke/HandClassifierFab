"""Shared building blocks for v2.0 series models.

All blocks use only ONNX opset-13 compatible operators (Conv, BatchNorm,
GELU, MaxPool, matmul/softmax), so every v2 model can be exported with the
standard ``scripts/export_onnx.py`` pipeline and keep the same I/O contract:
input ``(N, 1, 256, 256)`` → dict {``handedness``: (N, 2), ``hand_presence``: (N, 2)}.
"""

import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm2d + GELU (standard convolution, no depthwise)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = x.mean(dim=(2, 3)).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class DualHead(nn.Module):
    """Standard dual classification heads shared by all v2 models.

    Output dict matches the v1.0 contract:
      - handedness: logits (N, num_handedness)  [Left=0 / Right=1]
      - hand_presence: logits (N, num_presence) [no_hand=0 / has_hand=1]
    """

    def __init__(self, in_features, num_handedness=2, num_presence=2,
                 dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.handedness_head = nn.Linear(in_features, num_handedness)
        self.hand_presence_head = nn.Linear(in_features, num_presence)

    def forward(self, features):
        features = self.dropout(features)
        return {
            "handedness": self.handedness_head(features),
            "hand_presence": self.hand_presence_head(features),
        }
