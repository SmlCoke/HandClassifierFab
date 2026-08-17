"""v2.0 transfer family: ImageNet-pretrained torchvision backbones, adapted.

Each model keeps the dual-head I/O contract and accepts 256x256 single-channel
input. Pretrained RGB weights are adapted to 1 input channel by averaging the
first convolution's weights over the RGB axis (same technique as v1.0). These
are the strongest candidates when the from-scratch v2 models need a warm start:

  - ``v2_resnet50``:      ResNet-50, ~25M params
  - ``v2_convnext_tiny``: ConvNeXt-Tiny, ~28M params
  - ``v2_efficientnet_v2_s``: EfficientNetV2-S, ~21M params
  - ``v2_vit_b16``:       ViT-B/16 (position embedding interpolated from
                          224px to 256px input), ~86M params

``pretrained=True`` downloads ImageNet weights via torchvision on first use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.v2.blocks import DualHead

try:  # torchvision >= 0.13 weights API
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_tiny, ConvNeXt_Tiny_Weights,
        efficientnet_v2_s, EfficientNet_V2_S_Weights,
        vit_b_16, ViT_B_16_Weights,
    )
except Exception:  # pragma: no cover - older torchvision fallback
    from torchvision.models import (
        resnet50, convnext_tiny, efficientnet_v2_s, vit_b_16,
    )
    ResNet50_Weights = ConvNeXt_Tiny_Weights = None
    EfficientNet_V2_S_Weights = ViT_B_16_Weights = None


def _load_weights(fn, weights_enum, pretrained):
    """Build a torchvision model, handling pretrained flag variants."""
    if pretrained and weights_enum is not None:
        try:
            return fn(weights=weights_enum.IMAGENET1K_V1)
        except TypeError:
            return fn(weights="IMAGENET1K_V1")
    return fn(weights=None)


def _adapt_conv_1ch(conv):
    """Replace a 3-channel conv with a 1-channel conv (average RGB weights)."""
    old = conv.weight.data
    new = nn.Conv2d(
        1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups, conv.bias is not None,
    )
    new.weight.data = old.mean(dim=1, keepdim=True)
    if conv.bias is not None:
        new.bias.data = conv.bias.data
    return new


def _interpolate_vit_pos_embed(pos_embed, grid_h, grid_w):
    """Interpolate a ViT position embedding to a new token grid.

    pos_embed: (1, 1 + H*W, D) → (1, 1 + grid_h*grid_w, D)
    """
    cls_tok = pos_embed[:, :1]
    D = pos_embed.shape[-1]
    old_hw = int((pos_embed.shape[1] - 1) ** 0.5)
    grid = pos_embed[:, 1:].reshape(1, old_hw, old_hw, D)
    grid = grid.permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid, size=(grid_h, grid_w), mode="bicubic", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, D)
    return torch.cat([cls_tok, grid], dim=1)


def v2_resnet50(pretrained=True, num_handedness=2, num_presence=2):
    """ResNet-50 backbone with dual heads (1-channel input)."""
    model = _load_weights(resnet50, ResNet50_Weights, pretrained)
    model.conv1 = _adapt_conv_1ch(model.conv1)
    in_features = model.fc.in_features

    class ResNetDualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
            )
            self.avgpool = model.avgpool
            self.heads = DualHead(in_features, num_handedness,
                                  num_presence, dropout=0.1)

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x).flatten(1)
            return self.heads(x)

    return ResNetDualHead()


def v2_convnext_tiny(pretrained=True, num_handedness=2, num_presence=2):
    """ConvNeXt-Tiny backbone with dual heads (1-channel input)."""
    model = _load_weights(convnext_tiny, ConvNeXt_Tiny_Weights, pretrained)
    model.features[0][0] = _adapt_conv_1ch(model.features[0][0])
    in_features = model.classifier[2].in_features
    model.classifier[2] = DualHead(in_features, num_handedness,
                                   num_presence, dropout=0.1)

    class ConvNeXtDualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.classifier(x)  # LayerNorm2d -> Flatten -> DualHead
            return x

    return ConvNeXtDualHead()


def v2_efficientnet_v2_s(pretrained=True, num_handedness=2, num_presence=2):
    """EfficientNetV2-S backbone with dual heads (1-channel input)."""
    model = _load_weights(efficientnet_v2_s, EfficientNet_V2_S_Weights,
                          pretrained)
    model.features[0][0] = _adapt_conv_1ch(model.features[0][0])
    in_features = model.classifier[1].in_features
    model.classifier[1] = DualHead(in_features, num_handedness,
                                   num_presence, dropout=0.1)

    class EfficientNetDualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = model.features
            self.avgpool = model.avgpool
            self.classifier = model.classifier  # Dropout -> DualHead

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x).flatten(1)
            return self.classifier(x)

    return EfficientNetDualHead()


def v2_vit_b16(pretrained=True, num_handedness=2, num_presence=2):
    """ViT-B/16 backbone with dual heads (1-channel input, 256x256 input)."""
    model = _load_weights(vit_b_16, ViT_B_16_Weights, pretrained)
    in_features = model.heads.head.in_features
    embed_dim = in_features

    conv_proj = _adapt_conv_1ch(model.conv_proj)  # 16x16 patches

    # torchvision 0.16 keeps pos_embedding inside the encoder
    # (1, 197, 768): interpolate to the 16x16 token grid of a 256px input
    encoder = model.encoder
    pos_embed_src = getattr(encoder, "pos_embedding",
                            getattr(model, "pos_embedding", None))
    if pos_embed_src is None:
        raise AttributeError("ViT model has no position embedding attribute")
    pos_embed = _interpolate_vit_pos_embed(
        pos_embed_src.data, grid_h=16, grid_w=16
    )  # (1, 257, 768)
    encoder.pos_embedding = nn.Parameter(pos_embed)

    class ViTDualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_proj = conv_proj
            self.class_token = model.class_token
            self.encoder = encoder
            self.heads = DualHead(embed_dim, num_handedness,
                                  num_presence, dropout=0.1)

        def forward(self, x):
            x = self.conv_proj(x)                       # (B, D, 16, 16)
            x = x.flatten(2).transpose(1, 2)            # (B, 256, D)
            cls = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)              # (B, 257, D)
            x = self.encoder(x)                         # adds pos_embedding
            x = x[:, 0]                                 # class token
            return self.heads(x)

    return ViTDualHead()
