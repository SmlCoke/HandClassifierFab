"""MobileNetV3 adapted for single-channel input with dual classification heads.

Outputs:
  - handedness: Left (0) / Right (1)
  - hand_presence: no_hand (0) / has_hand (1)
"""

import torch
import torch.nn as nn


def _get_mobilenet_v3(model_fn, weights_cls, pretrained):
    """Load a MobileNetV3 model, handling different torchvision API versions."""
    try:
        weights = weights_cls.IMAGENET1K_V1 if pretrained else None
        model = model_fn(weights=weights)
    except TypeError:
        try:
            model = model_fn(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:
            model = model_fn(pretrained=pretrained)
    return model


def _adapt_first_conv(conv, in_channels):
    """Average pretrained RGB weights into fewer/more channels."""
    old_weight = conv.weight.data
    if in_channels == 1:
        new_weight = old_weight.mean(dim=1, keepdim=True)
    else:
        new_weight = old_weight
    new_conv = nn.Conv2d(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv


class DualHeadMobileNetV3(nn.Module):
    """MobileNetV3 with two classification heads sharing a backbone.

    Args:
        backbone: MobileNetV3 model (features + avgpool + partial classifier).
        in_features: Input dimension to the task-specific heads.
        num_handedness: Number of handedness classes (default 2: Left/Right).
        num_presence: Number of hand presence classes (default 2: no_hand/has_hand).
    """

    def __init__(self, backbone, in_features,
                 num_handedness=2, num_presence=2):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # Shared layers: [0] Linear(576→1024), [1] Hardswish, [2] Dropout
        # The original classifier[3] is left as Identity
        self.classifier = backbone.classifier

        self.handedness_head = nn.Linear(in_features, num_handedness)
        self.hand_presence_head = nn.Linear(in_features, num_presence)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        shared = self.classifier(x)
        return {
            "handedness": self.handedness_head(shared),
            "hand_presence": self.hand_presence_head(shared),
        }


def mobilenet_v3_small_1ch(pretrained=True, num_handedness=2, num_presence=2):
    """MobileNetV3-Small adapted for single-channel input and dual-head classification.

    Args:
        pretrained: Load ImageNet pretrained weights.
        num_handedness: Number of handedness output classes.
        num_presence: Number of hand presence output classes.

    Returns:
        DualHeadMobileNetV3
    """
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    model = _get_mobilenet_v3(
        mobilenet_v3_small, MobileNet_V3_Small_Weights, pretrained
    )

    # Replace first conv: RGB (3ch) → grayscale (1ch)
    first_conv = model.features[0][0]
    model.features[0][0] = _adapt_first_conv(first_conv, in_channels=1)

    # Keep shared feature extractor in classifier, replace final Linear with Identity
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Identity()

    return DualHeadMobileNetV3(model, in_features, num_handedness, num_presence)


def mobilenet_v3_large_1ch(pretrained=True, num_handedness=2, num_presence=2):
    """MobileNetV3-Large adapted for single-channel input and dual-head classification."""
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    model = _get_mobilenet_v3(
        mobilenet_v3_large, MobileNet_V3_Large_Weights, pretrained
    )

    first_conv = model.features[0][0]
    model.features[0][0] = _adapt_first_conv(first_conv, in_channels=1)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Identity()

    return DualHeadMobileNetV3(model, in_features, num_handedness, num_presence)
