"""MobileNetV3 adapted for single-channel grayscale input."""

import torch
import torch.nn as nn


def _get_mobilenet_v3(model_fn, weights_cls, pretrained):
    """Load a MobileNetV3 model, handling different torchvision API versions."""
    try:
        # torchvision >= 0.14: weights=Weights.IMAGENET1K_V1
        weights = weights_cls.IMAGENET1K_V1 if pretrained else None
        model = model_fn(weights=weights)
    except TypeError:
        try:
            # torchvision >= 0.13: weights="IMAGENET1K_V1"
            model = model_fn(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:
            # torchvision < 0.13: pretrained=bool
            model = model_fn(pretrained=pretrained)
    return model


def _adapt_first_conv(conv, in_channels):
    """Average pretrained RGB weights into fewer/more channels."""
    old_weight = conv.weight.data  # (out_channels, 3, k, k)
    if in_channels == 1:
        new_weight = old_weight.mean(dim=1, keepdim=True)  # (out, 1, k, k)
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


def mobilenet_v3_small_1ch(pretrained=True, num_classes=2):
    """MobileNetV3-Small adapted for single-channel input and binary classification.

    Args:
        pretrained: Load ImageNet pretrained weights.
        num_classes: Number of output classes (default 2 for Left/Right).

    Returns:
        nn.Module: Modified MobileNetV3-Small.
    """
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    model = _get_mobilenet_v3(
        mobilenet_v3_small, MobileNet_V3_Small_Weights, pretrained
    )

    # Replace first conv: RGB (3ch) → grayscale (1ch) by averaging weights
    first_conv = model.features[0][0]
    model.features[0][0] = _adapt_first_conv(first_conv, in_channels=1)

    # Replace classifier head: 1000-class → num_classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def mobilenet_v3_large_1ch(pretrained=True, num_classes=2):
    """MobileNetV3-Large adapted for single-channel input and binary classification."""
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    model = _get_mobilenet_v3(
        mobilenet_v3_large, MobileNet_V3_Large_Weights, pretrained
    )

    first_conv = model.features[0][0]
    model.features[0][0] = _adapt_first_conv(first_conv, in_channels=1)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model
