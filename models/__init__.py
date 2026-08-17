"""Hand classifier models package (v1.0 + v2.0 series)."""

from models.factory import build_model, list_architectures
from models.v1.mobilenetv3 import mobilenet_v3_small_1ch, mobilenet_v3_large_1ch
from models.v2.convnet import v2_convnet_s, v2_convnet_l, v2_multibranch
from models.v2.hybrid import v2_hybrid_s, v2_hybrid_l
from models.v2.transfer import (
    v2_resnet50, v2_convnext_tiny, v2_efficientnet_v2_s, v2_vit_b16,
)

__all__ = [
    "build_model",
    "list_architectures",
    # v1.0 series
    "mobilenet_v3_small_1ch",
    "mobilenet_v3_large_1ch",
    # v2.0 series
    "v2_convnet_s",
    "v2_convnet_l",
    "v2_multibranch",
    "v2_hybrid_s",
    "v2_hybrid_l",
    "v2_resnet50",
    "v2_convnext_tiny",
    "v2_efficientnet_v2_s",
    "v2_vit_b16",
]
