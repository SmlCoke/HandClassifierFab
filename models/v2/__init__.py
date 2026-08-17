"""v2.0 series: higher-capacity dual-head classifiers (accuracy-first).

Design goals for v2.0 (per project requirement): maximize handedness /
hand_presence accuracy on Hand ROI images, ignoring latency cost, while
keeping the exact same I/O contract as v1.0:

  input  : (N, 1, 256, 256) grayscale tensor
  output : dict {"handedness": (N, 2), "hand_presence": (N, 2)}

Families:
  - custom CNNs (standard conv / multi-branch / SE attention): ``convnet``
  - CNN + Transformer hybrid: ``hybrid``
  - ImageNet-pretrained torchvision backbones adapted to 1-channel: ``transfer``
"""

from models.v2.convnet import v2_convnet_s, v2_convnet_l, v2_multibranch
from models.v2.hybrid import v2_hybrid_s, v2_hybrid_l
from models.v2.transfer import (
    v2_resnet50, v2_convnext_tiny, v2_efficientnet_v2_s, v2_vit_b16,
)

__all__ = [
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
