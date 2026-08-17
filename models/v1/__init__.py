"""v1.0 series: MobileNetV3-based dual-head classifiers (original version).

This is the original v1.0 model family. It is kept unchanged for
backward compatibility; higher-capacity accuracy-first models live in
``models/v2``.
"""

from models.v1.mobilenetv3 import mobilenet_v3_small_1ch, mobilenet_v3_large_1ch

__all__ = [
    "mobilenet_v3_small_1ch",
    "mobilenet_v3_large_1ch",
]
