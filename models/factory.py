"""Model factory for hand classifier architectures.

Organized by series (``model.version`` in the YAML config):

- ``v1``: original MobileNetV3-based models (backward compatible)
- ``v2``: higher-capacity accuracy-first models (custom CNNs, CNN+Transformer
  hybrids, and ImageNet-pretrained torchvision backbones)

Every architecture keeps the same I/O contract:
  input (N, 1, 256, 256) → dict {"handedness": (N, 2), "hand_presence": (N, 2)}
"""

from models.v1.mobilenetv3 import mobilenet_v3_small_1ch, mobilenet_v3_large_1ch
from models.v2.convnet import v2_convnet_s, v2_convnet_l, v2_multibranch
from models.v2.hybrid import v2_hybrid_s, v2_hybrid_l
from models.v2.transfer import (
    v2_resnet50, v2_convnext_tiny, v2_efficientnet_v2_s, v2_vit_b16,
)

# v1.0 series: original MobileNetV3-based models
_V1_BUILDERS = {
    "mobilenet_v3_small": mobilenet_v3_small_1ch,
    "mobilenet_v3_large": mobilenet_v3_large_1ch,
}

# v2.0 series: accuracy-first, larger models
_V2_BUILDERS = {
    "v2_convnet_s": v2_convnet_s,
    "v2_convnet_l": v2_convnet_l,
    "v2_multibranch": v2_multibranch,
    "v2_hybrid_s": v2_hybrid_s,
    "v2_hybrid_l": v2_hybrid_l,
    "v2_resnet50": v2_resnet50,
    "v2_convnext_tiny": v2_convnext_tiny,
    "v2_efficientnet_v2_s": v2_efficientnet_v2_s,
    "v2_vit_b16": v2_vit_b16,
}

_VERSIONS = {"v1": _V1_BUILDERS, "v2": _V2_BUILDERS}


def _resolve_version(architecture, version):
    """Determine the model series for an architecture name."""
    if version is not None:
        if version not in _VERSIONS:
            raise ValueError(
                f"Unknown model version '{version}'. "
                f"Available: {list(_VERSIONS.keys())}"
            )
        return version
    if architecture in _V1_BUILDERS:
        return "v1"
    if architecture in _V2_BUILDERS:
        return "v2"
    return None


def build_model(architecture, pretrained=True,
                num_handedness=2, num_presence=2, version=None):
    """Build a hand classifier model by name and series.

    Args:
        architecture: One of the names from list_architectures().
        pretrained: Whether to load ImageNet pretrained weights
            (v1 models and v2 transfer models only; custom v2 models
            are always trained from scratch).
        num_handedness: Number of handedness output classes.
        num_presence: Number of hand presence output classes.
        version: Model series, 'v1' or 'v2'. When None, it is inferred
            from the architecture name.

    Returns:
        nn.Module with forward(x) -> dict with 'handedness' and
        'hand_presence' logits.

    Raises:
        ValueError: If architecture or version is unknown.
    """
    resolved = _resolve_version(architecture, version)
    if resolved is None:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(_V1_BUILDERS.keys()) + list(_V2_BUILDERS.keys())}"
        )
    builders = _VERSIONS[resolved]
    if architecture not in builders:
        raise ValueError(
            f"Architecture '{architecture}' does not exist in version "
            f"'{resolved}'. Available: {list(builders.keys())}"
        )
    return builders[architecture](
        pretrained=pretrained,
        num_handedness=num_handedness,
        num_presence=num_presence,
    )


def list_architectures(version=None):
    """Return list of available architecture names.

    Args:
        version: If 'v1' or 'v2', only that series; None returns all.
    """
    if version is not None:
        if version not in _VERSIONS:
            raise ValueError(
                f"Unknown model version '{version}'. "
                f"Available: {list(_VERSIONS.keys())}"
            )
        return list(_VERSIONS[version].keys())
    return list(_V1_BUILDERS.keys()) + list(_V2_BUILDERS.keys())
