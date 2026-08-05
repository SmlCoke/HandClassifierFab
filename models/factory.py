"""Model factory for hand classifier architectures."""

from models.mobilenetv3 import mobilenet_v3_small_1ch, mobilenet_v3_large_1ch


_MODEL_BUILDERS = {
    "mobilenet_v3_small": mobilenet_v3_small_1ch,
    "mobilenet_v3_large": mobilenet_v3_large_1ch,
}


def build_model(architecture, pretrained=True, num_classes=2):
    """Build a hand classifier model by name.

    Args:
        architecture: One of 'mobilenet_v3_small', 'mobilenet_v3_large'.
        pretrained: Whether to load ImageNet pretrained weights.
        num_classes: Number of output classes.

    Returns:
        nn.Module: The built model.

    Raises:
        ValueError: If architecture is unknown.
    """
    if architecture not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(_MODEL_BUILDERS.keys())}"
        )
    return _MODEL_BUILDERS[architecture](
        pretrained=pretrained, num_classes=num_classes
    )


def list_architectures():
    """Return list of available architecture names."""
    return list(_MODEL_BUILDERS.keys())
