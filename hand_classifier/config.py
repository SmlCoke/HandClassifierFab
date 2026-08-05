"""Config loading utilities."""

import os
import yaml


def load_config(config_path):
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path, base_dir=None):
    """Resolve a path that may use ~ or be relative."""
    path = os.path.expanduser(path)
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(path)
