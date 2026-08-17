"""Config loading utilities."""

import logging
import os
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


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


def resolve_output_paths(config):
    """Derive per-model output paths under ``paths.output_root``.

    When ``paths.output_root`` is configured, all training / evaluation /
    export artifacts are organized by model series and architecture so
    different models never overwrite each other::

        <output_root>/<model.version>/<model.architecture>/
            checkpoints/best.pth, last.pth
            train/metrics.jsonl
            eval/val_metrics.json
            splits.json
            model.onnx

    The derived keys are written back into ``config["paths"]`` as
    ``checkpoint_dir``, ``metrics_dir``, ``splits_dir``, ``eval_dir`` and
    ``onnx_path``. If ``output_root`` is absent, the explicitly configured
    legacy paths are kept unchanged (backward compatible).

    Args:
        config: Configuration dict.

    Returns:
        dict: A new config dict with resolved ``paths``.
    """
    config = dict(config)
    paths_cfg = dict(config.get("paths", {}))
    output_root = paths_cfg.get("output_root")
    if not output_root:
        return config

    model_cfg = config.get("model", {})
    version = str(model_cfg.get("version", "v1"))
    architecture = str(model_cfg.get("architecture", "unknown"))
    base = Path(resolve_path(str(output_root))) / version / architecture

    paths_cfg["checkpoint_dir"] = str(base / "checkpoints")
    paths_cfg["metrics_dir"] = str(base / "train")
    paths_cfg["splits_dir"] = str(base)
    paths_cfg["eval_dir"] = str(base / "eval")
    paths_cfg["onnx_path"] = str(base / "model.onnx")
    config["paths"] = paths_cfg
    return config


def align_config_to_checkpoint(config, checkpoint):
    """Align a config to the model actually stored in a checkpoint.

    Training saves the full config inside the checkpoint. When evaluating
    or exporting a checkpoint whose training config differs from the
    current config file (e.g. ``evaluate.yaml`` / ``export_onnx.yaml``
    still point at another model series/architecture), both the model
    section and the output paths are adopted from the checkpoint, so the
    correct architecture is always rebuilt and the artifacts land in that
    model's own directory (never in the directory of a different model).
    A warning is logged.

    Args:
        config: Current configuration dict.
        checkpoint: Loaded checkpoint dict (may carry 'config').

    Returns:
        tuple (config, changed): config with the model and paths sections
        aligned to the checkpoint; changed=True if any alignment happened.
    """
    ckpt_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not ckpt_cfg:
        return config, False

    ckpt_model = ckpt_cfg.get("model")
    cur_model = config.get("model", {})
    if not ckpt_model:
        return config, False

    # Compare only the architecture-defining fields: 'pretrained' is
    # legitimately different between train.yaml (true) and
    # evaluate.yaml / export_onnx.yaml (false) and must not trigger
    # alignment.
    def _arch_key(model_cfg):
        return {
            k: model_cfg.get(k) for k in (
                "version", "architecture", "num_handedness",
                "num_presence", "input_channels",
            )
        }

    if _arch_key(ckpt_model) == _arch_key(cur_model):
        return config, False

    config = dict(config)
    config["model"] = dict(ckpt_model)
    if ckpt_cfg.get("paths"):
        # Follow the training-time output layout so evaluation / export
        # artifacts land next to the checkpoint's own model directory.
        config["paths"] = dict(ckpt_cfg["paths"])
    logger.warning(
        "Checkpoint was trained with model %s/%s but the current config "
        "says %s/%s; using the checkpoint's model config and output "
        "paths for this run.",
        ckpt_model.get("version"), ckpt_model.get("architecture"),
        cur_model.get("version"), cur_model.get("architecture"),
    )
    return config, True
