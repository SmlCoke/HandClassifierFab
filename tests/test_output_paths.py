"""Tests for per-model output path resolution (paths.output_root)."""

import os
from pathlib import Path

from hand_classifier.config import (
    resolve_output_paths, load_config, align_config_to_checkpoint,
)


def _p(*parts):
    return str(Path(*parts))


def test_resolve_output_paths_derives_model_dirs():
    config = {
        "model": {"version": "v2", "architecture": "v2_convnet_l"},
        "paths": {"output_root": "/tmp/hcf_out"},
    }
    resolved = resolve_output_paths(config)
    p = resolved["paths"]
    assert p["checkpoint_dir"] == _p("/tmp/hcf_out", "v2", "v2_convnet_l", "checkpoints")
    assert p["metrics_dir"] == _p("/tmp/hcf_out", "v2", "v2_convnet_l", "train")
    assert p["splits_dir"] == _p("/tmp/hcf_out", "v2", "v2_convnet_l")
    assert p["eval_dir"] == _p("/tmp/hcf_out", "v2", "v2_convnet_l", "eval")
    assert p["onnx_path"] == _p("/tmp/hcf_out", "v2", "v2_convnet_l", "model.onnx")
    # the original config dict is not mutated
    assert "checkpoint_dir" not in config["paths"]


def test_resolve_output_paths_v1_defaults():
    config = {
        "model": {"architecture": "mobilenet_v3_small"},  # no version key
        "paths": {"output_root": "outputs"},
    }
    resolved = resolve_output_paths(config)
    p = resolved["paths"]
    assert p["splits_dir"] == _p("outputs", "v1", "mobilenet_v3_small")
    assert p["checkpoint_dir"] == _p(
        "outputs", "v1", "mobilenet_v3_small", "checkpoints"
    )


def test_resolve_output_paths_legacy_without_root():
    """Without output_root the explicit legacy paths are kept unchanged."""
    config = {
        "model": {"version": "v1", "architecture": "mobilenet_v3_small"},
        "paths": {"checkpoint_dir": "a/ckpt", "splits_dir": "b"},
    }
    resolved = resolve_output_paths(config)
    assert resolved["paths"] == config["paths"]
    assert "eval_dir" not in resolved["paths"]
    assert "onnx_path" not in resolved["paths"]


def test_resolve_output_paths_idempotent():
    config = {
        "model": {"version": "v2", "architecture": "v2_hybrid_s"},
        "paths": {"output_root": "/tmp/hcf_out"},
    }
    once = resolve_output_paths(config)
    twice = resolve_output_paths(once)
    assert once["paths"] == twice["paths"]


def test_resolve_output_paths_relative_root(tmp_path, monkeypatch):
    """Relative output_root resolves against the current working directory."""
    root = tmp_path / "out"
    monkeypatch.chdir(tmp_path)
    config = {
        "model": {"version": "v1", "architecture": "mobilenet_v3_small"},
        "paths": {"output_root": str(root)},
    }
    resolved = resolve_output_paths(config)
    assert resolved["paths"]["checkpoint_dir"] == str(
        root / "v1" / "mobilenet_v3_small" / "checkpoints"
    )


def test_configs_carry_output_root():
    """The shipped configs configure output_root."""
    for name in ["train", "evaluate", "export_onnx", "cvat_label_test"]:
        config = load_config(f"configs/{name}.yaml")
        assert config["paths"].get("output_root"), f"{name}.yaml missing output_root"


# --- align_config_to_checkpoint ---

def _ckpt(model_cfg, paths_cfg=None):
    ckpt = {"epoch": 5, "model_state_dict": {}}
    ckpt["config"] = {"model": dict(model_cfg)}
    if paths_cfg is not None:
        ckpt["config"]["paths"] = dict(paths_cfg)
    return ckpt


def test_align_auto_located_checkpoint():
    """Config says v1/mobilenet_v3_small but the checkpoint was trained as
    v2/v2_convnet_l: model AND output paths must follow the checkpoint."""
    config = {
        "model": {"version": "v1", "architecture": "mobilenet_v3_small"},
        "paths": {"output_root": "/tmp/out"},
    }
    ckpt_paths = {
        "checkpoint_dir": "/tmp/out/v2/v2_convnet_l/checkpoints",
        "splits_dir": "/tmp/out/v2/v2_convnet_l",
        "eval_dir": "/tmp/out/v2/v2_convnet_l/eval",
        "onnx_path": "/tmp/out/v2/v2_convnet_l/model.onnx",
    }
    ckpt = _ckpt({"version": "v2", "architecture": "v2_convnet_l"}, ckpt_paths)

    aligned, changed = align_config_to_checkpoint(config, ckpt)
    assert changed is True
    assert aligned["model"]["architecture"] == "v2_convnet_l"
    assert aligned["paths"]["eval_dir"] == ckpt_paths["eval_dir"]
    assert aligned["paths"]["onnx_path"] == ckpt_paths["onnx_path"]
    # original config untouched
    assert config["model"]["architecture"] == "mobilenet_v3_small"


def test_align_explicit_checkpoint_also_follows_paths():
    """Even an explicitly passed checkpoint must route outputs to the
    checkpoint's own model directory (never to a different model's)."""
    config = {
        "model": {"version": "v1", "architecture": "mobilenet_v3_small"},
        "paths": {"output_root": "/tmp/out"},
    }
    ckpt_paths = {
        "eval_dir": "/tmp/out/v2/v2_hybrid_s/eval",
        "onnx_path": "/tmp/out/v2/v2_hybrid_s/model.onnx",
    }
    ckpt = _ckpt({"version": "v2", "architecture": "v2_hybrid_s"}, ckpt_paths)

    aligned, changed = align_config_to_checkpoint(config, ckpt)
    assert changed is True
    assert aligned["model"]["architecture"] == "v2_hybrid_s"
    assert aligned["paths"]["eval_dir"] == ckpt_paths["eval_dir"]
    assert aligned["paths"]["onnx_path"] == ckpt_paths["onnx_path"]


def test_align_same_model_no_change():
    config = {
        "model": {"version": "v1", "architecture": "mobilenet_v3_small"},
        "paths": {"output_root": "/tmp/out"},
    }
    ckpt = _ckpt({"version": "v1", "architecture": "mobilenet_v3_small"})
    aligned, changed = align_config_to_checkpoint(config, ckpt)
    assert changed is False
    assert aligned is config


def test_align_pretrained_difference_no_alignment():
    """pretrained=true in the checkpoint vs false in eval/export configs
    is expected and must NOT trigger alignment."""
    config = {
        "model": {
            "version": "v1", "architecture": "mobilenet_v3_small",
            "num_handedness": 2, "num_presence": 2,
            "input_channels": 1, "pretrained": False,
        },
        "paths": {"output_root": "/tmp/out"},
    }
    ckpt = _ckpt({
        "version": "v1", "architecture": "mobilenet_v3_small",
        "num_handedness": 2, "num_presence": 2,
        "input_channels": 1, "pretrained": True,
    })
    aligned, changed = align_config_to_checkpoint(config, ckpt)
    assert changed is False
    assert aligned is config


def test_align_checkpoint_without_config_no_change():
    config = {
        "model": {"version": "v1", "architecture": "mobilenet_v3_small"},
        "paths": {"output_root": "/tmp/out"},
    }
    aligned, changed = align_config_to_checkpoint(config, {"epoch": 1})
    assert changed is False
    assert aligned is config
