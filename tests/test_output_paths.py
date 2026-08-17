"""Tests for per-model output path resolution (paths.output_root)."""

import os
from pathlib import Path

from hand_classifier.config import resolve_output_paths, load_config


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
