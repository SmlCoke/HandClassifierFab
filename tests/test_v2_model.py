"""Tests for v2.0 series model architectures (accuracy-first family).

Verifies the I/O contract stays identical to v1.0:
  input (N, 1, 256, 256) -> dict {"handedness": (N, 2), "hand_presence": (N, 2)}
"""

import torch
import pytest

from models.factory import build_model, list_architectures

V2_CUSTOM = [
    "v2_convnet_s",
    "v2_convnet_l",
    "v2_multibranch",
    "v2_hybrid_s",
    "v2_hybrid_l",
]
V2_TRANSFER = [
    "v2_resnet50",
    "v2_convnext_tiny",
    "v2_efficientnet_v2_s",
    "v2_vit_b16",
]


@pytest.mark.parametrize("name", V2_CUSTOM + V2_TRANSFER)
def test_v2_build_and_forward(name):
    model = build_model(name, pretrained=False)
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict)
    assert out["handedness"].shape == (2, 2)
    assert out["hand_presence"].shape == (2, 2)
    # logits are finite
    assert torch.isfinite(out["handedness"]).all()
    assert torch.isfinite(out["hand_presence"]).all()


def test_v2_list_architectures():
    archs = list_architectures()
    for name in V2_CUSTOM + V2_TRANSFER:
        assert name in archs
    v2_only = list_architectures(version="v2")
    assert set(v2_only) == set(V2_CUSTOM + V2_TRANSFER)
    v1_only = list_architectures(version="v1")
    assert "mobilenet_v3_small" in v1_only
    assert "v2_convnet_s" not in v1_only


def test_v2_version_parameter():
    model = build_model("v2_convnet_s", pretrained=False, version="v2")
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out["handedness"].shape == (1, 2)

    with pytest.raises(ValueError, match="does not exist in version"):
        build_model("v2_convnet_s", pretrained=False, version="v1")

    with pytest.raises(ValueError, match="Unknown model version"):
        build_model("v2_convnet_s", pretrained=False, version="v3")


def test_v2_parameter_counts():
    """v2 models must be strictly larger than the v1.0 small backbone."""
    v1_small = sum(p.numel() for p in
                   build_model("mobilenet_v3_small", pretrained=False).parameters())
    for name in V2_CUSTOM:
        n = sum(p.numel() for p in build_model(name, pretrained=False).parameters())
        assert n > v1_small, f"{name} not larger than v1 small: {n} vs {v1_small}"

    counts = {
        name: sum(p.numel() for p in
                  build_model(name, pretrained=False).parameters())
        for name in V2_CUSTOM
    }
    assert 8_000_000 < counts["v2_convnet_s"] < 18_000_000
    assert 40_000_000 < counts["v2_convnet_l"] < 90_000_000
    assert 8_000_000 < counts["v2_multibranch"] < 25_000_000
    assert 10_000_000 < counts["v2_hybrid_s"] < 25_000_000
    assert 30_000_000 < counts["v2_hybrid_l"] < 55_000_000


def test_v2_transfer_parameter_counts():
    counts = {
        name: sum(p.numel() for p in
                  build_model(name, pretrained=False).parameters())
        for name in V2_TRANSFER
    }
    assert counts["v2_resnet50"] > 20_000_000
    assert counts["v2_convnext_tiny"] > 20_000_000
    assert counts["v2_efficientnet_v2_s"] > 15_000_000
    assert counts["v2_vit_b16"] > 80_000_000


def test_v2_batch_inference_contract():
    model = build_model("v2_hybrid_s", pretrained=False)
    model.eval()
    x = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out["handedness"].shape == (4, 2)
    assert out["hand_presence"].shape == (4, 2)
    ph = torch.softmax(out["handedness"], dim=1)
    pp = torch.softmax(out["hand_presence"], dim=1)
    assert torch.allclose(ph.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(pp.sum(dim=1), torch.ones(4), atol=1e-5)


@pytest.mark.parametrize("name", ["v2_convnet_s", "v2_hybrid_s"])
def test_v2_onnx_export_contract(name, tmp_path):
    """ONNX export must keep the same I/O names/shapes as v1.0."""
    import onnxruntime as ort

    model = build_model(name, pretrained=False)
    model.eval()
    onnx_path = str(tmp_path / f"{name}.onnx")
    dummy = torch.randn(1, 1, 256, 256)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["handedness", "hand_presence"],
        dynamic_axes={
            "input": {0: "batch"},
            "handedness": {0: "batch"},
            "hand_presence": {0: "batch"},
        },
        opset_version=13,
    )
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_h, out_p = session.run(
        ["handedness", "hand_presence"], {"input": dummy.numpy()}
    )
    assert out_h.shape == (1, 2)
    assert out_p.shape == (1, 2)
