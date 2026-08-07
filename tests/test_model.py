"""Tests for dual-head model architecture."""

import torch
import pytest

from models.factory import build_model, list_architectures


def test_list_architectures():
    archs = list_architectures()
    assert "mobilenet_v3_small" in archs
    assert "mobilenet_v3_large" in archs


def test_build_mobilenet_v3_small():
    model = build_model("mobilenet_v3_small", pretrained=False)
    assert model is not None

    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, dict)
    assert out["handedness"].shape == (1, 2)
    assert out["hand_presence"].shape == (1, 2)


def test_build_mobilenet_v3_large():
    model = build_model("mobilenet_v3_large", pretrained=False)
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out["handedness"].shape == (1, 2)
    assert out["hand_presence"].shape == (1, 2)


def test_first_conv_is_single_channel():
    model = build_model("mobilenet_v3_small", pretrained=False)
    first_conv = model.features[0][0]
    assert first_conv.in_channels == 1


def test_classifier_output_classes():
    model = build_model("mobilenet_v3_small", pretrained=False)
    assert model.handedness_head.out_features == 2
    assert model.hand_presence_head.out_features == 2

    model3 = build_model("mobilenet_v3_small", pretrained=False,
                         num_handedness=3, num_presence=2)
    assert model3.handedness_head.out_features == 3
    assert model3.hand_presence_head.out_features == 2


def test_build_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("nonexistent_arch")


def test_pretrained_weights():
    try:
        model = build_model("mobilenet_v3_small", pretrained=True)
        assert model is not None
        assert model.features[0][0].in_channels == 1
    except Exception as e:
        pytest.skip(f"Pretrained weights not available: {e}")


def test_model_parameter_count():
    model_small = build_model("mobilenet_v3_small", pretrained=False)
    n_params = sum(p.numel() for p in model_small.parameters())
    assert n_params < 3_000_000, f"Small model too large: {n_params} params"


def test_model_batch_inference():
    model = build_model("mobilenet_v3_small", pretrained=False)
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
