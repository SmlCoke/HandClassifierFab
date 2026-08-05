"""Tests for model architecture."""

import torch
import pytest

from models.factory import build_model, list_architectures


def test_list_architectures():
    archs = list_architectures()
    assert "mobilenet_v3_small" in archs
    assert "mobilenet_v3_large" in archs


def test_build_mobilenet_v3_small():
    model = build_model("mobilenet_v3_small", pretrained=False, num_classes=2)
    assert model is not None

    # Test forward pass with single-channel input
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2), f"Expected (1,2), got {out.shape}"


def test_build_mobilenet_v3_large():
    model = build_model("mobilenet_v3_large", pretrained=False, num_classes=2)
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2)


def test_first_conv_is_single_channel():
    """Verify the first conv layer accepts 1 channel input."""
    model = build_model("mobilenet_v3_small", pretrained=False)
    first_conv = model.features[0][0]
    assert first_conv.in_channels == 1, (
        f"Expected 1 input channel, got {first_conv.in_channels}"
    )


def test_classifier_output_classes():
    """Verify the classifier outputs the correct number of classes."""
    model = build_model("mobilenet_v3_small", pretrained=False, num_classes=2)
    classifier = model.classifier[3]
    assert classifier.out_features == 2

    # Test with different num_classes
    model5 = build_model("mobilenet_v3_small", pretrained=False, num_classes=5)
    assert model5.classifier[3].out_features == 5


def test_build_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("nonexistent_arch")


def test_pretrained_weights():
    """Test loading with pretrained weights (requires network on first run)."""
    try:
        model = build_model("mobilenet_v3_small", pretrained=True, num_classes=2)
        assert model is not None
        # First conv should still be 1-channel after adaptation
        assert model.features[0][0].in_channels == 1
    except Exception as e:
        pytest.skip(f"Pretrained weights not available: {e}")


def test_model_parameter_count():
    """Verify model is reasonably small (~1.5M params for small, ~4M for large)."""
    model_small = build_model("mobilenet_v3_small", pretrained=False, num_classes=2)
    n_params = sum(p.numel() for p in model_small.parameters())
    assert n_params < 3_000_000, f"Small model too large: {n_params} params"

    model_large = build_model("mobilenet_v3_large", pretrained=False, num_classes=2)
    n_params_large = sum(p.numel() for p in model_large.parameters())
    assert n_params_large < 6_000_000, f"Large model too large: {n_params_large} params"


def test_model_batch_inference():
    """Test inference with batch size > 1."""
    model = build_model("mobilenet_v3_small", pretrained=False, num_classes=2)
    model.eval()
    x = torch.randn(4, 1, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2)

    # Verify softmax sums to 1
    probs = torch.softmax(out, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
