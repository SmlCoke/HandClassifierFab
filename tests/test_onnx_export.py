"""Tests for ONNX export functionality.

These tests require onnx and onnxruntime to be installed.
"""

import os
import pytest
import torch
import numpy as np

# Skip all tests if onnx/onnxruntime not available
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from models.factory import build_model


@pytest.fixture
def model():
    m = build_model("mobilenet_v3_small", pretrained=False, num_classes=2)
    m.eval()
    return m


@pytest.fixture
def dummy_input():
    return torch.randn(1, 1, 256, 256)


def test_export_and_verify(model, dummy_input, tmp_path):
    """Export model to ONNX and verify output matches PyTorch."""
    onnx_path = str(tmp_path / "test_model.onnx")

    # Export
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )

    assert os.path.exists(onnx_path)

    # Verify ONNX model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Run inference with onnxruntime
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(["output"], {"input": dummy_input.numpy()})[0]

    # Run PyTorch inference
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    # Compare outputs
    np.testing.assert_allclose(onnx_output, torch_output, rtol=1e-4, atol=1e-4)


def test_export_dynamic_batch(model, dummy_input, tmp_path):
    """Test that dynamic batch export handles different batch sizes."""
    onnx_path = str(tmp_path / "test_dynamic.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        },
        opset_version=13,
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Test batch size 3
    batch_input = torch.randn(3, 1, 256, 256)
    onnx_output = session.run(["output"], {"input": batch_input.numpy()})[0]
    assert onnx_output.shape == (3, 2)


def test_export_opset_compatibility(model, dummy_input, tmp_path):
    """Test that opset 13 and 14 both work."""
    for opset in [13, 14]:
        onnx_path = str(tmp_path / f"test_opset{opset}.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=["input"], output_names=["output"],
            opset_version=opset,
        )
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        out = session.run(["output"], {"input": dummy_input.numpy()})[0]
        assert out.shape == (1, 2)
