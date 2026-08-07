"""Tests for ONNX export of dual-head model."""

import os
import pytest
import torch
import numpy as np

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from models.factory import build_model


@pytest.fixture
def model():
    m = build_model("mobilenet_v3_small", pretrained=False)
    m.eval()
    return m


@pytest.fixture
def dummy_input():
    return torch.randn(1, 1, 256, 256)


def test_export_and_verify(model, dummy_input, tmp_path):
    """Export dual-head model to ONNX and verify outputs."""
    onnx_path = str(tmp_path / "test_model.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["handedness", "hand_presence"],
        dynamic_axes={
            "input": {0: "batch"},
            "handedness": {0: "batch"},
            "hand_presence": {0: "batch"},
        },
        opset_version=13,
    )

    assert os.path.exists(onnx_path)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(
        ["handedness", "hand_presence"],
        {"input": dummy_input.numpy()},
    )

    with torch.no_grad():
        torch_out = model(dummy_input)

    np.testing.assert_allclose(
        onnx_outputs[0], torch_out["handedness"].numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        onnx_outputs[1], torch_out["hand_presence"].numpy(), rtol=1e-4, atol=1e-4
    )


def test_export_dynamic_batch(model, dummy_input, tmp_path):
    """Test dynamic batch export handles different batch sizes."""
    onnx_path = str(tmp_path / "test_dynamic.onnx")

    torch.onnx.export(
        model, dummy_input, onnx_path,
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

    batch_input = torch.randn(3, 1, 256, 256)
    h, p = session.run(
        ["handedness", "hand_presence"],
        {"input": batch_input.numpy()},
    )
    assert h.shape == (3, 2)
    assert p.shape == (3, 2)


def test_export_opset_compatibility(model, dummy_input, tmp_path):
    """Test that opset 13 and 14 both work for dual-head."""
    for opset in [13, 14]:
        onnx_path = str(tmp_path / f"test_opset{opset}.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=["input"],
            output_names=["handedness", "hand_presence"],
            opset_version=opset,
        )
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        h, p = session.run(
            ["handedness", "hand_presence"],
            {"input": dummy_input.numpy()},
        )
        assert h.shape == (1, 2)
        assert p.shape == (1, 2)
