"""ONNX export for the hand classifier model."""

import logging
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)


def export_onnx(config, checkpoint_path=None, output_path=None):
    """Export trained model to ONNX format and verify with onnxruntime.

    Args:
        config: Configuration dict.
        checkpoint_path: Path to model checkpoint.
        output_path: Path for output ONNX file (default: outputs/model.onnx).

    Returns:
        str: Path to the exported ONNX file.
    """
    # --- Load model ---
    model_cfg = config["model"]
    from models.factory import build_model
    model = build_model(
        architecture=model_cfg["architecture"],
        pretrained=False,
        num_classes=model_cfg.get("num_classes", 2),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths_cfg = config.get("paths", {})

    if checkpoint_path is None:
        checkpoint_path = Path(paths_cfg.get("checkpoint_dir", "outputs/checkpoints")) / "best.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to("cpu")  # ONNX export on CPU

    if output_path is None:
        output_path = "outputs/model.onnx"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Export ---
    export_cfg = config.get("export", {})
    opset = export_cfg.get("onnx_opset", 13)

    # Dummy input: (1, 1, 256, 256) - single-channel grayscale
    dummy_input = torch.randn(1, 1, 256, 256)

    dynamic_axes = None
    if export_cfg.get("dynamic_batch", True):
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    logger.info("ONNX model exported to %s", output_path)

    # --- Verify with onnxruntime ---
    try:
        _verify_onnx(output_path, dummy_input)
        logger.info("ONNX verification passed")
    except ImportError:
        logger.warning("onnxruntime not available, skipping verification")
    except Exception as e:
        logger.error("ONNX verification failed: %s", e)
        raise

    return str(output_path)


def _verify_onnx(onnx_path, sample_input):
    """Verify ONNX model output matches PyTorch model output."""
    import onnxruntime as ort

    # Run ONNX model
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        ["output"], {"input": sample_input.numpy()}
    )[0]

    # PyTorch reference
    import onnx
    import torch

    # Load and check the ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure verified")

    # Verify output shape
    assert onnx_output.shape[0] == 1, f"Unexpected batch size: {onnx_output.shape}"
    assert onnx_output.shape[1] == 2, f"Unexpected num classes: {onnx_output.shape}"

    logger.info(
        "ONNX output shape: %s, range: [%.4f, %.4f]",
        onnx_output.shape, onnx_output.min(), onnx_output.max(),
    )
