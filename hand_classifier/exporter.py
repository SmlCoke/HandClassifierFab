"""ONNX export for dual-head hand classifier with two outputs."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_onnx(config, checkpoint_path=None, output_path=None):
    """Export trained dual-head model to ONNX and verify.

    Args:
        config: Configuration dict.
        checkpoint_path: Path to checkpoint (.pth). Auto-detected if None.
        output_path: Path for the ONNX file. Auto-derived if None.

    Returns:
        str: Path to the exported ONNX model.
    """
    device = torch.device("cpu")

    model_cfg = config["model"]
    from models.factory import build_model
    model = build_model(
        architecture=model_cfg["architecture"],
        pretrained=False,
        num_handedness=model_cfg.get("num_handedness", 2),
        num_presence=model_cfg.get("num_presence", 2),
        version=model_cfg.get("version", "v1"),
    )

    paths_cfg = config.get("paths", {})
    if checkpoint_path is None:
        checkpoint_path = (
            Path(paths_cfg.get("checkpoint_dir", "outputs/checkpoints"))
            / "best.pth"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint from %s (epoch %d)",
        checkpoint_path, checkpoint.get("epoch", -1),
    )

    if output_path is None:
        output_dir = Path(paths_cfg.get("splits_dir", "outputs")).parent
        output_path = str(output_dir / "model.onnx")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_cfg = config.get("export", {})
    opset = export_cfg.get("onnx_opset", 13)

    dummy_input = torch.randn(1, 1, 256, 256)

    dynamic_axes = None
    if export_cfg.get("dynamic_batch", True):
        dynamic_axes = {
            "input": {0: "batch"},
            "handedness": {0: "batch"},
            "hand_presence": {0: "batch"},
        }

    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["input"],
        output_names=["handedness", "hand_presence"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    logger.info("ONNX model exported to %s", output_path)

    # Verify
    try:
        _verify_onnx(output_path, dummy_input)
    except Exception as e:
        logger.error("ONNX verification failed: %s", e)
        raise

    return output_path


def _verify_onnx(onnx_path, sample_input):
    """Verify the ONNX model with onnx and onnxruntime."""
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure verified")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["handedness", "hand_presence"],
        {"input": sample_input.numpy()},
    )
    logger.info(
        "ONNX output shapes: handedness=%s, hand_presence=%s",
        outputs[0].shape, outputs[1].shape,
    )
    logger.info("ONNX verification passed")
