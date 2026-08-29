"""Batch inference with ONNX model for hand_presence screening.

Scans input directory for images, runs dual-head ONNX inference,
and copies low-confidence (no_hand) samples to an output directory.
Images are copied verbatim (shutil.copy2) to preserve SHA256.
"""

import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as T

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".tiff", ".tif", ".jpg", ".jpeg", ".bmp"}


def _collect_images(input_dir, extensions):
    """Recursively collect image paths from input_dir."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    extensions = set(e.lower() for e in extensions)
    images = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            images.append(p)
    logger.info("Found %d images in %s", len(images), input_dir)
    return images


def _load_transform():
    """Image preprocessing matching training normalization."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229]),
    ])


def _infer_batch(session, image_paths, transform):
    """Run inference on a batch of images.

    Returns:
        list of float: has_hand probability for each image.
    """
    batch = []
    valid_indices = []
    for i, p in enumerate(image_paths):
        try:
            img = Image.open(p)
            tensor = transform(img)
            batch.append(tensor)
            valid_indices.append(i)
        except Exception as e:
            logger.warning("Failed to load %s: %s", p, e)

    if not batch:
        return [1.0] * len(image_paths)  # default: keep all

    batch_tensor = np.stack([t.numpy() for t in batch])
    outputs = session.run(
        ["handedness", "hand_presence"],
        {"input": batch_tensor},
    )
    presence_logits = outputs[1]  # (N, 2): [no_hand_logit, has_hand_logit]
    presence_probs = _softmax(presence_logits)  # (N, 2)

    # Fill results: valid positions get computed prob, others default 1.0
    results = [1.0] * len(image_paths)
    for j, idx in enumerate(valid_indices):
        results[idx] = float(presence_probs[j, 1])  # has_hand probability

    return results


def _softmax(x):
    """Stable softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_inference(config):
    """Run batch inference to find low hand_presence (no_hand) images.

    Loads ONNX model, walks input_dir for images, runs inference,
    copies no_hand samples to output_dir.  Images are copied with
    shutil.copy2 to preserve all metadata and SHA256.

    Args:
        config: Configuration dict with an 'infer' section.

    Returns:
        dict: Statistics (total, kept, copied, errors).
    """
    infer_cfg = config["infer"]

    onnx_model = Path(infer_cfg["onnx_model"])
    input_dir = Path(infer_cfg["input_dir"])
    output_dir = Path(infer_cfg["output_dir"])
    threshold = infer_cfg.get("presence_threshold", 0.5)
    extensions = infer_cfg.get("image_extensions", list(_IMAGE_EXTENSIONS))
    batch_size = infer_cfg.get("batch_size", 64)

    # Validate
    if not onnx_model.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")

    import onnxruntime as ort

    logger.info("Loading ONNX model: %s", onnx_model)
    session = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])

    images = _collect_images(input_dir, extensions)
    if not images:
        logger.warning("No images found in %s", input_dir)
        return {"total": 0, "kept": 0, "copied": 0, "errors": 0}

    transform = _load_transform()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(images), "kept": 0, "copied": 0, "errors": 0}

    for i in tqdm(range(0, len(images), batch_size), desc="Infer"):
        batch_paths = images[i : i + batch_size]
        probs = _infer_batch(session, batch_paths, transform)

        for path, prob in zip(batch_paths, probs):
            if prob < threshold:
                # Low hand presence → no_hand → copy
                try:
                    dst = output_dir / path.name
                    # Avoid overwriting: add suffix if name collision
                    if dst.exists():
                        stem = dst.stem
                        suffix = dst.suffix
                        counter = 1
                        while dst.exists():
                            dst = output_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    shutil.copy2(path, dst)
                    stats["copied"] += 1
                except Exception as e:
                    logger.warning("Failed to copy %s: %s", path, e)
                    stats["errors"] += 1
            else:
                stats["kept"] += 1

    logger.info(
        "Inference complete: total=%d, kept=%d, copied=%d, errors=%d",
        stats["total"], stats["kept"], stats["copied"], stats["errors"],
    )

    return stats
