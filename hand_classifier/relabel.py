"""CVAT XML relabeling: replace unknown_handedness with predicted Left/Right.

Uses a byte-preserving regex approach that maintains all whitespace,
skeleton/points content, and self-closing vs multi-line styles verbatim.
"""

import os
import re
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
import onnxruntime as ort

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "Left", 1: "Right"}

# Regex to match <image ... name="xxx.png" ...> blocks up to </image>
_IMAGE_BLOCK_RE = re.compile(
    r'(<image\s[^>]*?name="([^"]*)"[^>]*>.*?</image>)', re.DOTALL
)
# Regex to find the tag attribute to replace
_TAG_LABEL_RE = re.compile(
    r'(<tag\s[^>]*?)label="unknown_handedness"'
    r'([^>]*(?:/>|>.*?</tag>))',
    re.DOTALL,
)


def _load_image(img_path):
    """Load and preprocess a single ROI image for inference."""
    image = Image.open(img_path)  # 'L' mode, grayscale
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229]),
    ])
    return transform(image).unsqueeze(0)  # (1, 1, 256, 256)


def relabel_cvat_xml(xml_path, model_path, output_path, images_dir=None):
    """Replace unknown_handedness tags with predicted Left/Right labels.

    Uses byte-preserving regex: all content outside the tag label attribute
    (skeleton, points, whitespace, formatting) is left unchanged.

    Args:
        xml_path: Path to the input CVAT XML (usually cvat_autolabel.xml).
        model_path: Path to ONNX model file.
        output_path: Path to write the relabeled XML.
        images_dir: Directory containing images (default: xml_path's ../images).

    Returns:
        dict: Statistics including total, Left count, Right count, errors.
    """
    xml_path = Path(xml_path)
    if images_dir is None:
        images_dir = xml_path.parent / "images"
    else:
        images_dir = Path(images_dir)

    # Load ONNX model
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Read original XML
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Find all image blocks
    stats = {"total": 0, "Left": 0, "Right": 0, "errors": 0}

    def _replace_tag(match):
        nonlocal stats
        full_block = match.group(1)
        img_name = match.group(2)
        basename = os.path.basename(img_name)
        img_path = images_dir / basename

        # Find the tag element in this block
        tag_match = _TAG_LABEL_RE.search(full_block)
        if not tag_match:
            return full_block  # No unknown_handedness tag, leave as-is

        if not img_path.exists():
            stats["errors"] += 1
            logger.warning("Image not found: %s", img_path)
            return full_block

        # Run inference
        try:
            input_tensor = _load_image(str(img_path))
            output = session.run(["output"], {input_name: input_tensor.numpy()})[0]
            pred = int(np.argmax(output, axis=1)[0])
            label = CLASS_NAMES[pred]
        except Exception as e:
            stats["errors"] += 1
            logger.warning("Inference error for %s: %s", basename, e)
            return full_block

        stats["total"] += 1
        if label == "Left":
            stats["Left"] += 1
        else:
            stats["Right"] += 1

        # Replace label in this block
        new_block = _TAG_LABEL_RE.sub(
            rf'\1label="{label}"\2', full_block
        )
        return new_block

    # Process all image blocks
    new_content = _IMAGE_BLOCK_RE.sub(_replace_tag, xml_content)

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info(
        "Relabeled %s: total=%d, Left=%d, Right=%d, errors=%d",
        xml_path.name, stats["total"], stats["Left"],
        stats["Right"], stats["errors"],
    )

    return stats


def compute_agreement(xml_pred_path, xml_gold_path, images_dir=None):
    """Compute agreement rate between predicted and gold-standard labels.

    Args:
        xml_pred_path: Path to the relabeled XML.
        xml_gold_path: Path to the cvat_reviewed.xml (gold).
        images_dir: Images directory for resolving paths.

    Returns:
        dict: Agreement statistics.
    """
    import xml.etree.ElementTree as ET

    pred_tree = ET.parse(xml_pred_path)
    gold_tree = ET.parse(xml_gold_path)

    if images_dir is None:
        images_dir = Path(xml_pred_path).parent / "images"

    # Build gold lookup by basename
    gold_labels = {}
    for img_elem in gold_tree.findall("image"):
        basename = os.path.basename(img_elem.get("name"))
        tag = img_elem.find("tag")
        if tag is not None:
            label = tag.get("label", "")
            if label in ("Left", "Right"):
                gold_labels[basename] = label

    # Compare
    total = 0
    agree = 0
    only_pred = 0
    only_gold = 0

    pred_labels = {}
    for img_elem in pred_tree.findall("image"):
        basename = os.path.basename(img_elem.get("name"))
        tag = img_elem.find("tag")
        if tag is not None:
            label = tag.get("label", "")
            if label in ("Left", "Right"):
                pred_labels[basename] = label

    for basename, pred_label in pred_labels.items():
        if basename in gold_labels:
            total += 1
            if pred_label == gold_labels[basename]:
                agree += 1
        else:
            only_pred += 1

    for basename in gold_labels:
        if basename not in pred_labels:
            only_gold += 1

    agreement = agree / max(total, 1)
    logger.info(
        "Agreement: %.4f (%d/%d), only_pred=%d, only_gold=%d",
        agreement, agree, total, only_pred, only_gold,
    )

    return {
        "total_compared": total,
        "agree": agree,
        "disagree": total - agree,
        "agreement_rate": round(agreement, 6),
        "only_in_pred": only_pred,
        "only_in_gold": only_gold,
    }
