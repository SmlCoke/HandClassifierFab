"""CVAT XML parser and negative sample collector for dual-head classification.

Each sample dict has:
  - image_path: str
  - handedness_label: -1 (ignore), 0 (Left), or 1 (Right)
  - presence_label: 0 (no_hand) or 1 (has_hand)
  - source: str
"""

import os
import logging
from pathlib import Path
from glob import glob
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

_LABEL_MAP = {"Left": 0, "Right": 1}


def parse_cvat_xml(xml_path, images_dir):
    """Parse a CVAT XML annotation file and return samples with dual labels.

    Label logic:
      - Left/Right tag           → handedness=0/1,  presence=1
      - unknown_handedness tag   → handedness=-1,   presence=1
      - no_hand tag              → handedness=-1,   presence=0
      - ignore_for_training tag  → excluded entirely

    Args:
        xml_path: Path to the CVAT XML file.
        images_dir: Path to the directory containing images.

    Returns:
        list of dict with 'image_path', 'handedness_label', 'presence_label'.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    samples = []
    skipped_missing = 0
    skipped_ignore = 0

    for image_elem in root.findall("image"):
        img_name = image_elem.get("name")
        basename = os.path.basename(img_name)
        img_path = os.path.join(images_dir, basename)

        if not os.path.exists(img_path):
            skipped_missing += 1
            continue

        tag_elem = image_elem.find("tag")
        if tag_elem is None:
            continue

        label_str = tag_elem.get("label", "")

        if label_str == "ignore_for_training":
            skipped_ignore += 1
            continue

        if label_str in ("Left", "Right"):
            samples.append({
                "image_path": img_path,
                "handedness_label": _LABEL_MAP[label_str],
                "presence_label": 1,  # has hand
            })
        elif label_str == "unknown_handedness":
            # Has hand but handedness is unknown
            samples.append({
                "image_path": img_path,
                "handedness_label": -1,  # ignore for handedness training
                "presence_label": 1,     # has hand
            })
        elif label_str == "no_hand":
            samples.append({
                "image_path": img_path,
                "handedness_label": -1,  # ignore
                "presence_label": 0,     # no hand
            })
        # Other unknown labels are silently skipped

    if skipped_missing > 0 or skipped_ignore > 0:
        logger.info(
            "  Parsed %s: %d samples, %d missing, %d ignore_for_training",
            os.path.basename(xml_path), len(samples),
            skipped_missing, skipped_ignore,
        )
    else:
        logger.info(
            "  Parsed %s: %d samples", os.path.basename(xml_path), len(samples)
        )

    return samples


def collect_negative_samples(source_dir):
    """Collect negative (no_hand) samples from an image-only directory.

    The directory should contain PNG images directly (no XML required).
    Every image is treated as presence_label=0, handedness_label=-1.

    Args:
        source_dir: Path containing images/ subdirectory with .png files.

    Returns:
        list of dict with 'image_path', 'handedness_label', 'presence_label'.
    """
    source_dir = Path(source_dir)
    img_dir = source_dir / "images"
    if not img_dir.is_dir():
        logger.warning("No images/ in %s, skipping", source_dir)
        return []

    pngs = sorted(img_dir.glob("*.png"))
    samples = []
    for p in pngs:
        samples.append({
            "image_path": str(p),
            "handedness_label": -1,
            "presence_label": 0,
        })
    logger.info("  %s: %d negative samples", source_dir.name, len(samples))
    return samples


def collect_all_samples(source_dirs, negative_dirs=None):
    """Collect samples from all source directories (positive + negative).

    For directories with XML: parses labels for both tasks.
    For directories without XML (image-only): treats all as negative.

    Args:
        source_dirs: List of directory paths (may contain glob patterns).
        negative_dirs: Optional list of image-only directories (all negative).

    Returns:
        list of dict: All samples with dual labels and 'source' key.
    """
    all_samples = []

    # Collect from source dirs (with XML)
    for source_pattern in source_dirs:
        source_pattern = os.path.expanduser(source_pattern)
        if "*" in source_pattern:
            dirs = sorted(glob(source_pattern))
        else:
            dirs = [source_pattern]

        for source_dir in dirs:
            source_dir = Path(source_dir)
            if not source_dir.is_dir():
                logger.warning("Skipping non-existent: %s", source_dir)
                continue

            # Check if this is an image-only directory (no XML)
            has_xml = any(source_dir.glob("cvat_*.xml"))
            images_dir = source_dir / "images"

            if has_xml and images_dir.is_dir():
                # Find XML
                xml_path = None
                for candidate in ["cvat_reviewed.xml", "cvat_autolabel.xml"]:
                    p = source_dir / candidate
                    if p.exists():
                        xml_path = p
                        break

                if xml_path is None:
                    logger.warning("No XML found in %s", source_dir)
                    continue

                samples = parse_cvat_xml(str(xml_path), str(images_dir))
                source_name = source_dir.name
                for s in samples:
                    s["source"] = source_name
                all_samples.extend(samples)
                logger.info("  %s: %d samples", source_name, len(samples))

            elif images_dir.is_dir():
                # Image-only directory: treat as negative
                samples = collect_negative_samples(source_dir)
                source_name = source_dir.name
                for s in samples:
                    s["source"] = source_name
                all_samples.extend(samples)
            else:
                logger.warning("Skipping %s: no images/ found", source_dir)

    # Collect from explicit negative dirs
    if negative_dirs:
        for neg_pattern in negative_dirs:
            neg_pattern = os.path.expanduser(neg_pattern)
            if "*" in neg_pattern:
                dirs = sorted(glob(neg_pattern))
            else:
                dirs = [neg_pattern]

            for neg_dir in dirs:
                neg_dir = Path(neg_dir)
                if not neg_dir.is_dir():
                    logger.warning("Skipping non-existent: %s", neg_dir)
                    continue
                samples = collect_negative_samples(neg_dir)
                source_name = neg_dir.name
                for s in samples:
                    s["source"] = source_name
                all_samples.extend(samples)

    logger.info("Total samples collected: %d", len(all_samples))
    return all_samples
