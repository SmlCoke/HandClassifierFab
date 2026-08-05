"""CVAT XML parser for hand classification datasets.

Handles both cvat_reviewed.xml (manual gold labels) and cvat_autolabel.xml
(automated labels with unknown_handedness placeholder).
"""

import os
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# Labels to exclude from training/evaluation
_EXCLUDE_LABELS = {"ignore_for_training", "no_hand", "unknown_handedness"}

# Label mapping
_LABEL_MAP = {"Left": 0, "Right": 1}


def parse_cvat_xml(xml_path, images_dir):
    """Parse a CVAT XML annotation file and return labeled samples.

    Args:
        xml_path: Path to the CVAT XML file.
        images_dir: Path to the directory containing images.

    Returns:
        list of dict: Each dict has keys 'image_path', 'label' (int), 'source' (str).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    samples = []
    skipped_missing = 0
    skipped_label = 0

    for image_elem in root.findall("image"):
        img_name = image_elem.get("name")
        # Resolve both "images/roi_xxx.png" and "roi_xxx.png" formats
        basename = os.path.basename(img_name)
        img_path = os.path.join(images_dir, basename)

        if not os.path.exists(img_path):
            skipped_missing += 1
            continue

        # Find the first tag element with a relevant label
        tag_elem = image_elem.find("tag")
        if tag_elem is None:
            continue

        label_str = tag_elem.get("label", "")

        if label_str in _EXCLUDE_LABELS:
            skipped_label += 1
            continue

        if label_str not in _LABEL_MAP:
            continue

        samples.append({
            "image_path": img_path,
            "label": _LABEL_MAP[label_str],
        })

    if skipped_missing > 0 or skipped_label > 0:
        logger.info(
            "  Parsed %s: %d samples, %d missing files, %d excluded by label",
            os.path.basename(xml_path), len(samples),
            skipped_missing, skipped_label,
        )
    else:
        logger.info(
            "  Parsed %s: %d samples", os.path.basename(xml_path), len(samples)
        )

    return samples


def collect_all_samples(source_dirs):
    """Collect samples from all source directories.

    Each source directory should contain:
      - images/*.png
      - cvat_reviewed.xml or cvat_autolabel.xml

    Args:
        source_dirs: List of directory paths (may contain glob patterns).

    Returns:
        list of dict: All samples with 'image_path', 'label', 'source' keys.
    """
    all_samples = []

    for source_pattern in source_dirs:
        source_pattern = os.path.expanduser(source_pattern)
        # Expand glob if needed
        if "*" in source_pattern:
            from glob import glob
            dirs = sorted(glob(source_pattern))
        else:
            dirs = [source_pattern]

        for source_dir in dirs:
            source_dir = Path(source_dir)
            if not source_dir.is_dir():
                logger.warning("Skipping non-existent directory: %s", source_dir)
                continue

            xml_path = None
            for candidate in ["cvat_reviewed.xml", "cvat_autolabel.xml"]:
                p = source_dir / candidate
                if p.exists():
                    xml_path = p
                    break

            images_dir = source_dir / "images"
            if xml_path is None or not images_dir.exists():
                logger.warning(
                    "Skipping %s: no XML or images/ found", source_dir
                )
                continue

            samples = parse_cvat_xml(str(xml_path), str(images_dir))
            source_name = source_dir.name
            for s in samples:
                s["source"] = source_name

            all_samples.extend(samples)
            logger.info(
                "  %s: %d samples", source_name, len(samples)
            )

    logger.info("Total samples collected: %d", len(all_samples))
    return all_samples
