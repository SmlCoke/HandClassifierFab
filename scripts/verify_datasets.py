#!/usr/bin/env python
"""Verify dataset warehouse usability: structure, XML, labels, image integrity.

Checks every data-source directory resolved from the config's
``data.train_sources`` / ``data.val_sources`` globs (or from ``--sources``):

- directory structure (images/ + CVAT XML, or images/ only for negatives)
- XML parses; every <image> referenced in XML exists on disk
- every PNG on disk is loadable, 256x256
- label distribution (Left/Right/no_hand/ignore_for_training/unknown_handedness)

Directories without images/ or XML (e.g. ``old/``, ``NegativeTrain/`` container
dirs) are skipped, matching ``hand_classifier.parser.collect_all_samples``.

Exit code: 0 = ok, 1 = usage/config error, 2 = dataset errors found.
"""

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from hand_classifier.config import load_config
from hand_classifier.parser import parse_cvat_xml

logger = logging.getLogger("verify_datasets")

EXPECTED_SIZE = 256


def _expand(pattern):
    pattern = os.path.expanduser(pattern)
    if "*" in pattern:
        return sorted(glob(pattern))
    return [pattern]


def _xml_candidates(source_dir):
    for name in ["cvat_reviewed.xml", "cvat_autolabel.xml"]:
        p = source_dir / name
        if p.exists():
            return p
    # fall back to any cvat_*.xml
    found = sorted(source_dir.glob("cvat_*.xml"))
    return found[0] if found else None


def verify_source_dir(source_dir):
    """Verify one data-source directory. Returns (status, report dict)."""
    source_dir = Path(source_dir)
    report = {"dir": str(source_dir), "errors": [], "warnings": []}

    images_dir = source_dir / "images"
    xml_path = _xml_candidates(source_dir)

    if not images_dir.is_dir() and xml_path is None:
        report["status"] = "skipped"
        return report

    if not images_dir.is_dir():
        report["errors"].append("images/ directory missing")
        report["status"] = "error"
        return report

    pngs = sorted(images_dir.glob("*.png"))
    report["n_images"] = len(pngs)
    report["xml"] = xml_path.name if xml_path else None

    if xml_path is None:
        # Image-only directory: all treated as negative samples
        report["kind"] = "negative-only"
    else:
        report["kind"] = "labeled"
        try:
            samples = parse_cvat_xml(str(xml_path), str(images_dir))
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"XML parse failed: {e}")
            report["status"] = "error"
            return report

        # Label distribution from XML
        labels = Counter()
        xml_names = set()
        tree_labels = None
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            xml_names = {
                os.path.basename(img.get("name", ""))
                for img in root.findall("image")
            }
            for img in root.findall("image"):
                tag = img.find("tag")
                if tag is not None:
                    labels[tag.get("label", "")] += 1
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"XML re-parse failed: {e}")
            report["status"] = "error"
            return report

        report["xml_images"] = len(xml_names)
        report["labels"] = dict(labels)

        # Missing images referenced in XML
        missing = sorted(xml_names - {p.name for p in pngs})
        if missing:
            report["errors"].append(
                f"{len(missing)} image(s) referenced in XML but missing on disk: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        # Orphan images not referenced in XML
        orphan = sorted({p.name for p in pngs} - xml_names)
        if orphan:
            report["warnings"].append(
                f"{len(orphan)} image(s) on disk not referenced in XML: "
                f"{orphan[:5]}{'...' if len(orphan) > 5 else ''}"
            )

    # Image integrity: load every PNG, check size/mode
    bad_size = []
    bad_load = []
    modes = Counter()
    for p in pngs:
        try:
            with Image.open(p) as im:
                im.load()
                modes[im.mode] += 1
                if im.size != (EXPECTED_SIZE, EXPECTED_SIZE):
                    bad_size.append((p.name, im.size))
        except (UnidentifiedImageError, OSError, EOFError):
            bad_load.append(p.name)

    if bad_load:
        report["errors"].append(
            f"{len(bad_load)} image(s) failed to load: "
            f"{bad_load[:5]}{'...' if len(bad_load) > 5 else ''}"
        )
    if bad_size:
        report["errors"].append(
            f"{len(bad_size)} image(s) not {EXPECTED_SIZE}x{EXPECTED_SIZE}: "
            f"{bad_size[:5]}{'...' if len(bad_size) > 5 else ''}"
        )
    report["modes"] = dict(modes)
    if modes and set(modes) - {"L", "LA", "I;16"}:
        report["warnings"].append(
            f"non-grayscale image modes found: {dict(modes)}"
        )

    report["status"] = "error" if report["errors"] else "ok"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", "-c", default="configs/train.yaml",
        help="Path to YAML config (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--sources", default=None,
        help="Comma-separated glob patterns; overrides config train+val sources",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    if args.sources:
        patterns = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        data_cfg = config.get("data", {})
        patterns = list(data_cfg.get("train_sources", []))
        patterns += list(data_cfg.get("val_sources", []))
    if not patterns:
        print("No source patterns found (config or --sources).", file=sys.stderr)
        return 1

    all_dirs = []
    for pat in patterns:
        all_dirs.extend(_expand(pat))

    reports = []
    seen = set()
    for d in all_dirs:
        d = os.path.normpath(d)
        if d in seen:
            continue
        seen.add(d)
        reports.append(verify_source_dir(d))

    n_ok = n_error = n_skipped = 0
    error_dirs = []
    print("\n=== Dataset verification report ===")
    for r in reports:
        if r["status"] == "skipped":
            n_skipped += 1
            continue
        status = r["status"].upper()
        print(f"\n[{status}] {r['dir']}")
        print(f"  kind={r['kind']} images={r.get('n_images')} "
              f"xml={r.get('xml')} xml_images={r.get('xml_images', '-')}")
        if "labels" in r:
            print(f"  labels: {r['labels']}")
        if r["modes"]:
            print(f"  modes: {r['modes']}")
        for w in r["warnings"]:
            print(f"  WARN: {w}")
        for e in r["errors"]:
            print(f"  ERROR: {e}")
        if r["status"] == "ok":
            n_ok += 1
        else:
            n_error += 1
            error_dirs.append(r["dir"])

    print(f"\n=== Summary: {n_ok} ok, {n_error} with errors, "
          f"{n_skipped} skipped (container/old dirs) ===")
    if error_dirs:
        print("Directories with errors:")
        for d in error_dirs:
            print(f"  - {d}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
