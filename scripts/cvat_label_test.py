#!/usr/bin/env python
"""CVAT label test: run handedness inference and relabel XML.

Reads cvat_autolabel.xml from the configured source directory,
runs inference with the ONNX model, and writes cvat_hcf.xml
alongside the original file. The original cvat_autolabel.xml is
never modified.
"""

import argparse
import logging
import sys
from pathlib import Path

from hand_classifier import load_config, relabel_cvat_xml, compute_agreement


def main():
    parser = argparse.ArgumentParser(
        description="Relabel CVAT XML with predicted handedness"
    )
    parser.add_argument(
        "--config", "-c", default="configs/cvat_label_test.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to ONNX model (default: derived from config paths)",
    )
    parser.add_argument(
        "--xml", default=None,
        help="Path to input cvat_autolabel.xml (default: from config cvat_label_test.source_dir)",
    )
    parser.add_argument(
        "--images-dir", default=None,
        help="Path to images directory (default: XML's ../images)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output XML path (default: cvat_hcf.xml alongside source XML)",
    )
    parser.add_argument(
        "--gold-xml", default=None,
        help="Path to gold-standard cvat_reviewed.xml for agreement computation",
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

    # --- Determine model path ---
    if args.checkpoint:
        model_path = Path(args.checkpoint)
    else:
        paths_cfg = config.get("paths", {})
        model_path = (
            Path(paths_cfg.get("splits_dir", "outputs")).parent / "model.onnx"
        )
    model_path = str(model_path)

    # --- Determine input XML path ---
    if args.xml:
        xml_path = Path(args.xml)
    else:
        cvat_cfg = config.get("cvat_label_test", {})
        source_dir = cvat_cfg.get(
            "source_dir", "../autodl-tmp/DatasetFab/HCFCVATTestSource/dataset_source"
        )
        xml_path = Path(source_dir) / "cvat_autolabel.xml"

    # --- Determine output XML path ---
    if args.output:
        output_path = args.output
    else:
        cvat_cfg = config.get("cvat_label_test", {})
        output_filename = cvat_cfg.get("output_filename", "cvat_hcf.xml")
        output_path = str(xml_path.parent / output_filename)

    # --- Relabel ---
    print(f"Input XML:    {xml_path}")
    print(f"Output XML:   {output_path}")
    print(f"ONNX model:   {model_path}")

    stats = relabel_cvat_xml(
        str(xml_path), model_path, str(output_path), args.images_dir,
    )
    print(f"Relabeling complete: {stats}")

    # --- Agreement check ---
    if args.gold_xml:
        gold_path = args.gold_xml
    else:
        gold_candidate = xml_path.parent / "cvat_reviewed.xml"
        if gold_candidate.exists():
            gold_path = str(gold_candidate)
        else:
            gold_path = None

    if gold_path:
        print(f"\nGold XML: {gold_path}")
        agreement = compute_agreement(str(output_path), gold_path, args.images_dir)
        print(f"Agreement with gold: {agreement}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
