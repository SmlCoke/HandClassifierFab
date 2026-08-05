#!/usr/bin/env python
"""CVAT label test: run handedness inference and relabel XML."""

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
        "--config", "-c", default="configs/hand_classifier.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to ONNX model (default: outputs/model.onnx)",
    )
    parser.add_argument(
        "--xml", default=None,
        help="Path to input CVAT XML (default: first test source)",
    )
    parser.add_argument(
        "--images-dir", default=None,
        help="Path to images directory (default: XML's ../images)",
    )
    parser.add_argument(
        "--output", "-o", default="outputs/cvat_relabeled.xml",
        help="Output XML path",
    )
    parser.add_argument(
        "--gold-xml", default=None,
        help="Path to gold-standard XML for agreement computation",
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

    # Determine model path
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = Path(
            config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
        ).parent / "model.onnx"
    model_path = str(model_path)

    # Determine XML path
    if args.xml:
        xml_path = args.xml
    else:
        test_sources = config.get("data", {}).get("test_sources", [])
        if test_sources:
            xml_path = Path(test_sources[0]) / "cvat_autolabel.xml"
        else:
            xml_path = "data/dataset_test/complex-near-bright-random-val-s01-peak/cvat_autolabel.xml"

    # Relabel
    stats = relabel_cvat_xml(
        str(xml_path), model_path, args.output, args.images_dir,
    )
    print(f"Relabeling complete: {stats}")

    # Agreement check
    if args.gold_xml:
        agreement = compute_agreement(args.output, args.gold_xml, args.images_dir)
        print(f"Agreement with gold: {agreement}")
    else:
        # Auto-detect gold XML
        xml_dir = Path(xml_path).parent
        gold_candidate = xml_dir / "cvat_reviewed.xml"
        if gold_candidate.exists():
            agreement = compute_agreement(args.output, str(gold_candidate), args.images_dir)
            print(f"Agreement with gold: {agreement}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
