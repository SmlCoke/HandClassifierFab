#!/usr/bin/env python
"""Inference entry point: screen for no_hand images using ONNX model."""

import argparse
import logging
import sys

from hand_classifier import load_config
from hand_classifier.infer import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="Run ONNX inference to find low hand_presence (no_hand) images"
    )
    parser.add_argument(
        "--config", "-c", default="configs/infer.yaml",
        help="Path to inference YAML config",
    )
    parser.add_argument(
        "--onnx-model", default=None,
        help="Override ONNX model path",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Override input directory",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override presence threshold (0.0-1.0)",
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

    # CLI overrides
    infer_cfg = config.setdefault("infer", {})
    if args.onnx_model:
        infer_cfg["onnx_model"] = args.onnx_model
    if args.input_dir:
        infer_cfg["input_dir"] = args.input_dir
    if args.output_dir:
        infer_cfg["output_dir"] = args.output_dir
    if args.threshold is not None:
        infer_cfg["presence_threshold"] = args.threshold

    stats = run_inference(config)
    print(
        f"\nInference done: {stats['total']} images scanned, "
        f"{stats['kept']} kept, {stats['copied']} copied to output, "
        f"{stats['errors']} errors"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
