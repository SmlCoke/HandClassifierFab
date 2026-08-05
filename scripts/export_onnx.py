#!/usr/bin/env python
"""ONNX export entry point for the hand classifier."""

import argparse
import logging
import sys

from hand_classifier import load_config, export_onnx


def main():
    parser = argparse.ArgumentParser(description="Export hand classifier to ONNX")
    parser.add_argument(
        "--config", "-c", default="configs/hand_classifier.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint (default: outputs/checkpoints/best.pth)",
    )
    parser.add_argument(
        "--output", "-o", default="outputs/model.onnx",
        help="Output ONNX file path",
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
    onnx_path = export_onnx(config, args.checkpoint, args.output)
    print(f"ONNX model exported to: {onnx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
