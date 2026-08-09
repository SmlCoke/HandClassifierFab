#!/usr/bin/env python
"""ONNX export entry point for the hand classifier."""

import argparse
import logging
import sys
from pathlib import Path

from hand_classifier import load_config, export_onnx


def main():
    parser = argparse.ArgumentParser(description="Export hand classifier to ONNX")
    parser.add_argument(
        "--config", "-c", default="configs/export_onnx.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint (default: from config paths.checkpoint_dir/best.pth)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output ONNX file path (default: from config paths.splits_dir/../model.onnx)",
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
    
    # Set default output path from config if not specified
    if args.output is None:
        paths_cfg = config.get("paths", {})
        output_dir = Path(paths_cfg.get("splits_dir", "outputs")).parent
        args.output = str(output_dir / "model.onnx")
    
    onnx_path = export_onnx(config, args.checkpoint, args.output)
    print(f"ONNX model exported to: {onnx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
