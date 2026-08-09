#!/usr/bin/env python
"""Evaluation entry point for the hand classifier."""

import argparse
import logging
import sys
from pathlib import Path

from hand_classifier import load_config, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate hand classifier")
    parser.add_argument(
        "--config", "-c", default="configs/evaluate.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint (default: from config paths.checkpoint_dir/best.pth)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for evaluation outputs (default: from config paths.splits_dir/../eval)",
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
    
    if args.output_dir is None:
        paths_cfg = config.get("paths", {})
        parent_dir = Path(paths_cfg.get("splits_dir", "outputs")).parent
        args.output_dir = str(parent_dir / "eval")
    
    evaluate(config, args.checkpoint, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
