#!/usr/bin/env python
"""Evaluation entry point for the hand classifier."""

import argparse
import logging
import sys

from hand_classifier import load_config, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate hand classifier")
    parser.add_argument(
        "--config", "-c", default="configs/hand_classifier.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint (default: outputs/checkpoints/best.pth)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/eval",
        help="Directory for evaluation outputs",
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
    evaluate(config, args.checkpoint, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
