#!/usr/bin/env python
"""Training entry point for the hand classifier."""

import argparse
import logging
import sys

from hand_classifier import load_config, train


def main():
    parser = argparse.ArgumentParser(description="Train hand classifier")
    parser.add_argument(
        "--config", "-c", default="configs/hand_classifier.yaml",
        help="Path to YAML config file",
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
    checkpoint_path = train(config)
    print(f"Best checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
