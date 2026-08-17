#!/usr/bin/env python
"""Evaluation entry point for the hand classifier."""

import argparse
import logging
import sys

from hand_classifier import load_config, evaluate, resolve_output_paths


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
        help="Directory for evaluation outputs "
             "(default: paths.eval_dir, or <splits_dir>/../eval)",
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
    config = resolve_output_paths(config)

    # output_dir stays None here so evaluate() derives it AFTER aligning
    # the config with the checkpoint's training config (see
    # align_config_to_checkpoint) — otherwise eval artifacts of a model
    # trained with a different config would land in the wrong directory.
    evaluate(config, args.checkpoint, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
