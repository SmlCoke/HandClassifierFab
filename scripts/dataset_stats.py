#!/usr/bin/env python
"""Print per-source label distribution statistics."""

import argparse
import logging
import sys
from collections import defaultdict

from hand_classifier import load_config, collect_all_samples


def main():
    parser = argparse.ArgumentParser(
        description="Print dataset label distribution statistics"
    )
    parser.add_argument(
        "--config", "-c", default="configs/hand_classifier.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--source", "-s", default=None,
        help="Override source directory (single)",
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

    if args.source:
        source_dirs = [args.source]
    else:
        config = load_config(args.config)
        source_dirs = config.get("data", {}).get("train_sources", [])

    samples = collect_all_samples(source_dirs)

    # Aggregate per source
    per_source = defaultdict(lambda: {"Left": 0, "Right": 0, "total": 0})
    for s in samples:
        src = s["source"]
        per_source[src]["total"] += 1
        if s["label"] == 0:
            per_source[src]["Left"] += 1
        else:
            per_source[src]["Right"] += 1

    print("\n=== Dataset Statistics ===\n")
    total_left = 0
    total_right = 0
    total_all = 0

    for src in sorted(per_source.keys()):
        stats = per_source[src]
        ratio = stats["Left"] / max(stats["Right"], 1)
        print(
            f"  {src}: total={stats['total']:5d}  "
            f"Left={stats['Left']:5d}  Right={stats['Right']:5d}  "
            f"L/R={ratio:.2f}"
        )
        total_left += stats["Left"]
        total_right += stats["Right"]
        total_all += stats["total"]

    print(f"\n  TOTAL: {total_all}  Left={total_left}  Right={total_right}  "
          f"L/R={total_left / max(total_right, 1):.2f}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
