#!/usr/bin/env python
"""Print per-source label distribution statistics (dual-head version)."""

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

    per_source = defaultdict(
        lambda: {"Left": 0, "Right": 0, "no_hand": 0, "unknown_hand": 0, "total": 0}
    )
    for s in samples:
        src = s["source"]
        per_source[src]["total"] += 1
        if s["presence_label"] == 0:
            per_source[src]["no_hand"] += 1
        elif s["handedness_label"] == 0:
            per_source[src]["Left"] += 1
        elif s["handedness_label"] == 1:
            per_source[src]["Right"] += 1
        elif s["handedness_label"] == -1:
            per_source[src]["unknown_hand"] += 1

    print("\n=== Dataset Statistics ===\n")
    total_left = total_right = total_nohand = total_unknown = total_all = 0

    for src in sorted(per_source.keys()):
        st = per_source[src]
        lr = st["Left"] / max(st["Right"], 1)
        print(
            f"  {src}: total={st['total']:5d}  "
            f"Left={st['Left']:5d}  Right={st['Right']:5d}  "
            f"no_hand={st['no_hand']:5d}  unknown={st['unknown_hand']:3d}  "
            f"L/R={lr:.2f}"
        )
        total_left += st["Left"]
        total_right += st["Right"]
        total_nohand += st["no_hand"]
        total_unknown += st["unknown_hand"]
        total_all += st["total"]

    lr = total_left / max(total_right, 1)
    print(f"\n  TOTAL: {total_all}  Left={total_left}  Right={total_right}  "
          f"no_hand={total_nohand}  unknown={total_unknown}  L/R={lr:.2f}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
