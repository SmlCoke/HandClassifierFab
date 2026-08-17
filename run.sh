#!/bin/bash
# One-shot pipeline: verify -> stats -> train -> evaluate -> export.
# The best-checkpoint path is captured from the training log and passed to
# evaluate/export, so the artifacts of the model just trained (whatever
# model.version / model.architecture configs/train.yaml selects) are
# evaluated and exported. evaluate.py / export_onnx.py auto-align their
# model config from the checkpoint (see align_config_to_checkpoint), so
# evaluate.yaml / export_onnx.yaml do not need to be kept in sync.
#
# NOTE: keep this file with LF line endings. CRLF breaks bash scripts
# ($'\r' errors); on Windows use "git config core.autocrlf true" + a
# checkout, or convert with: sed -i 's/\r$//' run.sh

# 01 data verification
make dataset-verify

# 02 data stats
python scripts/dataset_stats.py --config configs/train.yaml

# 03 train (capture the best checkpoint path from the training log)
TRAIN_LOG=$(mktemp)
python scripts/train.py --config configs/train.yaml 2>&1 | tee "$TRAIN_LOG"
CKPT=$(sed -n 's/^Best checkpoint: //p' "$TRAIN_LOG" | tail -1)
rm -f "$TRAIN_LOG"

if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
  echo
  echo ">>> Best checkpoint: $CKPT"

  # 04 evaluate
  python scripts/evaluate.py --config configs/evaluate.yaml --checkpoint "$CKPT"

  # 05 export
  python scripts/export_onnx.py --config configs/export_onnx.yaml --checkpoint "$CKPT"
else
  echo "WARNING: training did not produce a checkpoint; skipping evaluate/export" >&2
  exit 1
fi
