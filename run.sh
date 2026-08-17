#!/bin/bash


# 01 data verification
make dataset-verify

# 02 data stats
python scripts/dataset_stats.py --config configs/train.yaml

# 03 train
python scripts/train.py --config configs/train.yaml

# 04 evaulate
python scripts/evaluate.py --config configs/evaluate.yaml

# 05 export
python scripts/export_onnx.py --config configs/export_onnx.yaml