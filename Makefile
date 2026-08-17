.PHONY: setup dataset-stats dataset-verify train evaluate export-onnx cvat-label-test infer test all clean

# Config files (one per functional target)
TRAIN_CONFIG       ?= configs/train.yaml
EVAL_CONFIG        ?= configs/evaluate.yaml
EXPORT_CONFIG      ?= configs/export_onnx.yaml
CVAT_CONFIG        ?= configs/cvat_label_test.yaml
INFER_CONFIG       ?= configs/infer.yaml
CHECKPOINT         ?= outputs/checkpoints/best.pth

setup:
	pip install -r requirements.txt

dataset-stats:
	python scripts/dataset_stats.py --config $(TRAIN_CONFIG)

dataset-verify:
	python scripts/verify_datasets.py --config $(TRAIN_CONFIG)

train:
	python scripts/train.py --config $(TRAIN_CONFIG)

evaluate:
	python scripts/evaluate.py --config $(EVAL_CONFIG) --checkpoint $(CHECKPOINT)

export-onnx:
	python scripts/export_onnx.py --config $(EXPORT_CONFIG) --checkpoint $(CHECKPOINT)

cvat-label-test:
	python scripts/cvat_label_test.py --config $(CVAT_CONFIG) --checkpoint $(CHECKPOINT)

infer:
	python scripts/infer.py --config $(INFER_CONFIG)

test:
	python -m pytest tests/ -v

all: test train evaluate export-onnx cvat-label-test

clean:
	rm -rf outputs/ __pycache__/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
