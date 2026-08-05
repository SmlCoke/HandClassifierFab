.PHONY: setup dataset-stats train evaluate export-onnx cvat-label-test test all clean

# Default configuration
CONFIG ?= configs/hand_classifier.yaml
CHECKPOINT ?= outputs/checkpoints/best.pth

setup:
	pip install -r requirements.txt

dataset-stats:
	python scripts/dataset_stats.py --config $(CONFIG)

train:
	python scripts/train.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

export-onnx:
	python scripts/export_onnx.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

cvat-label-test:
	python scripts/cvat_label_test.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

test:
	python -m pytest tests/ -v

all: test train evaluate export-onnx cvat-label-test

clean:
	rm -rf outputs/ __pycache__/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
