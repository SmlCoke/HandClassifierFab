"""Tests for dataset splitting logic."""

import os
import pytest
import numpy as np

from hand_classifier.dataset import split_dataset, compute_class_weights
from hand_classifier.parser import parse_cvat_xml


def _make_samples(source_counts, seed=42):
    """Create synthetic samples for split testing.

    Args:
        source_counts: dict of {source_name: (n_left, n_right)}

    Returns:
        list of sample dicts.
    """
    samples = []
    idx = 0
    for src, (n_left, n_right) in source_counts.items():
        for _ in range(n_left):
            samples.append({
                "image_path": f"/fake/{src}/img_{idx}.png",
                "label": 0,
                "source": src,
            })
            idx += 1
        for _ in range(n_right):
            samples.append({
                "image_path": f"/fake/{src}/img_{idx}.png",
                "label": 1,
                "source": src,
            })
            idx += 1
    return samples


@pytest.fixture
def config_no_test():
    return {
        "data": {
            "split": {
                "val_ratio": 0.1,
                "test_ratio": 0.0,
                "seed": 42,
                "stratified": True,
            }
        }
    }


@pytest.fixture
def config_with_test():
    return {
        "data": {
            "split": {
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "seed": 42,
                "stratified": True,
            }
        }
    }


def test_split_no_test(config_no_test):
    """All samples go to train/val, no test set."""
    samples = _make_samples({"srcA": (100, 50), "srcB": (80, 40)})
    train, val, test = split_dataset(samples, config_no_test)

    assert len(test) == 0
    total = len(train) + len(val)
    assert total == len(samples), f"Expected {len(samples)}, got {total}"


def test_split_with_test(config_with_test):
    """Test set is split off when test_ratio > 0."""
    samples = _make_samples({"srcA": (100, 50), "srcB": (80, 40)})
    train, val, test = split_dataset(samples, config_with_test)

    assert len(test) > 0, "Test set should not be empty"
    total = len(train) + len(val) + len(test)
    assert total == len(samples)


def test_split_deterministic(config_no_test):
    """Same seed produces same split."""
    samples = _make_samples({"srcA": (100, 50)})
    train1, val1, _ = split_dataset(samples, config_no_test)
    train2, val2, _ = split_dataset(samples, config_no_test)

    paths1 = sorted([s["image_path"] for s in train1])
    paths2 = sorted([s["image_path"] for s in train2])
    assert paths1 == paths2


def test_split_preserves_labels(config_no_test):
    """Check that both classes appear in train and val."""
    samples = _make_samples({"srcA": (100, 50)})
    train, val, _ = split_dataset(samples, config_no_test)

    train_labels = [s["label"] for s in train]
    val_labels = [s["label"] for s in val]

    assert 0 in train_labels
    assert 1 in train_labels
    assert 0 in val_labels
    assert 1 in val_labels


def test_compute_class_weights():
    """Balanced weights: more samples → lower weight."""
    samples = [{"label": 0}] * 100 + [{"label": 1}] * 50
    weights = compute_class_weights(samples, num_classes=2)
    # Class 0 has more samples → lower weight
    assert weights[0].item() < weights[1].item(), (
        f"Class 0 (100 samples) should have lower weight than class 1 (50 samples), "
        f"got {weights.tolist()}"
    )


def test_compute_class_weights_balanced():
    """Equal classes → equal weights."""
    samples = [{"label": 0}] * 50 + [{"label": 1}] * 50
    weights = compute_class_weights(samples, num_classes=2)
    assert abs(weights[0].item() - weights[1].item()) < 1e-6, (
        f"Weights should be equal, got {weights.tolist()}"
    )


def test_split_small_source(config_no_test):
    """Single-sample source should go to train."""
    samples = _make_samples({"srcA": (1, 0)})
    train, val, test = split_dataset(samples, config_no_test)
    assert len(train) == 1
    assert len(val) == 0
    assert len(test) == 0


def test_split_real_data(example_xml, example_images, config_no_test):
    """Test split on real data from example dataset."""
    samples = parse_cvat_xml(example_xml, example_images)
    # Add source info (parser doesn't add it directly)
    for s in samples:
        s["source"] = "dataset1"

    train, val, test = split_dataset(samples, config_no_test)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) == 0
    assert len(train) + len(val) == len(samples)
