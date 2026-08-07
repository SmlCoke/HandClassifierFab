"""Tests for dataset splitting logic (dual-head version)."""

import os
import pytest
import numpy as np

from hand_classifier.dataset import split_dataset, compute_class_weights
from hand_classifier.parser import parse_cvat_xml


def _make_samples(source_counts, seed=42):
    """Create synthetic dual-label samples.

    Args:
        source_counts: dict of {source_name: (n_has_hand, n_no_hand)}
    """
    samples = []
    idx = 0
    for src, (n_has, n_no) in source_counts.items():
        for _ in range(n_has):
            samples.append({
                "image_path": f"/fake/{src}/img_{idx}.png",
                "handedness_label": 0 if idx % 2 == 0 else 1,
                "presence_label": 1,
                "source": src,
            })
            idx += 1
        for _ in range(n_no):
            samples.append({
                "image_path": f"/fake/{src}/img_{idx}.png",
                "handedness_label": -1,
                "presence_label": 0,
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
    samples = _make_samples({"srcA": (100, 20), "srcB": (80, 10)})
    train, val, test = split_dataset(samples, config_no_test)
    assert len(test) == 0
    assert len(train) + len(val) == len(samples)


def test_split_with_test(config_with_test):
    samples = _make_samples({"srcA": (100, 20), "srcB": (80, 10)})
    train, val, test = split_dataset(samples, config_with_test)
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(samples)


def test_split_deterministic(config_no_test):
    samples = _make_samples({"srcA": (100, 20)})
    train1, val1, _ = split_dataset(samples, config_no_test)
    train2, val2, _ = split_dataset(samples, config_no_test)
    paths1 = sorted([s["image_path"] for s in train1])
    paths2 = sorted([s["image_path"] for s in train2])
    assert paths1 == paths2


def test_split_preserves_labels(config_no_test):
    samples = _make_samples({"srcA": (100, 20)})
    train, val, _ = split_dataset(samples, config_no_test)

    train_p = [s["presence_label"] for s in train]
    val_p = [s["presence_label"] for s in val]
    assert 0 in train_p
    assert 1 in train_p
    assert 0 in val_p
    assert 1 in val_p


def test_compute_class_weights():
    samples = (
        [{"presence_label": 1, "handedness_label": 0}] * 100 +
        [{"presence_label": 1, "handedness_label": 1}] * 50 +
        [{"presence_label": 0, "handedness_label": -1}] * 30
    )
    weights = compute_class_weights(samples, num_classes=2, task="handedness")
    assert weights[0].item() < weights[1].item(), (
        f"Class 0 (100) < class 1 (50): {weights.tolist()}"
    )


def test_compute_class_weights_balanced():
    samples = (
        [{"presence_label": 1, "handedness_label": 0}] * 50 +
        [{"presence_label": 1, "handedness_label": 1}] * 50
    )
    weights = compute_class_weights(samples, num_classes=2, task="handedness")
    assert abs(weights[0].item() - weights[1].item()) < 1e-6


def test_split_small_source(config_no_test):
    samples = [{
        "image_path": "/fake/img_0.png",
        "handedness_label": 0,
        "presence_label": 1,
        "source": "srcA",
    }]
    train, val, test = split_dataset(samples, config_no_test)
    assert len(train) == 1
    assert len(val) == 0


def test_split_real_data(example_xml, example_images, config_no_test):
    """Test split on real data from example dataset."""
    samples = parse_cvat_xml(example_xml, example_images)
    for s in samples:
        s["source"] = "dataset1"
    train, val, test = split_dataset(samples, config_no_test)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) == 0
    assert len(train) + len(val) == len(samples)
