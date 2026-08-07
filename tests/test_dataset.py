"""Tests for dual-head HandROIDataset and transforms."""

import os
import pytest
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from hand_classifier.dataset import HandROIDataset, get_transforms
from hand_classifier.parser import parse_cvat_xml


@pytest.fixture
def sample_data(example_xml, example_images):
    """Load samples from example dataset (dual labels)."""
    return parse_cvat_xml(example_xml, example_images)


@pytest.fixture
def synthetic_samples(tmp_path):
    """Create synthetic samples with both labels."""
    samples = []
    for i in range(10):
        img = Image.new("L", (256, 256), color=i * 25)
        path = os.path.join(str(tmp_path), f"img_{i}.png")
        img.save(path)
        samples.append({
            "image_path": path,
            "handedness_label": i % 2,          # alternating Left/Right
            "presence_label": 1 if i < 8 else 0,  # 8 has_hand, 2 no_hand
        })
    return samples


def test_dataset_len(sample_data):
    dataset = HandROIDataset(sample_data)
    assert len(dataset) == len(sample_data)


def test_dataset_getitem(sample_data):
    dataset = HandROIDataset(sample_data, transform=T.ToTensor())
    img, h_label, p_label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 256, 256)
    assert isinstance(h_label, int)
    assert isinstance(p_label, int)
    assert p_label in (0, 1)


def test_dataset_getitem_no_transform(sample_data):
    dataset = HandROIDataset(sample_data)
    img, h_label, p_label = dataset[0]
    assert isinstance(img, Image.Image)
    assert img.size == (256, 256)


def test_train_transform():
    transform = get_transforms(is_train=True)
    img = Image.new("L", (256, 256), color=128)
    tensor = transform(img)
    assert tensor.shape == (1, 256, 256)
    # Should have been normalized (mean ~ -0.8 if all pixels are 128)
    assert abs(tensor.mean().item()) > 0.01


def test_val_transform():
    transform = get_transforms(is_train=False)
    img = Image.new("L", (256, 256), color=128)
    tensor = transform(img)
    assert tensor.shape == (1, 256, 256)


def test_synthetic_dataset(synthetic_samples):
    dataset = HandROIDataset(synthetic_samples, transform=T.ToTensor())
    h_labels = []
    p_labels = []
    for i in range(len(dataset)):
        _, h, p = dataset[i]
        h_labels.append(h)
        p_labels.append(p)
    assert sum(1 for p in p_labels if p == 1) == 8
    assert sum(1 for p in p_labels if p == 0) == 2


def test_flip_label_swap(synthetic_samples):
    """Horizontal flip with high prob swaps Left↔Right, preserves presence."""
    dataset = HandROIDataset(
        synthetic_samples, transform=T.ToTensor(), flip_prob=1.0,
    )
    for i in range(len(dataset)):
        _, h, p = dataset[i]
        if h >= 0:
            assert h == 1 - (synthetic_samples[i]["handedness_label"])
        assert p == synthetic_samples[i]["presence_label"]


def test_flip_no_swap_at_zero_prob(synthetic_samples):
    dataset = HandROIDataset(
        synthetic_samples, transform=T.ToTensor(), flip_prob=0.0,
    )
    for i in range(len(dataset)):
        _, h, p = dataset[i]
        assert h == synthetic_samples[i]["handedness_label"]
        assert p == synthetic_samples[i]["presence_label"]


def test_normalize_range(sample_data):
    """Normalized tensor values should be in a reasonable range."""
    transform = get_transforms(is_train=False)
    dataset = HandROIDataset(sample_data, transform=transform)
    img, _, _ = dataset[0]
    # ImageNet grayscale normalization: mean=0.485, std=0.229
    # Grayscale values are 0-255, so normalized range should be roughly [-2.1, 2.2]
    assert -3.0 < img.min() < 0, f"min={img.min()}"
    assert 0 < img.max() < 4.0, f"max={img.max()}"
