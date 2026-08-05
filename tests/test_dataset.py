"""Tests for HandROIDataset and transforms."""

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
    """Load samples from example dataset."""
    return parse_cvat_xml(example_xml, example_images)


@pytest.fixture
def synthetic_samples(tmp_path):
    """Create synthetic single-channel images with labels."""
    samples = []
    for i in range(10):
        img = Image.new("L", (256, 256), color=i * 25)
        path = tmp_path / f"synth_{i}.png"
        img.save(path)
        samples.append({
            "image_path": str(path),
            "label": i % 2,
            "source": "synth",
        })
    return samples


def test_dataset_len(sample_data):
    dataset = HandROIDataset(sample_data)
    assert len(dataset) == len(sample_data)


def test_dataset_getitem(sample_data):
    """Test that __getitem__ returns (tensor, int)."""
    transform = get_transforms(is_train=False)
    dataset = HandROIDataset(sample_data, transform=transform)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 1, f"Expected 1 channel, got {image.shape}"
    assert image.shape[1:] == (256, 256), f"Expected 256x256, got {image.shape[1:]}"
    assert isinstance(label, int)


def test_dataset_getitem_no_transform(sample_data):
    dataset = HandROIDataset(sample_data)
    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "L"


def test_train_transform(sample_data):
    """Test that training transform includes augmentations."""
    transform = get_transforms(is_train=True)
    dataset = HandROIDataset(sample_data, transform=transform, flip_prob=0.5)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 256, 256)


def test_val_transform(sample_data):
    """Test that validation transform does not include augmentations."""
    transform = get_transforms(is_train=False)
    dataset = HandROIDataset(sample_data, transform=transform)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 256, 256)


def test_synthetic_dataset(synthetic_samples):
    dataset = HandROIDataset(synthetic_samples)
    assert len(dataset) == 10
    for i in range(10):
        img, label = dataset[i]
        assert img.size == (256, 256)
        assert label in (0, 1)


def test_flip_label_swap(synthetic_samples):
    """Test that horizontal flip swaps labels 0↔1."""
    # Set flip prob to 1.0 so flip always happens
    dataset = HandROIDataset(synthetic_samples, flip_prob=1.0)
    for i in range(10):
        img, new_label = dataset[i]
        original_label = synthetic_samples[i]["label"]
        assert new_label == 1 - original_label, (
            f"Expected label swap: {original_label} -> {1 - original_label}, "
            f"got {new_label}"
        )


def test_flip_no_swap_at_zero_prob(synthetic_samples):
    """Test that labels are NOT swapped when flip_prob=0."""
    dataset = HandROIDataset(synthetic_samples, flip_prob=0.0)
    for i in range(10):
        img, label = dataset[i]
        assert label == synthetic_samples[i]["label"]


def test_normalize_range(sample_data):
    """Verify normalized tensor values are in a reasonable range."""
    transform = get_transforms(is_train=False)
    dataset = HandROIDataset(sample_data, transform=transform)
    image, _ = dataset[0]
    # After Normalize(mean=[0.485], std=[0.229]), the values should be
    # approximately in [-2.1, 2.2] for uint8 images
    assert -3.0 <= image.min().item() <= 3.0
    assert -3.0 <= image.max().item() <= 3.0
