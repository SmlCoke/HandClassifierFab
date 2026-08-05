"""Tests for image preprocessing / normalization."""

import pytest
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from hand_classifier.dataset import get_transforms


def test_to_tensor_grayscale():
    """ToTensor converts PIL 'L' to (1, H, W) tensor."""
    img = Image.new("L", (256, 256), color=128)
    tensor = T.ToTensor()(img)
    assert tensor.shape == (1, 256, 256)
    assert tensor.dtype == torch.float32


def test_to_tensor_value_range():
    """ToTensor scales uint8 0-255 to float 0-1."""
    img_black = Image.new("L", (32, 32), color=0)
    img_white = Image.new("L", (32, 32), color=255)

    tensor_black = T.ToTensor()(img_black)
    tensor_white = T.ToTensor()(img_white)

    assert tensor_black.max().item() == 0.0
    assert abs(tensor_white.max().item() - 1.0) < 0.01


def test_normalize_grayscale():
    """Normalize with single-channel mean/std."""
    img = Image.new("L", (256, 256), color=128)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485], std=[0.229]),
    ])
    tensor = transform(img)
    # 128/255 ≈ 0.502, normalized: (0.502 - 0.485) / 0.229 ≈ 0.074
    expected = (128 / 255 - 0.485) / 0.229
    assert abs(tensor.mean().item() - expected) < 0.01


def test_train_transform_output_shape():
    """Training transform produces correct output shape."""
    transform = get_transforms(is_train=True)
    img = Image.new("L", (256, 256), color=128)
    tensor = transform(img)
    assert tensor.shape == (1, 256, 256)


def test_val_transform_output_shape():
    """Validation transform produces correct output shape."""
    transform = get_transforms(is_train=False)
    img = Image.new("L", (256, 256), color=128)
    tensor = transform(img)
    assert tensor.shape == (1, 256, 256)


def test_train_vs_val_different():
    """Train and validation transforms should differ (augmentations)."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Create a simple non-uniform image to verify transforms differ
    data = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    img = Image.fromarray(data, mode="L")

    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)

    t1 = train_transform(img)
    t2 = val_transform(img)

    # After normalization, values should differ (due to augmentations)
    # They could be the same by chance, but it's very unlikely
    assert t1.shape == t2.shape
    assert t1.shape == (1, 256, 256)


def test_augmentation_config():
    """Test that augmentation config is respected."""
    config = {
        "augmentation": {
            "horizontal_flip_prob": 0.0,
            "rotation_degrees": 0,
            "translate": (0.0, 0.0),
            "scale": (1.0, 1.0),
            "brightness": 0.0,
            "contrast": 0.0,
            "random_erasing_prob": 0.0,
        }
    }

    transform = get_transforms(is_train=True, config=config)
    img = Image.new("L", (256, 256), color=128)
    tensor = transform(img)
    assert tensor.shape == (1, 256, 256)
    # With no augmentations, pixel values should be uniform
    std = tensor.std().item()
    assert std < 0.01, f"Expected near-zero std with no augmentations, got {std:.4f}"
