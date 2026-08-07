"""PyTorch Dataset, transforms, and data splitting for dual-head classifier.

Each sample returns (image, handedness_label, presence_label).
"""

import json
import random
import logging
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


class HandROIDataset(Dataset):
    """PyTorch Dataset for dual-head Hand ROI classification.

    Returns:
        tuple: (image, handedness_label, presence_label)
          - handedness_label: -1 (ignore), 0 (Left), 1 (Right)
          - presence_label: 0 (no_hand) or 1 (has_hand)
    """

    def __init__(self, samples, transform=None, flip_prob=0.0):
        self.samples = samples
        self.transform = transform
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"])
        handedness = sample["handedness_label"]
        presence = sample["presence_label"]

        # Horizontal flip with label swap (only for positive handedness)
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if handedness >= 0:
                handedness = 1 - handedness  # Swap 0↔1

        if self.transform:
            image = self.transform(image)

        return image, handedness, presence


def get_transforms(is_train, config=None):
    """Build transform pipeline.

    Args:
        is_train: If True, include augmentations.
        config: Optional config dict with augmentation settings.

    Returns:
        torchvision.transforms.Compose
    """
    aug_cfg = config.get("augmentation", {}) if config else {}

    if is_train:
        transform = T.Compose([
            T.RandomAffine(
                degrees=aug_cfg.get("rotation_degrees", 10),
                translate=aug_cfg.get("translate", (0.1, 0.1)),
                scale=aug_cfg.get("scale", (0.9, 1.1)),
            ),
            T.ColorJitter(
                brightness=aug_cfg.get("brightness", 0.2),
                contrast=aug_cfg.get("contrast", 0.2),
            ),
            T.ToTensor(),
            T.RandomErasing(
                p=aug_cfg.get("random_erasing_prob", 0.3),
                scale=(0.02, 0.1),
            ),
            T.Normalize(mean=[0.485], std=[0.229]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229]),
        ])

    return transform


def split_dataset(samples, config):
    """Split samples into train/val (and optionally test) sets.

    Splits preserve presence_label distribution per source.

    Args:
        samples: List of sample dicts.
        config: Full config dict.

    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    split_cfg = config["data"]["split"]
    seed = split_cfg.get("seed", 42)
    val_ratio = split_cfg.get("val_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.0)
    stratified = split_cfg.get("stratified", True)

    by_source = defaultdict(list)
    for s in samples:
        by_source[s["source"]].append(s)

    train_samples, val_samples, test_samples = [], [], []

    for source_name, source_samples in by_source.items():
        n = len(source_samples)
        if n < 2:
            train_samples.extend(source_samples)
            continue

        remaining = source_samples

        if test_ratio > 0 and n >= 3:
            if stratified:
                labels = [s["presence_label"] for s in remaining]
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=test_ratio, random_state=seed
                )
                trainval_idx, test_idx = next(sss.split(remaining, labels))
            else:
                rng = np.random.RandomState(seed)
                idx = rng.permutation(n)
                split_point = int(n * test_ratio)
                trainval_idx, test_idx = idx[split_point:], idx[:split_point]

            test_samples.extend([remaining[i] for i in test_idx])
            remaining = [remaining[i] for i in trainval_idx]

        n_remain = len(remaining)
        if val_ratio > 0 and n_remain >= 2:
            if stratified:
                labels = [s["presence_label"] for s in remaining]
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=val_ratio, random_state=seed
                )
                train_idx, val_idx = next(sss.split(remaining, labels))
            else:
                rng = np.random.RandomState(seed)
                idx = rng.permutation(n_remain)
                split_point = max(1, int(n_remain * (1 - val_ratio)))
                train_idx, val_idx = idx[:split_point], idx[split_point:]

            train_samples.extend([remaining[i] for i in train_idx])
            val_samples.extend([remaining[i] for i in val_idx])
        else:
            train_samples.extend(remaining)

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_samples), len(val_samples), len(test_samples),
    )
    return train_samples, val_samples, test_samples


def save_split_info(train_samples, val_samples, test_samples, output_path):
    """Save split metadata to JSON for reproducibility."""
    def _summarize(split_samples):
        if not split_samples:
            return {"total": 0}
        sources = defaultdict(lambda: {
            "Left": 0, "Right": 0, "no_hand": 0, "unknown_hand": 0, "total": 0
        })
        for s in split_samples:
            src = s["source"]
            if s["presence_label"] == 0:
                sources[src]["no_hand"] += 1
            elif s["handedness_label"] == 0:
                sources[src]["Left"] += 1
            elif s["handedness_label"] == 1:
                sources[src]["Right"] += 1
            elif s["handedness_label"] == -1:
                sources[src]["unknown_hand"] += 1
            sources[src]["total"] += 1
        return {
            "total": len(split_samples),
            "per_source": {k: dict(v) for k, v in sources.items()},
        }

    info = {
        "train": _summarize(train_samples),
        "val": _summarize(val_samples),
        "test": _summarize(test_samples),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logger.info("Split info saved to %s", output_path)


def compute_class_weights(train_samples, num_classes=2, task="handedness"):
    """Compute balanced class weights for a specific task.

    Args:
        train_samples: List of training sample dicts.
        num_classes: Number of classes.
        task: "handedness" or "hand_presence".

    Returns:
        torch.Tensor: Class weights of shape (num_classes,).
    """
    if task == "handedness":
        labels = [s["handedness_label"] for s in train_samples
                  if s["handedness_label"] >= 0]
    else:
        labels = [s["presence_label"] for s in train_samples]

    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
