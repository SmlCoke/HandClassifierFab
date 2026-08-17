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


class StratifiedBatchSampler:
    """Per-batch class-ratio sampler for training.

    The no_hand pool of the current dataset is much larger than the
    has_hand pool, so plain random sampling would make most batches
    dominated by no_hand samples and skew the hand presence head.
    This sampler enforces a fixed target composition for **every** batch:

      - ``round(batch_size * no_hand_ratio)`` no_hand samples
      - the remaining ``batch_size - n_no_hand`` has_hand samples,
        split between Left / Right according to ``left_right_ratio``
        (has_hand samples with unknown handedness are used as fallback)

    Each pool is shuffled at the start of every epoch; if a pool is
    exhausted before the epoch ends, it is reshuffled and reused so the
    per-batch composition stays exact. Only full batches are emitted
    (equivalent to ``drop_last=True``). Epoch length is one pass over
    the has_hand pool (``len(has_hand) // n_has_hand_per_batch``).

    Args:
        samples: List of sample dicts (same format as HandROIDataset).
        batch_size: Batch size.
        no_hand_ratio: Target fraction of no_hand samples per batch (0~1).
        left_right_ratio: Target Left/Right split within the has_hand
            part, e.g. [0.5, 0.5].
        seed: Random seed for shuffling (per-epoch).
    """

    def __init__(self, samples, batch_size, no_hand_ratio=0.3,
                 left_right_ratio=(0.5, 0.5), seed=None):
        if not (0.0 <= no_hand_ratio <= 1.0):
            raise ValueError(f"no_hand_ratio must be in [0, 1], got {no_hand_ratio}")
        if len(left_right_ratio) != 2 or sum(left_right_ratio) <= 0:
            raise ValueError(
                f"left_right_ratio must be two positive numbers, got {left_right_ratio}"
            )
        self.batch_size = int(batch_size)
        self.no_hand_ratio = float(no_hand_ratio)
        lr_sum = float(sum(left_right_ratio))
        self.left_ratio = float(left_right_ratio[0]) / lr_sum
        self.rng = random.Random(seed)

        self.pools = {"neg": [], "left": [], "right": [], "unknown": []}
        for idx, s in enumerate(samples):
            if s["presence_label"] == 0:
                self.pools["neg"].append(idx)
            elif s["handedness_label"] == 0:
                self.pools["left"].append(idx)
            elif s["handedness_label"] == 1:
                self.pools["right"].append(idx)
            else:
                self.pools["unknown"].append(idx)

        self.n_pos_total = sum(len(self.pools[k])
                               for k in ("left", "right", "unknown"))

    def __len__(self):
        """Number of full batches in one epoch (one pass over has_hand)."""
        n_neg = self._n_neg() if self.pools["neg"] else 0
        n_pos = self.batch_size - n_neg
        if self.n_pos_total == 0 or n_pos <= 0:
            return 0
        return self.n_pos_total // n_pos

    def _n_neg(self):
        return int(round(self.batch_size * self.no_hand_ratio))

    def _pool_iter(self, key):
        """Infinite iterator over a shuffled pool (reshuffles when empty)."""
        pool = self.pools[key]
        if not pool:
            return iter(())
        while True:
            self.rng.shuffle(pool)
            yield from pool

    def __iter__(self):
        iters = {k: self._pool_iter(k) for k in self.pools}

        # Effective quotas: if a pool is empty its quota is redistributed
        # to the remaining classes so every batch stays full.
        n_neg = self._n_neg() if self.pools["neg"] else 0
        n_pos = self.batch_size - n_neg
        if self.pools["left"] and self.pools["right"]:
            n_left = int(n_pos * self.left_ratio + 0.5)  # round-half-up
            n_right = n_pos - n_left
        elif self.pools["left"]:
            n_left, n_right = n_pos, 0
        elif self.pools["right"]:
            n_left, n_right = 0, n_pos
        else:
            n_left, n_right = 0, 0

        for _ in range(len(self)):
            batch = []
            if n_neg > 0:
                batch.extend(next(iters["neg"]) for _ in range(n_neg))

            # has_hand quota: Left/Right first, then any remaining
            # has_hand (unknown-handedness) samples as fallback
            remaining_pos = n_pos
            for key, quota in (("left", n_left), ("right", n_right)):
                take = min(quota, remaining_pos)
                if take > 0 and self.pools[key]:
                    batch.extend(next(iters[key]) for _ in range(take))
                    remaining_pos -= take
            while remaining_pos > 0 and self.n_pos_total > 0:
                for key in ("left", "right", "unknown"):
                    if self.pools[key]:
                        batch.append(next(iters[key]))
                        remaining_pos -= 1
                        break
                else:
                    break  # no has_hand samples at all

            if len(batch) == self.batch_size:
                yield batch


def compute_target_weights(sampling_cfg, num_classes=2):
    """Inverse-frequency class weights derived from target sampling ratios.

    When per-batch sampling is enabled, the effective training
    distribution is the target ratio (not the raw dataset counts), so
    class weights must be computed from the ratios to avoid
    double-compensating the imbalance.

    Returns:
        torch.Tensor: (has_hand_weight, no_hand_weight) of shape (2,).
    """
    neg_ratio = float(sampling_cfg.get("no_hand_ratio", 0.3))
    pos_ratio = 1.0 - neg_ratio
    weights = torch.tensor([1.0 / pos_ratio, 1.0 / neg_ratio],
                           dtype=torch.float32)
    return weights / weights.sum() * num_classes
