"""Tests for per-batch stratified sampling (hand presence / handedness ratios)."""

import pytest
import torch

from hand_classifier.dataset import (
    StratifiedBatchSampler, compute_target_weights,
)


def _make_samples(n_neg, n_left, n_right, n_unknown=0):
    samples = []
    for _ in range(n_neg):
        samples.append({"presence_label": 0, "handedness_label": -1})
    for _ in range(n_left):
        samples.append({"presence_label": 1, "handedness_label": 0})
    for _ in range(n_right):
        samples.append({"presence_label": 1, "handedness_label": 1})
    for _ in range(n_unknown):
        samples.append({"presence_label": 1, "handedness_label": -1})
    return samples


def test_batch_composition_exact():
    """Every batch must contain the exact target no_hand / Left / Right counts."""
    samples = _make_samples(n_neg=500, n_left=300, n_right=300, n_unknown=10)
    sampler = StratifiedBatchSampler(
        samples, batch_size=64, no_hand_ratio=0.3,
        left_right_ratio=[0.5, 0.5], seed=42,
    )
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 64
        neg = sum(1 for i in batch if samples[i]["presence_label"] == 0)
        pos = 64 - neg
        left = sum(1 for i in batch if samples[i]["handedness_label"] == 0)
        right = sum(1 for i in batch if samples[i]["handedness_label"] == 1)
        assert neg == 19  # round(64 * 0.3)
        assert pos == 45
        assert left in (22, 23) and right == 45 - left


def test_epoch_covers_all_positives():
    """One epoch = one pass over the has_hand pool."""
    samples = _make_samples(n_neg=2000, n_left=200, n_right=200)
    sampler = StratifiedBatchSampler(
        samples, batch_size=32, no_hand_ratio=0.25,
        left_right_ratio=[0.5, 0.5], seed=7,
    )
    batches = list(sampler)
    # pos per batch = 32 - 8 = 24; 400 // 24 = 16 batches
    assert len(batches) == 16
    seen_pos = {
        i for b in batches for i in b
        if samples[i]["presence_label"] == 1
    }
    assert len(seen_pos) >= 384  # at most one batch worth of leftovers


def test_no_hand_ratio_zero():
    """no_hand_ratio=0 → batches contain no negatives at all."""
    samples = _make_samples(n_neg=100, n_left=100, n_right=100)
    sampler = StratifiedBatchSampler(
        samples, batch_size=32, no_hand_ratio=0.0,
        left_right_ratio=[0.5, 0.5], seed=1,
    )
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert all(samples[i]["presence_label"] == 1 for i in batch)


def test_no_negatives_at_all():
    """Dataset without negatives: batches are all has_hand."""
    samples = _make_samples(n_neg=0, n_left=50, n_right=50)
    sampler = StratifiedBatchSampler(
        samples, batch_size=32, no_hand_ratio=0.3,
        left_right_ratio=[0.5, 0.5], seed=1,
    )
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 32
        assert all(samples[i]["presence_label"] == 1 for i in batch)


def test_empty_left_pool_redistributes():
    """No Left samples: the has_hand quota shifts entirely to Right."""
    samples = _make_samples(n_neg=100, n_left=0, n_right=300, n_unknown=20)
    sampler = StratifiedBatchSampler(
        samples, batch_size=32, no_hand_ratio=0.25,
        left_right_ratio=[0.5, 0.5], seed=1,
    )
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 32
        neg = sum(1 for i in batch if samples[i]["presence_label"] == 0)
        assert neg == 8  # round(32 * 0.25)
        assert all(samples[i]["handedness_label"] != 0
                   for i in batch if samples[i]["presence_label"] == 1)


def test_invalid_ratio_raises():
    samples = _make_samples(n_neg=10, n_left=10, n_right=10)
    with pytest.raises(ValueError, match="no_hand_ratio"):
        StratifiedBatchSampler(samples, 32, no_hand_ratio=1.5)
    with pytest.raises(ValueError, match="left_right_ratio"):
        StratifiedBatchSampler(samples, 32, left_right_ratio=[0.0, 0.0])


def test_deterministic_with_seed():
    """Same seed → same batch composition sequence."""
    samples = _make_samples(n_neg=300, n_left=200, n_right=200)
    a = list(StratifiedBatchSampler(
        samples, 32, no_hand_ratio=0.3,
        left_right_ratio=[0.5, 0.5], seed=99,
    ))
    b = list(StratifiedBatchSampler(
        samples, 32, no_hand_ratio=0.3,
        left_right_ratio=[0.5, 0.5], seed=99,
    ))
    assert a == b


def test_compute_target_weights():
    """Weights derive from the target ratios (not raw counts)."""
    w = compute_target_weights({"no_hand_ratio": 0.3})
    # inverse frequency of [0.7, 0.3], normalized to sum 2
    expected = torch.tensor([1.0 / 0.7, 1.0 / 0.3])
    expected = expected / expected.sum() * 2
    assert torch.allclose(w, expected)
    assert w[0] < w[1]  # no_hand class gets the larger weight
