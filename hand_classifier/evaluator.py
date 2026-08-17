"""Evaluation for dual-head hand classifier."""

import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)

from hand_classifier.dataset import HandROIDataset, get_transforms
from hand_classifier.parser import collect_all_samples
from hand_classifier.config import resolve_output_paths, align_config_to_checkpoint

logger = logging.getLogger(__name__)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate_split(model, samples, config, device, split_name):
    """Evaluate model on a set of samples."""
    if not samples:
        return {}

    transform = get_transforms(is_train=False, config=config)
    dataset = HandROIDataset(samples, transform=transform)
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 4)
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model.eval()
    all_h_preds, all_h_labels = [], []
    all_p_preds, all_p_labels = [], []
    all_h_probs, all_p_probs = [], []

    with torch.no_grad():
        for images, h_labels, p_labels in tqdm(
            loader, desc=f"Eval {split_name}", leave=False
        ):
            images = images.to(device)
            outputs = model(images)  # No autocast: fp16 softmax → NaN

            h_prob = torch.softmax(outputs["handedness"], dim=1).cpu().numpy()
            p_prob = torch.softmax(outputs["hand_presence"], dim=1).cpu().numpy()
            h_pred = h_prob.argmax(axis=1)
            p_pred = p_prob.argmax(axis=1)

            all_h_probs.append(h_prob)
            all_p_probs.append(p_prob)
            all_h_preds.append(h_pred)
            all_p_preds.append(p_pred)
            all_h_labels.append(h_labels.numpy())
            all_p_labels.append(p_labels.numpy())

    h_preds = np.concatenate(all_h_preds)
    p_preds = np.concatenate(all_p_preds)
    h_labels = np.concatenate(all_h_labels)
    p_labels = np.concatenate(all_p_labels)
    h_probs = np.concatenate(all_h_probs)
    p_probs = np.concatenate(all_p_probs)

    results = {}

    # Hand presence metrics
    results["presence_accuracy"] = accuracy_score(p_labels, p_preds)
    p_prec, p_rec, p_f1, _ = precision_recall_fscore_support(
        p_labels, p_preds, average=None, labels=[0, 1], zero_division=0
    )
    results["presence_no_hand"] = {
        "precision": round(p_prec[0], 6), "recall": round(p_rec[0], 6),
        "f1": round(p_f1[0], 6), "n": int((p_labels == 0).sum()),
    }
    results["presence_has_hand"] = {
        "precision": round(p_prec[1], 6), "recall": round(p_rec[1], 6),
        "f1": round(p_f1[1], 6), "n": int((p_labels == 1).sum()),
    }
    if len(np.unique(p_labels)) == 2:
        results["presence_roc_auc"] = round(
            roc_auc_score(p_labels, p_probs[:, 1]), 6
        )
    results["presence_confusion"] = confusion_matrix(p_labels, p_preds).tolist()

    # Handedness metrics (only on valid labels)
    valid = h_labels >= 0
    if valid.sum() > 0:
        h_labels_v = h_labels[valid]
        h_preds_v = h_preds[valid]
        h_probs_v = h_probs[valid]

        results["handedness_accuracy"] = accuracy_score(h_labels_v, h_preds_v)
        h_prec, h_rec, h_f1, _ = precision_recall_fscore_support(
            h_labels_v, h_preds_v, average=None, labels=[0, 1], zero_division=0
        )
        results["handedness_left"] = {
            "precision": round(h_prec[0], 6), "recall": round(h_rec[0], 6),
            "f1": round(h_f1[0], 6), "n": int((h_labels_v == 0).sum()),
        }
        results["handedness_right"] = {
            "precision": round(h_prec[1], 6), "recall": round(h_rec[1], 6),
            "f1": round(h_f1[1], 6), "n": int((h_labels_v == 1).sum()),
        }
        if len(np.unique(h_labels_v)) == 2:
            results["handedness_roc_auc"] = round(
                roc_auc_score(h_labels_v, h_probs_v[:, 1]), 6
            )
        results["handedness_confusion"] = confusion_matrix(
            h_labels_v, h_preds_v
        ).tolist()

    # Per-source breakdown
    per_source = defaultdict(lambda: {"total": 0, "presence_correct": 0})
    for idx, s in enumerate(samples):
        src = s["source"]
        per_source[src]["total"] += 1
        if p_preds[idx] == p_labels[idx]:
            per_source[src]["presence_correct"] += 1

    results["per_source"] = {}
    for src, counts in sorted(per_source.items()):
        results["per_source"][src] = {
            "total": counts["total"],
            "presence_accuracy": round(
                counts["presence_correct"] / max(counts["total"], 1), 6
            ),
        }

    return results


def evaluate(config, checkpoint_path=None, output_dir=None):
    """Evaluate a trained dual-head model.

    Args:
        config: Configuration dict.
        checkpoint_path: Path to checkpoint (.pth). Auto-detected if None.
        output_dir: Directory for evaluation outputs. Auto-derived if None.

    Returns:
        dict: Evaluation results.
    """
    # Organize outputs as <output_root>/<version>/<architecture>/ when
    # paths.output_root is configured (no cross-model overwrites).
    config = resolve_output_paths(config)

    device = _get_device()

    # Locate and load the checkpoint first: its embedded training config
    # decides which model to rebuild (see align_config_to_checkpoint),
    # so evaluate/export always match the trained model even when the
    # config file still names another architecture.
    if checkpoint_path is None:
        paths_cfg = config.get("paths", {})
        checkpoint_path = (
            Path(paths_cfg.get("checkpoint_dir", "outputs/checkpoints"))
            / "best.pth"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config, _ = align_config_to_checkpoint(config, checkpoint)

    model_cfg = config["model"]
    from models.factory import build_model
    model = build_model(
        architecture=model_cfg["architecture"],
        pretrained=False,
        num_handedness=model_cfg.get("num_handedness", 2),
        num_presence=model_cfg.get("num_presence", 2),
        version=model_cfg.get("version", "v1"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint from %s (epoch %d)",
        checkpoint_path, checkpoint.get("epoch", -1),
    )

    if output_dir is None:
        paths_cfg = config.get("paths", {})
        output_dir = paths_cfg.get("eval_dir") or str(
            Path(paths_cfg.get("splits_dir", "outputs")).parent / "eval"
        )

    # Collect data
    data_cfg = config["data"]
    val_sources = data_cfg.get("val_sources", [])
    test_sources = data_cfg.get("test_sources", [])

    all_results = {}

    if val_sources:
        val_samples = collect_all_samples(val_sources)
        all_results["val"] = _evaluate_split(
            model, val_samples, config, device, "val"
        )
        _log_results(all_results["val"], "VAL")

    if test_sources:
        test_samples = collect_all_samples(test_sources)
        all_results["test"] = _evaluate_split(
            model, test_samples, config, device, "test"
        )
        _log_results(all_results["test"], "TEST")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for split_name, metrics in all_results.items():
            path = out / f"{split_name}_metrics.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info("Saved %s metrics to %s", split_name, path)

    return all_results


def _log_results(results, name):
    """Log evaluation results."""
    logger.info("=== %s Results ===", name)
    total = (results.get("presence_has_hand", {}).get("n", 0) +
             results.get("presence_no_hand", {}).get("n", 0))
    logger.info("  Samples: %d", total)

    if "presence_accuracy" in results:
        logger.info("  Presence Accuracy: %.4f", results["presence_accuracy"])
        logger.info("  Presence ROC-AUC: %.4f",
                    results.get("presence_roc_auc", 0))
        p = results["presence_no_hand"]
        logger.info("  no_hand  - P: %.4f, R: %.4f, F1: %.4f (n=%d)",
                    p["precision"], p["recall"], p["f1"], p["n"])
        p = results["presence_has_hand"]
        logger.info("  has_hand - P: %.4f, R: %.4f, F1: %.4f (n=%d)",
                    p["precision"], p["recall"], p["f1"], p["n"])
        logger.info("  Presence Confusion Matrix:\n%s",
                    np.array(results["presence_confusion"]))

    if "handedness_accuracy" in results:
        logger.info("  Handedness Accuracy: %.4f",
                    results["handedness_accuracy"])
        h = results["handedness_left"]
        logger.info("  Left  - P: %.4f, R: %.4f, F1: %.4f (n=%d)",
                    h["precision"], h["recall"], h["f1"], h["n"])
        h = results["handedness_right"]
        logger.info("  Right - P: %.4f, R: %.4f, F1: %.4f (n=%d)",
                    h["precision"], h["recall"], h["f1"], h["n"])
