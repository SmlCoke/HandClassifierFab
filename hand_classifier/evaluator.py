"""Model evaluation: accuracy, per-class metrics, confusion matrix, per-source breakdown."""

import json
import logging
from collections import defaultdict
from pathlib import Path

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
from hand_classifier.dataset import split_dataset

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Left", "Right"]


def evaluate(config, checkpoint_path=None, output_dir=None):
    """Evaluate model on val (and optionally test) sets.

    Args:
        config: Configuration dict.
        checkpoint_path: Path to model checkpoint (default: best.pth).
        output_dir: Directory for evaluation outputs.

    Returns:
        dict: Evaluation metrics.
    """
    # --- Load model ---
    model_cfg = config["model"]
    from models.factory import build_model
    model = build_model(
        architecture=model_cfg["architecture"],
        pretrained=False,
        num_classes=model_cfg.get("num_classes", 2),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load checkpoint
    paths_cfg = config.get("paths", {})
    if checkpoint_path is None:
        checkpoint_path = Path(paths_cfg.get("checkpoint_dir", "outputs/checkpoints")) / "best.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from %s (epoch %d, val_acc=%.4f)",
                checkpoint_path, checkpoint.get("epoch", -1),
                checkpoint.get("val_acc", 0.0))

    # --- Collect and load data ---
    data_cfg = config["data"]
    train_sources = data_cfg.get("train_sources", [])
    val_sources = data_cfg.get("val_sources", [])
    test_sources = data_cfg.get("test_sources", [])

    all_results = {}

    # Validation set
    if val_sources:
        val_samples = collect_all_samples(val_sources)
        val_metrics = _evaluate_split(
            model, val_samples, config, device, "val"
        )
        all_results["val"] = val_metrics

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "val_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2, ensure_ascii=False)
    else:
        # Split val from train sources
        all_samples = collect_all_samples(train_sources)
        _, val_samples, _ = split_dataset(all_samples, config)
        val_metrics = _evaluate_split(
            model, val_samples, config, device, "val"
        )
        all_results["val"] = val_metrics

    # Test set (if available)
    if test_sources:
        test_samples = collect_all_samples(test_sources)
        test_metrics = _evaluate_split(
            model, test_samples, config, device, "test"
        )
        all_results["test"] = test_metrics

        if output_dir:
            with open(Path(output_dir) / "test_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    return all_results


def _evaluate_split(model, samples, config, device, split_name):
    """Evaluate model on a given set of samples."""
    if len(samples) == 0:
        logger.warning("No samples for %s split", split_name)
        return {}

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 2)

    transform = get_transforms(is_train=False, config=config)
    dataset = HandROIDataset(samples, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    use_amp = train_cfg.get("amp", True) and device.type == "cuda"

    all_preds = []
    all_labels = []
    all_probs = []
    all_sources = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Eval {split_name}", leave=False):
            images = images.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    # Also track source info from the dataset (not from DataLoader batch)
    all_sources = [s["source"] for s in samples]

    return _compute_metrics(
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), all_sources, split_name,
    )


def _compute_metrics(y_true, y_pred, y_prob, sources, split_name):
    """Compute and print classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    metrics = {
        "split": split_name,
        "num_samples": len(y_true),
        "accuracy": round(float(acc), 6),
        "roc_auc": round(float(auc), 6),
        "per_class": {
            "Left": {
                "precision": round(float(precision[0]), 6),
                "recall": round(float(recall[0]), 6),
                "f1": round(float(f1[0]), 6),
                "support": int(support[0]),
            },
            "Right": {
                "precision": round(float(precision[1]), 6),
                "recall": round(float(recall[1]), 6),
                "f1": round(float(f1[1]), 6),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": cm.tolist(),
    }

    # Per-source breakdown
    if sources:
        per_source = defaultdict(lambda: {"total": 0, "correct": 0, "Left": 0, "Right": 0})
        for i in range(len(y_true)):
            src = sources[i]
            per_source[src]["total"] += 1
            if y_pred[i] == y_true[i]:
                per_source[src]["correct"] += 1
            if y_true[i] == 0:
                per_source[src]["Left"] += 1
            else:
                per_source[src]["Right"] += 1

        per_source_metrics = {}
        for src, stats in sorted(per_source.items()):
            per_source_metrics[src] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / max(stats["total"], 1), 6),
                "Left": stats["Left"],
                "Right": stats["Right"],
            }
        metrics["per_source"] = per_source_metrics

    # Print summary
    logger.info("=== %s Results ===", split_name.upper())
    logger.info("  Samples: %d", len(y_true))
    logger.info("  Accuracy: %.4f", acc)
    logger.info("  ROC-AUC: %.4f", auc)
    logger.info(
        "  Left  - P: %.4f, R: %.4f, F1: %.4f (n=%d)",
        precision[0], recall[0], f1[0], support[0],
    )
    logger.info(
        "  Right - P: %.4f, R: %.4f, F1: %.4f (n=%d)",
        precision[1], recall[1], f1[1], support[1],
    )
    logger.info("  Confusion Matrix:\n%s", cm)

    return metrics
