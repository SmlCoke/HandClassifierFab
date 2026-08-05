"""Training loop with AMP, warmup, cosine annealing, and early stopping."""

import os
import json
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
)

from hand_classifier.dataset import (
    HandROIDataset, get_transforms, compute_class_weights,
)
from hand_classifier.parser import collect_all_samples
from hand_classifier.dataset import split_dataset, save_split_info
from hand_classifier.evaluator import evaluate as run_evaluation

logger = logging.getLogger(__name__)


def _create_dataloaders(train_samples, val_samples, config):
    """Build train and val DataLoaders."""
    aug_cfg = config.get("augmentation", {})
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 4)

    train_transform = get_transforms(is_train=True, config=config)
    val_transform = get_transforms(is_train=False, config=config)

    train_dataset = HandROIDataset(
        train_samples, transform=train_transform,
        flip_prob=aug_cfg.get("horizontal_flip_prob", 0.5),
    )
    val_dataset = HandROIDataset(
        val_samples, transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


def train(config):
    """Run the full training pipeline.

    Args:
        config: Configuration dict.

    Returns:
        str: Path to the best checkpoint.
    """
    # --- Collect and split data ---
    logger.info("=== Collecting samples ===")
    data_cfg = config["data"]
    train_sources = data_cfg.get("train_sources", [])
    val_sources = data_cfg.get("val_sources", [])

    if val_sources:
        # Pre-defined val sources: collect separately, exclude from train
        val_samples = collect_all_samples(val_sources)
        val_source_names = set(s["source"] for s in val_samples)
        all_samples = collect_all_samples(train_sources)
        train_samples = [
            s for s in all_samples if s["source"] not in val_source_names
        ]
        test_samples = []
        logger.info(
            "Excluded %d val source(s) from train: %s",
            len(val_source_names), ", ".join(sorted(val_source_names)),
        )
    else:
        # Split from train sources
        all_samples = collect_all_samples(train_sources)
        train_samples, val_samples, test_samples = split_dataset(
            all_samples, config
        )

    # Save split info
    paths_cfg = config.get("paths", {})
    splits_dir = Path(paths_cfg.get("splits_dir", "outputs"))
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_split_info(
        train_samples, val_samples, test_samples,
        str(splits_dir / "splits.json"),
    )

    if len(train_samples) == 0:
        raise RuntimeError("No training samples found!")
    if len(val_samples) == 0:
        raise RuntimeError("No validation samples found!")

    # --- Build model ---
    logger.info("=== Building model ===")
    model_cfg = config["model"]
    from models.factory import build_model
    model = build_model(
        architecture=model_cfg["architecture"],
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 2),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Device: %s", device)

    # --- DataLoaders ---
    train_loader, val_loader = _create_dataloaders(
        train_samples, val_samples, config
    )

    # --- Loss, optimizer, scheduler ---
    train_cfg = config.get("training", {})
    class_weights_cfg = train_cfg.get("class_weights", "balanced")

    if class_weights_cfg == "balanced":
        class_weights = compute_class_weights(train_samples, num_classes=2).to(device)
    elif class_weights_cfg is not None:
        class_weights = torch.tensor(class_weights_cfg, dtype=torch.float32).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential learning rates
    lr = train_cfg.get("learning_rate", 1e-4)
    head_lr_mult = train_cfg.get("head_lr_multiplier", 10)
    wd = train_cfg.get("weight_decay", 1e-4)

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ], weight_decay=wd)

    epochs = train_cfg.get("epochs", 100)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"

    # Cosine annealing after warmup
    if warmup_epochs > 0 and warmup_epochs < epochs:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Checkpointing ---
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(paths_cfg.get("metrics_dir", "outputs/train"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / "metrics.jsonl"
    scaler = GradScaler(enabled=use_amp)

    # --- Training loop ---
    logger.info("=== Starting training ===")
    best_val_loss = float("inf")
    best_epoch = 0
    patience = train_cfg.get("early_stopping_patience", 15)
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += images.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_loss_epoch = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        val_metrics = _validate(model, val_loader, criterion, device, use_amp)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss_epoch, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "lr": round(current_lr, 8),
            "time_s": round(elapsed, 1),
        }
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f | lr=%.2e | time=%.1fs",
            epoch, epochs, train_loss_epoch, train_acc,
            val_metrics["loss"], val_metrics["acc"], current_lr, elapsed,
        )

        # Save metrics
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        # Checkpointing
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
                 "config": config},
                checkpoint_dir / "best.pth",
            )
        else:
            no_improve += 1

        # Save latest
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "config": config},
            checkpoint_dir / "last.pth",
        )

        if no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (best: epoch %d, val_loss=%.4f)",
                epoch, best_epoch, best_val_loss,
            )
            break

    logger.info("=== Training complete, best epoch: %d ===", best_epoch)
    return str(checkpoint_dir / "best.pth")


def _validate(model, dataloader, criterion, device, use_amp):
    """Run validation and return metrics dict."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }
