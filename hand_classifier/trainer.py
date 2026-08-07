"""Training loop for dual-head hand classifier with multi-task loss."""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
)
from tqdm import tqdm

from hand_classifier.dataset import (
    HandROIDataset, get_transforms, compute_class_weights,
)
from hand_classifier.parser import collect_all_samples
from hand_classifier.dataset import split_dataset, save_split_info

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
    """Run the full training pipeline for dual-head model.

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
        all_samples = collect_all_samples(train_sources)
        train_samples, val_samples, test_samples = split_dataset(
            all_samples, config
        )

    # Log per-task counts
    _log_sample_counts(train_samples, "Train")
    _log_sample_counts(val_samples, "Val")

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
        num_handedness=model_cfg.get("num_handedness", 2),
        num_presence=model_cfg.get("num_presence", 2),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Device: %s", device)

    # --- DataLoaders ---
    train_loader, val_loader = _create_dataloaders(
        train_samples, val_samples, config
    )

    # --- Losses ---
    train_cfg = config.get("training", {})
    hp_cfg = config.get("hand_presence", {})
    class_weights_cfg = train_cfg.get("class_weights", "balanced")

    if class_weights_cfg == "balanced":
        h_weights = compute_class_weights(
            train_samples, num_classes=2, task="handedness"
        ).to(device)
        p_weights = compute_class_weights(
            train_samples, num_classes=2, task="hand_presence"
        ).to(device)
    elif class_weights_cfg is not None:
        h_weights = torch.tensor(class_weights_cfg, dtype=torch.float32).to(device)
        p_weights = h_weights
    else:
        h_weights = p_weights = None

    criterion_h = nn.CrossEntropyLoss(weight=h_weights, ignore_index=-1)
    criterion_p = nn.CrossEntropyLoss(weight=p_weights)

    h_loss_weight = train_cfg.get("handedness_loss_weight", 1.0)
    p_loss_weight = hp_cfg.get("loss_weight", 1.0)

    # --- Optimizer ---
    lr = train_cfg.get("learning_rate", 1e-4)
    head_lr_mult = train_cfg.get("head_lr_multiplier", 10)
    wd = train_cfg.get("weight_decay", 1e-4)

    # Separate head params (both heads get higher LR)
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "head" in name or "classifier" in name:
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
        train_loss_total = 0.0
        train_loss_h = 0.0
        train_loss_p = 0.0
        train_correct_h = 0
        train_total_h = 0
        train_correct_p = 0
        train_total = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for images, h_labels, p_labels in pbar:
            images = images.to(device, non_blocking=True)
            h_labels = h_labels.to(device, non_blocking=True)
            p_labels = p_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(images)
                # Guard: CrossEntropyLoss + weight + all-ignored → NaN
                valid_h = h_labels >= 0
                if valid_h.any():
                    loss_h = criterion_h(outputs["handedness"], h_labels)
                else:
                    loss_h = torch.tensor(0.0, device=device)
                loss_p = criterion_p(outputs["hand_presence"], p_labels)
                loss = h_loss_weight * loss_h + p_loss_weight * loss_p

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item() * images.size(0)
            train_loss_h += loss_h.item() * images.size(0)
            train_loss_p += loss_p.item() * images.size(0)

            # Handedness accuracy (only on valid labels)
            if valid_h.any():
                _, pred_h = outputs["handedness"].max(1)
                train_correct_h += pred_h[valid_h].eq(
                    h_labels[valid_h]
                ).sum().item()
                train_total_h += valid_h.sum().item()

            # Presence accuracy (on all)
            _, pred_p = outputs["hand_presence"].max(1)
            train_correct_p += pred_p.eq(p_labels).sum().item()
            train_total += images.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "h": f"{loss_h.item():.4f}",
                "p": f"{loss_p.item():.4f}",
            })

        scheduler.step()

        n = max(train_total, 1)
        nh = max(train_total_h, 1)
        train_metrics = {
            "loss": train_loss_total / n,
            "loss_h": train_loss_h / n,
            "loss_p": train_loss_p / n,
            "acc_h": round(train_correct_h / nh, 6) if train_total_h > 0 else None,
            "acc_p": round(train_correct_p / n, 6),
        }

        # Validation
        val_metrics = _validate(model, val_loader, device,
                                criterion_h, criterion_p,
                                h_loss_weight, p_loss_weight)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        log_msg = (
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(h={train_metrics['loss_h']:.4f} p={train_metrics['loss_p']:.4f}) | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc_h={val_metrics.get('acc_h')} val_acc_p={val_metrics['acc_p']:.4f} | "
            f"lr={current_lr:.2e} | time={elapsed:.1f}s"
        )
        logger.info(log_msg)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_loss_h": round(train_metrics["loss_h"], 6),
            "train_loss_p": round(train_metrics["loss_p"], 6),
            "train_acc_h": train_metrics["acc_h"],
            "train_acc_p": train_metrics["acc_p"],
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc_h": val_metrics.get("acc_h"),
            "val_acc_p": round(val_metrics["acc_p"], 6),
            "lr": round(current_lr, 8),
            "time_s": round(elapsed, 1),
        }
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": val_metrics["loss"],
                 "val_acc_h": val_metrics.get("acc_h"),
                 "val_acc_p": val_metrics["acc_p"],
                 "config": config},
                checkpoint_dir / "best.pth",
            )
        else:
            no_improve += 1

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(), "config": config},
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


def _validate(model, dataloader, device,
              criterion_h, criterion_p, h_weight, p_weight):
    """Run validation for dual-head model."""
    model.eval()
    total_loss = total_h = total_p = 0.0
    correct_h = total_h_valid = 0
    correct_p = total = 0

    with torch.no_grad():
        for images, h_labels, p_labels in dataloader:
            images = images.to(device, non_blocking=True)
            h_labels = h_labels.to(device, non_blocking=True)
            p_labels = p_labels.to(device, non_blocking=True)

            outputs = model(images)  # No AMP in validation
            # Guard: CrossEntropyLoss + weight + all-ignored → NaN
            valid_h = h_labels >= 0
            if valid_h.any():
                loss_h = criterion_h(outputs["handedness"], h_labels)
            else:
                loss_h = torch.tensor(0.0, device=device)
            loss_p = criterion_p(outputs["hand_presence"], p_labels)
            loss = h_weight * loss_h + p_weight * loss_p

            n = images.size(0)
            total_loss += loss.item() * n
            total_h += loss_h.item() * n
            total_p += loss_p.item() * n

            if valid_h.any():
                _, pred_h = outputs["handedness"].max(1)
                correct_h += pred_h[valid_h].eq(
                    h_labels[valid_h]
                ).sum().item()
                total_h_valid += valid_h.sum().item()

            _, pred_p = outputs["hand_presence"].max(1)
            correct_p += pred_p.eq(p_labels).sum().item()
            total += n

    n = max(total, 1)
    result = {
        "loss": total_loss / n,
        "loss_h": total_h / n,
        "loss_p": total_p / n,
        "acc_p": correct_p / n,
    }
    if total_h_valid > 0:
        result["acc_h"] = round(correct_h / total_h_valid, 6)
    return result


def _log_sample_counts(samples, name):
    """Log per-task sample counts."""
    n_hand = sum(1 for s in samples if s["handedness_label"] >= 0)
    n_pres = sum(1 for s in samples if s["presence_label"] == 1)
    logger.info(
        "%s: total=%d, handedness_valid=%d, has_hand=%d, no_hand=%d",
        name, len(samples), n_hand, n_pres,
        len(samples) - n_pres,
    )
