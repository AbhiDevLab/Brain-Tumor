from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.brain_tumor.data import PairedSegmentationDataset
from src.brain_tumor.losses import soft_dice_loss, dice_iou_from_masks
from src.brain_tumor.models import UNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _atomic_torch_save(payload: dict, destination: Path) -> None:
    """Write a checkpoint completely before replacing the previous copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _atomic_write_text(content: str, destination: Path) -> None:
    """Persist small metadata files without leaving a half-written file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def evaluate(model, loader, device, loss_mode: str):
    model.eval()
    losses, dices, ious = [], [], []
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = bce(logits, masks)
            if loss_mode == "bce_dice":
                loss = loss + soft_dice_loss(logits, masks)
            losses.append(loss.item())
            preds = torch.sigmoid(logits) >= 0.5
            for p, t in zip(preds, masks >= 0.5):
                d, i = dice_iou_from_masks(p, t)
                dices.append(d)
                ious.append(i)
    if not losses:
        raise ValueError("Validation loader is empty")
    return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious))


def main():
    parser = argparse.ArgumentParser(description="Train a binary-mask U-Net segmentation baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/segmentation"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/segmenter.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--loss-mode", choices=("bce", "bce_dice"), default="bce",
                        help="bce establishes the baseline; bce_dice is the first controlled improvement experiment")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from a latest recovery checkpoint")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size) < 1:
        parser.error("epochs, batch-size and image-size must be positive")
    seed_everything(args.seed)
    train_ds = PairedSegmentationDataset(args.data_dir, "train", args.image_size)
    val_ds = PairedSegmentationDataset(args.data_dir, "val", args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_loss, stale, history = float("inf"), 0, []
    best_val_dice = None
    best_val_iou = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest_output = args.output.with_name(
        f"{args.output.stem}.latest{args.output.suffix}"
    )
    history_output = args.output.with_suffix(".history.json")
    start_epoch = 1

    if args.resume_from is not None:
        if not args.resume_from.is_file():
            parser.error(f"Recovery checkpoint not found: {args.resume_from}")
        recovery = torch.load(
            args.resume_from, map_location=device, weights_only=False
        )
        if recovery.get("seed") != args.seed:
            parser.error("Resume seed must match the checkpoint seed.")
        if recovery.get("image_size") != args.image_size:
            parser.error("Resume image size must match the checkpoint.")
        if recovery.get("loss_mode") != args.loss_mode:
            parser.error("Resume loss mode must match the checkpoint.")

        model.load_state_dict(recovery["model_state_dict"])
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        start_epoch = int(recovery["epoch"]) + 1
        best_val_loss = float(recovery["best_val_loss"])
        best_val_dice = recovery.get("best_val_dice_all_images")
        best_val_iou = recovery.get("best_val_iou_all_images")
        stale = int(recovery["stale"])
        history = list(recovery.get("history", []))

        torch.set_rng_state(recovery["torch_rng_state"].cpu())
        np.random.set_state(recovery["numpy_rng_state"])
        random.setstate(recovery["python_rng_state"])
        cuda_rng = recovery.get("cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_rng is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])

        if start_epoch > args.epochs:
            parser.error("Checkpoint already reached the requested epoch count.")
        print(f"Resuming at epoch {start_epoch} from {args.resume_from}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum, batches = 0.0, 0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks)
            if args.loss_mode == "bce_dice":
                loss = loss + soft_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        train_loss = loss_sum / max(1, batches)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device, args.loss_mode)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "val_dice_all_images": val_dice, "val_iou_all_images": val_iou,
                  "loss_mode": args.loss_mode}
        history.append(record)
        print(json.dumps(record))
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, stale = val_loss, 0
            best_val_dice, best_val_iou = val_dice, val_iou
            _atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "image_size": args.image_size,
                "architecture": "unet_binary",
                "loss_mode": args.loss_mode,
                "seed": args.seed,
                "best_val_loss": best_val_loss,
                "best_val_dice_all_images": best_val_dice,
                "best_val_iou_all_images": best_val_iou,
                "history": history,
            }, args.output)
        else:
            stale += 1

        # Persist latest weights, optimizer/RNG state and history every epoch.
        _atomic_torch_save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "image_size": args.image_size,
            "architecture": "unet_binary",
            "loss_mode": args.loss_mode,
            "seed": args.seed,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_dice_all_images": best_val_dice,
            "best_val_iou_all_images": best_val_iou,
            "stale": stale,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        }, latest_output)
        _atomic_write_text(json.dumps(history, indent=2), history_output)

        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break
    _atomic_write_text(json.dumps(history, indent=2), history_output)
    print(f"Best checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
