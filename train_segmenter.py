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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
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
        if val_loss < best_val_loss:
            best_val_loss, stale = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "image_size": args.image_size,
                        "architecture": "unet_binary", "loss_mode": args.loss_mode, "seed": args.seed,
                        "best_val_loss": val_loss, "best_val_dice_all_images": val_dice,
                        "best_val_iou_all_images": val_iou, "history": history}, args.output)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
