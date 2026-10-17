from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.brain_tumor.data import classifier_transform
from src.brain_tumor.models import build_classifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ValueError("Validation loader is empty")
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the EfficientNet-B0 brain MRI classification baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/classification"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/classifier.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet EfficientNet-B0 weights; may download on first run")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs/batch-size must be positive and image-size must be at least 32")
    seed_everything(args.seed)
    train_dir, val_dir = args.data_dir / "train", args.data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        parser.error(f"Expected train/ and val/ under {args.data_dir}")
    train_ds = datasets.ImageFolder(train_dir, transform=classifier_transform(args.image_size, train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=classifier_transform(args.image_size, train=False))
    if train_ds.classes != val_ds.classes:
        parser.error(f"Train/val classes differ: {train_ds.classes} vs {val_ds.classes}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        parser.error("Train and validation sets must not be empty")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(len(train_ds.classes), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    best_val_loss, stale = float("inf"), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, total, correct = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        record = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                  "val_loss": val_loss, "val_accuracy": val_acc, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(json.dumps(record))
        if val_loss < best_val_loss:
            best_val_loss, stale = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "classes": train_ds.classes,
                        "image_size": args.image_size, "pretrained_requested": args.pretrained,
                        "seed": args.seed, "best_val_loss": best_val_loss, "best_val_accuracy": val_acc,
                        "architecture": "efficientnet_b0", "history": history}, args.output)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs (no validation-loss improvement).")
                break
    args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {args.output.resolve()}")
    print("Reminder: evaluate the final checkpoint once on the untouched test split.")


if __name__ == "__main__":
    main()
