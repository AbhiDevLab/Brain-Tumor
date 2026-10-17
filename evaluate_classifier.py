from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from src.brain_tumor.data import classifier_transform
from src.brain_tumor.models import build_classifier


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained classifier on the untouched test split")
    p.add_argument("--data-dir", type=Path, default=Path("data/classification"))
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/classifier.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/classification"))
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f"Checkpoint not found: {args.checkpoint}")
    test_dir = args.data_dir / "test"
    if not test_dir.is_dir():
        p.error(f"Test split not found: {test_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    ds = datasets.ImageFolder(test_dir, transform=classifier_transform(checkpoint.get("image_size", 224), train=False))
    if ds.classes != classes:
        p.error(f"Test classes {ds.classes} do not match checkpoint classes {classes}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_classifier(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes,
                                   output_dict=True, zero_division=0)
    summary = {"accuracy": float(accuracy_score(y_true, y_pred)),
               "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
               "n_test": len(y_true), "classes": classes, "classification_report": report,
               "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
               "checkpoint": str(args.checkpoint)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["confusion_matrix"], index=classes, columns=classes).to_csv(args.output_dir / "confusion_matrix.csv")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
