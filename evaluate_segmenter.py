from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.brain_tumor.data import PairedSegmentationDataset
from src.brain_tumor.losses import dice_iou_from_masks
from src.brain_tumor.models import UNet


def main():
    p = argparse.ArgumentParser(description="Evaluate binary segmentation on held-out masks")
    p.add_argument("--data-dir", type=Path, default=Path("data/segmentation"))
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/segmenter.pt"))
    p.add_argument("--output", type=Path, default=Path("artifacts/evaluation/segmentation.json"))
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ds = PairedSegmentationDataset(args.data_dir, "test", ckpt.get("image_size", 256))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for images, masks, names in loader:
            preds = torch.sigmoid(model(images.to(device))).cpu() >= 0.5
            for pred, mask, name in zip(preds, masks >= 0.5, names):
                dice, iou = dice_iou_from_masks(pred, mask)
                target_pixels = int(mask.sum().item())
                predicted_pixels = int(pred.sum().item())
                rows.append({"file": name, "dice": dice, "iou": iou,
                             "target_tumor_pixels": target_pixels,
                             "predicted_tumor_pixels": predicted_pixels,
                             "target_mask_empty": target_pixels == 0,
                             "false_positive_on_empty_mask": target_pixels == 0 and predicted_pixels > 0})
    if not rows:
        p.error("The test set is empty")
    tumor_present = [r for r in rows if not r["target_mask_empty"]]
    empty_targets = [r for r in rows if r["target_mask_empty"]]
    result = {
        "n_test_images": len(rows),
        "mean_dice_all_images": float(np.mean([r["dice"] for r in rows])),
        "mean_iou_all_images": float(np.mean([r["iou"] for r in rows])),
        "tumor_present_image_count": len(tumor_present),
        "mean_dice_tumor_present_only": float(np.mean([r["dice"] for r in tumor_present])) if tumor_present else None,
        "mean_iou_tumor_present_only": float(np.mean([r["iou"] for r in tumor_present])) if tumor_present else None,
        "empty_ground_truth_mask_count": len(empty_targets),
        "empty_mask_false_positive_rate": (sum(r["false_positive_on_empty_mask"] for r in empty_targets) / len(empty_targets)) if empty_targets else None,
        "per_image": rows,
        "checkpoint": str(args.checkpoint),
        "metric_note": "All-image means include empty-mask cases; tumor-present-only scores are reported separately to reduce masking of boundary errors by easy empty cases. Empty-mask false-positive rate measures predicted foreground on scans with no annotated tumor mask."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
