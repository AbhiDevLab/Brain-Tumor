from __future__ import annotations

import torch


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def dice_iou_from_masks(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> tuple[float, float]:
    pred = pred.bool()
    target = target.bool()
    intersection = torch.logical_and(pred, target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    union = torch.logical_or(pred, target).sum().item()
    dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return float(dice), float(iou)
