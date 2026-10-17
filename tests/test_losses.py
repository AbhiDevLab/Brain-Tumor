import torch

from src.brain_tumor.losses import dice_iou_from_masks, soft_dice_loss
from src.brain_tumor.models import UNet


def test_perfect_mask_scores_one():
    pred = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float32)
    target = pred.clone()
    dice, iou = dice_iou_from_masks(pred, target)
    assert dice == 1.0
    assert iou == 1.0


def test_empty_masks_are_defined():
    pred = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    target = torch.zeros_like(pred)
    dice, iou = dice_iou_from_masks(pred, target)
    assert dice == 1.0
    assert iou == 1.0


def test_soft_dice_loss_is_finite():
    logits = torch.zeros((2, 1, 8, 8))
    targets = torch.zeros_like(logits)
    loss = soft_dice_loss(logits, targets)
    assert torch.isfinite(loss)
    assert 0 <= float(loss) <= 1


def test_unet_output_shape_matches_input():
    model = UNet(base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_gradcam_returns_normalized_map():
    from src.brain_tumor.explain import generate_gradcam
    from src.brain_tumor.models import build_classifier

    model = build_classifier(num_classes=4, pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    cam = generate_gradcam(model, x, class_index=1, output_size=(32, 32))
    assert cam.shape == (32, 32)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0
