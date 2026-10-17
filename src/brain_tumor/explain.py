from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def generate_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor, class_index: int,
                     output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Return a normalized Grad-CAM map for the EfficientNet-B0 feature extractor.

    The map is a model-attribution visualization, not a tumor mask or proof of
    clinical reasoning. Input should already use the classifier's preprocessing.
    """
    features = getattr(model, "features", None)
    if features is None or len(features) == 0:
        raise ValueError("Grad-CAM currently supports classifiers exposing a non-empty features module")
    target_layer = features[-1]
    activations: list[torch.Tensor] = []
    handle = target_layer[-1].register_forward_hook(lambda _m, _i, out: activations.append(out))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("Grad-CAM currently expects a single input image")
            if not (0 <= class_index < logits.shape[1]):
                raise ValueError("class_index is outside the model output range")
            gradients = torch.autograd.grad(logits[0, class_index], activations[0], retain_graph=False)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        if output_size is not None:
            cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()


def overlay_gradcam(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Return a lightweight red heat overlay aligned to the provided image."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if cam.shape != base.shape[:2]:
        heat = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(
            (base.shape[1], base.shape[0]), Image.Resampling.BILINEAR)
        cam = np.asarray(heat, dtype=np.float32) / 255.0
    strength = np.clip(cam[..., None], 0.0, 1.0)
    red = np.zeros_like(base)
    red[..., 0] = 255
    merged = np.clip(base * (1.0 - alpha * strength) + red * alpha * strength, 0, 255).astype(np.uint8)
    return Image.fromarray(merged)
