from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MASK_THRESHOLD = 128


def classifier_transform(image_size: int = 224, train: bool = False):
    ops = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(7)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def segmentation_transform(image_size: int = 256):
    return transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)


def binarize_mask(mask: Image.Image | np.ndarray, threshold: int = DEFAULT_MASK_THRESHOLD) -> np.ndarray:
    """Convert a grayscale annotation to a binary float32 mask.

    BRISC masks inspected during setup contain near-black values (0-7) and
    near-white foreground values (248-255), including intermediate grayscale
    edge pixels. Thresholding at 128 avoids treating tiny non-zero background
    artifacts as tumor. Keep mask resizing on nearest-neighbor interpolation.
    """
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    if isinstance(mask, Image.Image):
        mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        mask_array = np.asarray(mask, dtype=np.uint8)

    return (mask_array >= threshold).astype(np.float32)


class PairedSegmentationDataset(Dataset):
    """Pairs image/mask files by stem under images/<split> and masks/<split>."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected {self.image_dir} and {self.mask_dir}")
        if not 0 <= mask_threshold <= 255:
            raise ValueError("mask_threshold must be between 0 and 255")

        valid = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid)
        masks = {p.stem: p for p in self.mask_dir.iterdir() if p.is_file() and p.suffix.lower() in valid}
        missing = [p.name for p in images if p.stem not in masks]
        if missing:
            raise ValueError(f"Missing masks for {len(missing)} image(s); examples: {missing[:5]}")
        image_stems = {p.stem for p in images}
        extra_masks = sorted(stem for stem in masks if stem not in image_stems)
        if extra_masks:
            raise ValueError(f"Found {len(extra_masks)} mask(s) without a matching image; examples: {extra_masks[:5]}")
        if not images:
            raise ValueError(f"No images found in {self.image_dir}")
        self.pairs = [(p, masks[p.stem]) for p in images]
        self.image_size = image_size
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = binarize_mask(mask, threshold=self.mask_threshold)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor, image_path.name


def classify_image_tensor(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    return classifier_transform(image_size, train=False)(image.convert("RGB")).unsqueeze(0)


def segmentation_image_tensor(image: Image.Image, image_size: int = 256) -> torch.Tensor:
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
