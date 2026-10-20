import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_brisc import EXPECTED_CLASSES, prepare_dataset


def _image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), color, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=95)


def _mask(path: Path, foreground: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((16, 16), dtype=np.uint8)
    array[3:9, 4:10] = foreground
    Image.fromarray(array, mode="L").save(path, format="PNG")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_dataset(root: Path) -> tuple[Path, Path]:
    # Each train/test class has multiple examples so validation splitting can
    # keep at least one sample in both train and validation.
    shared_test_image = None
    for class_index, class_name in enumerate(EXPECTED_CLASSES):
        class_root = root / "classification_task"
        test_image = class_root / "test" / class_name / f"test_{class_name}.jpg"
        _image(test_image, 80 + class_index * 20)

        for n in range(3):
            train_image = class_root / "train" / class_name / f"train_{class_name}_{n}.jpg"
            _image(train_image, 10 + class_index * 20 + n)

        if class_name == "glioma":
            shared_test_image = test_image
            duplicate_training_image = class_root / "train" / class_name / "exact_duplicate.jpg"
            duplicate_training_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(test_image, duplicate_training_image)

    assert shared_test_image is not None

    for split in ("train", "test"):
        image_dir = root / "segmentation_task" / split / "images"
        mask_dir = root / "segmentation_task" / split / "masks"

        for n in range(3 if split == "train" else 1):
            stem = f"brisc2025_{split}_{n:05d}_gl_ax_t1"
            image_path = image_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.png"
            _image(image_path, 15 + (n * 15) + (0 if split == "train" else 100))
            _mask(mask_path)

        if split == "test":
            test_image = image_dir / "brisc2025_test_overlap_gl_ax_t1.jpg"
            _image(test_image, 240)
            _mask(mask_dir / "brisc2025_test_overlap_gl_ax_t1.png")

    # Add an image to the segmentation training set whose exact bytes match a
    # test image, and provide a paired mask. The training pair should be removed.
    seg_test_image = root / "segmentation_task" / "test" / "images" / "brisc2025_test_overlap_gl_ax_t1.jpg"
    seg_train_image = root / "segmentation_task" / "train" / "images" / "brisc2025_train_overlap_gl_ax_t1.jpg"
    seg_train_mask = root / "segmentation_task" / "train" / "masks" / "brisc2025_train_overlap_gl_ax_t1.png"
    seg_train_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seg_test_image, seg_train_image)
    _mask(seg_train_mask)

    return shared_test_image, seg_test_image


def test_preparation_removes_cross_split_duplicates_and_preserves_test(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    shared_cls_test, shared_seg_test = _build_dataset(source)

    cls_test_hash = _digest(shared_cls_test)
    seg_test_hash = _digest(shared_seg_test)
    report = prepare_dataset(source, output, val_fraction=0.25, seed=7)

    assert report["tasks"]["classification"]["train_test_overlapping_training_files_removed"] == 1
    assert report["tasks"]["segmentation"]["train_test_overlapping_training_files_removed"] == 1
    assert report["tasks"]["classification"]["prepared_train_test_hash_overlap"] == 0
    assert report["tasks"]["segmentation"]["prepared_train_test_hash_overlap"] == 0

    prepared_cls_test = output / "classification" / "test" / "glioma" / shared_cls_test.name
    prepared_seg_test = output / "segmentation" / "images" / "test" / shared_seg_test.name
    assert _digest(prepared_cls_test) == cls_test_hash
    assert _digest(prepared_seg_test) == seg_test_hash

    prepared_cls_trainval = list((output / "classification" / "train").rglob("*.*"))
    prepared_cls_trainval += list((output / "classification" / "val").rglob("*.*"))
    assert cls_test_hash not in {_digest(path) for path in prepared_cls_trainval}

    prepared_seg_trainval = list((output / "segmentation" / "images" / "train").glob("*"))
    prepared_seg_trainval += list((output / "segmentation" / "images" / "val").glob("*"))
    assert seg_test_hash not in {_digest(path) for path in prepared_seg_trainval}

    for split in ("train", "val", "test"):
        images = list((output / "segmentation" / "images" / split).glob("*"))
        masks = list((output / "segmentation" / "masks" / split).glob("*"))
        assert {path.stem for path in images} == {path.stem for path in masks}

    saved_report = json.loads((output / "preparation_report.json").read_text(encoding="utf-8"))
    assert saved_report["patient_level_isolation_confirmed"] is False
    assert (output / "prepared_manifest.csv").is_file()


def test_preparation_refuses_nonempty_output_by_default(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    _build_dataset(source)
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    try:
        prepare_dataset(source, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Nonempty output directory should not be overwritten implicitly")

    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
