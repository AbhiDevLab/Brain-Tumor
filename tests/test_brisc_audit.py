import csv
from pathlib import Path

import numpy as np
from PIL import Image

from audit_brisc import audit_classification, audit_manifest, audit_segmentation


def _write_image(path: Path, pixels: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(pixels, pixels, pixels)).save(path)


def _write_mask(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[4:10, 5:11] = value
    Image.fromarray(arr, mode="L").save(path)


def _classification_tree(root: Path) -> None:
    for split, color in (("train", 40), ("test", 100)):
        for class_name in ("glioma", "meningioma", "pituitary", "no_tumor"):
            _write_image(
                root / "classification_task" / split / class_name / f"{split}_{class_name}.jpg",
                pixels=color,
            )


def _segmentation_tree(root: Path, *, include_test_mask: bool = True) -> None:
    for split, color in (("train", 40), ("test", 100)):
        stem = f"{split}_sample"
        _write_image(
            root / "segmentation_task" / split / "images" / f"{stem}.jpg",
            pixels=color,
        )
        if split != "test" or include_test_mask:
            _write_mask(
                root / "segmentation_task" / split / "masks" / f"{stem}.png"
            )


def test_classification_audit_counts_classes(tmp_path):
    _classification_tree(tmp_path)

    report = audit_classification(tmp_path)

    assert report["class_sets_match"] is True
    assert set(report["class_counts"]["train"]) == {
        "glioma", "meningioma", "pituitary", "no_tumor"
    }
    assert all(value == 1 for value in report["class_counts"]["test"].values())


def test_segmentation_audit_passes_for_matching_pairs(tmp_path):
    _segmentation_tree(tmp_path)

    report = audit_segmentation(tmp_path, sample_masks=2)

    assert report["pairing_check_pass"] is True
    assert report["splits"]["train"]["paired_by_filename_stem"] == 1
    assert report["splits"]["test"]["paired_by_filename_stem"] == 1


def test_segmentation_audit_flags_missing_mask(tmp_path):
    _segmentation_tree(tmp_path, include_test_mask=False)

    report = audit_segmentation(tmp_path, sample_masks=2)

    assert report["pairing_check_pass"] is False
    assert report["splits"]["test"]["image_missing_mask_count"] == 1


def test_manifest_audit_reports_patient_overlap(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "relative_path,split,patient_id\n"
        "train/a.jpg,train,p001\n"
        "test/b.jpg,test,p001\n"
        "train/c.jpg,train,p002\n",
        encoding="utf-8",
    )

    report = audit_manifest(tmp_path)

    assert report["available"] is True
    assert report["patient_or_subject_ids_crossing_splits"] == 1
    assert report["patient_or_subject_overlap_examples"][0]["id"] == "p001"


def test_manifest_hash_validation_and_cross_split_overlap(tmp_path):
    train_image = tmp_path / "classification_task" / "train" / "glioma" / "train.jpg"
    test_image = tmp_path / "classification_task" / "test" / "glioma" / "test.jpg"
    _write_image(train_image, pixels=70)
    test_image.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(train_image, test_image)

    import hashlib
    image_hash = hashlib.sha256(train_image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "relative_path,task,split,is_mask,sha256\n"
        + rf"classification_task\train\glioma\train.jpg,classification,train,False,{image_hash}" + "\n"
        + rf"classification_task\test\glioma\test.jpg,classification,test,False,{image_hash}" + "\n",
        encoding="utf-8",
    )

    report = audit_manifest(tmp_path)

    assert report["file_path_validation"]["files_found_and_hashed"] == 2
    assert report["file_path_validation"]["missing_or_unresolved_files"] == 0
    assert report["sha256_validation"]["files_with_hash_mismatches"] == 0
    assert report["manifest_image_hashes_crossing_train_test"]["duplicate_hash_groups"] == 1
