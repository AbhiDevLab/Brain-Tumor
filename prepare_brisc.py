from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")


@dataclass(frozen=True)
class Sample:
    image: Path
    label: str | None = None
    mask: Path | None = None
    digest: str = ""
    mask_digest: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _samples(paths: list[Path], label: str | None = None) -> list[Sample]:
    return [Sample(image=path, label=label, digest=sha256(path)) for path in paths]


def _stratum(path: Path) -> str:
    # BRISC filenames look like brisc2025_train_00001_gl_ax_t1.jpg.
    parts = path.stem.split("_")
    return parts[3] if len(parts) > 3 else "unknown"


def _split_train_val(
    samples: list[Sample],
    *,
    val_fraction: float,
    seed: int,
    stratify,
) -> tuple[list[Sample], list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[stratify(sample)].append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    rng = random.Random(seed)

    for group_key in sorted(groups):
        members = sorted(groups[group_key], key=lambda sample: sample.image.name)
        rng.shuffle(members)
        if len(members) < 2:
            # Retain a singleton in train: validation must not consume the
            # only training example in a stratum.
            train.extend(members)
            continue
        val_count = min(len(members) - 1, max(1, math.ceil(len(members) * val_fraction)))
        val.extend(members[:val_count])
        train.extend(members[val_count:])

    return sorted(train, key=lambda sample: str(sample.image)), sorted(val, key=lambda sample: str(sample.image))


def _copy_sample(
    sample: Sample,
    *,
    task: str,
    split: str,
    output_root: Path,
    manifest_rows: list[dict[str, str]],
) -> None:
    if task == "classification":
        if not sample.label:
            raise ValueError(f"Classification sample has no label: {sample.image}")
        image_rel = Path("classification") / split / sample.label / sample.image.name
    elif task == "segmentation":
        image_rel = Path("segmentation") / "images" / split / sample.image.name
    else:
        raise ValueError(f"Unknown task: {task}")

    image_dest = output_root / image_rel
    image_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image, image_dest)

    mask_rel = ""
    if sample.mask is not None:
        mask_rel_path = Path("segmentation") / "masks" / split / sample.mask.name
        mask_dest = output_root / mask_rel_path
        mask_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.mask, mask_dest)
        mask_rel = mask_rel_path.as_posix()

    manifest_rows.append({
        "task": task,
        "split": split,
        "label": sample.label or "",
        "image_path": image_rel.as_posix(),
        "mask_path": mask_rel,
        "source_image": str(sample.image),
        "source_mask": str(sample.mask) if sample.mask else "",
        "image_sha256": sample.digest,
        "mask_sha256": sample.mask_digest or "",
    })


def _classification_samples(root: Path, split: str) -> list[Sample]:
    split_root = root / "classification_task" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing classification split: {split_root}")

    found_classes = {path.name for path in split_root.iterdir() if path.is_dir()}
    missing = set(EXPECTED_CLASSES) - found_classes
    if missing:
        raise ValueError(f"Missing classification class folders under {split_root}: {sorted(missing)}")

    samples: list[Sample] = []
    for label in EXPECTED_CLASSES:
        files = image_files(split_root / label)
        if not files:
            raise ValueError(f"No images found for class {label} in {split_root}")
        samples.extend(_samples(files, label))
    return samples


def _clean_training_samples(
    train_samples: list[Sample],
    test_samples: list[Sample],
    *,
    task: str,
) -> tuple[list[Sample], dict]:
    test_hashes = {sample.digest for sample in test_samples}
    train_by_hash: dict[str, list[Sample]] = defaultdict(list)
    for sample in train_samples:
        train_by_hash[sample.digest].append(sample)

    kept: list[Sample] = []
    overlap_hash_groups = []
    overlap_files_removed = 0
    train_duplicate_groups = []
    train_duplicates_removed = 0
    label_conflict_groups = []

    for digest in sorted(train_by_hash):
        group = sorted(train_by_hash[digest], key=lambda sample: str(sample.image))
        if digest in test_hashes:
            overlap_hash_groups.append({
                "sha256": digest,
                "training_files_removed": [str(sample.image) for sample in group],
                "labels": sorted({sample.label or "" for sample in group}),
            })
            overlap_files_removed += len(group)
            continue

        if len(group) > 1:
            if task == "classification":
                labels = {sample.label for sample in group}
                if len(labels) > 1:
                    label_conflict_groups.append({
                        "sha256": digest,
                        "files_excluded": [str(sample.image) for sample in group],
                        "labels": sorted(label or "" for label in labels),
                        "reason": "Byte-identical training images have conflicting class labels.",
                    })
                    continue

            if task == "segmentation":
                # Same exact image but incompatible masks is annotation ambiguity;
                # exclude all copies rather than silently choose one target.
                mask_hashes = {sample.mask_digest for sample in group}
                if len(mask_hashes) > 1:
                    label_conflict_groups.append({
                        "sha256": digest,
                        "files_excluded": [str(sample.image) for sample in group],
                        "mask_sha256": sorted(mask_hash or "" for mask_hash in mask_hashes),
                        "reason": "Byte-identical training images have different masks.",
                    })
                    continue

            train_duplicate_groups.append({
                "sha256": digest,
                "kept_file": str(group[0].image),
                "duplicate_files_removed": [str(sample.image) for sample in group[1:]],
            })
            train_duplicates_removed += len(group) - 1
            kept.append(group[0])
        else:
            kept.append(group[0])

    return sorted(kept, key=lambda sample: str(sample.image)), {
        "task": task,
        "test_unique_image_hashes": len(test_hashes),
        "train_test_duplicate_hash_groups": len(overlap_hash_groups),
        "train_test_overlapping_training_files_removed": overlap_files_removed,
        "train_test_overlap_examples": overlap_hash_groups[:30],
        "train_only_duplicate_hash_groups": len(train_duplicate_groups),
        "train_only_duplicate_files_removed": train_duplicates_removed,
        "train_only_duplicate_examples": train_duplicate_groups[:30],
        "conflicting_training_groups_excluded": len(label_conflict_groups),
        "conflicting_training_examples": label_conflict_groups[:30],
    }


def _segmentation_samples(root: Path, split: str) -> list[Sample]:
    task_root = root / "segmentation_task" / split
    image_dir = task_root / "images"
    mask_dir = task_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected image and mask directories under {task_root}")

    images = image_files(image_dir)
    masks = {path.stem: path for path in image_files(mask_dir)}
    image_stems = {path.stem for path in images}
    mask_stems = set(masks)
    missing_masks = sorted(image_stems - mask_stems)
    extra_masks = sorted(mask_stems - image_stems)

    if missing_masks or extra_masks:
        raise ValueError(
            f"Unpaired {split} segmentation data: "
            f"{len(missing_masks)} images missing masks; "
            f"{len(extra_masks)} masks missing images. "
            f"Examples missing masks: {missing_masks[:5]}; "
            f"orphan masks: {extra_masks[:5]}"
        )
    if not images:
        raise ValueError(f"No segmentation images found in {image_dir}")

    return [
        Sample(
            image=image,
            mask=masks[image.stem],
            digest=sha256(image),
            mask_digest=sha256(masks[image.stem]),
        )
        for image in images
    ]


def _copy_task(
    samples: list[Sample],
    *,
    task: str,
    split: str,
    output_root: Path,
    manifest_rows: list[dict[str, str]],
) -> None:
    for sample in samples:
        _copy_sample(
            sample,
            task=task,
            split=split,
            output_root=output_root,
            manifest_rows=manifest_rows,
        )


def prepare_dataset(
    root: Path,
    output: Path,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    root = root.resolve()
    output = output.resolve()
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be greater than 0 and less than 0.5")
    if not (root / "classification_task").is_dir() or not (root / "segmentation_task").is_dir():
        raise FileNotFoundError(
            f"Expected classification_task and segmentation_task under dataset root: {root}"
        )
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output directory must be separate from and not nested inside the source dataset.")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output}. Choose another path or pass --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    report: dict = {
        "source_root": str(root),
        "prepared_root": str(output),
        "validation_fraction_target": val_fraction,
        "seed": seed,
        "test_set_policy": "Copy supplied official test files unchanged; only remove duplicate overlap from the training side.",
        "patient_level_isolation_confirmed": False,
        "patient_level_warning": (
            "The inspected dataset manifest does not contain patient/subject/study IDs. "
            "This preparation removes exact-byte duplicate images across train/test but cannot "
            "establish patient-wise independence or detect near-duplicates."
        ),
        "mask_policy": (
            "Masks are copied unchanged. During training, binarize grayscale masks with threshold >=128 "
            "and resize masks using nearest-neighbor interpolation."
        ),
        "tasks": {},
    }

    # Classification: retain all supplied test images; remove any exact hash
    # overlap from training; deduplicate remaining training duplicates; then
    # split eligible training files into train/validation within each class.
    cls_train_original = _classification_samples(root, "train")
    cls_test = _classification_samples(root, "test")
    cls_train, cls_clean_report = _clean_training_samples(
        cls_train_original, cls_test, task="classification"
    )
    cls_train_out: list[Sample] = []
    cls_val_out: list[Sample] = []
    class_counts = {}
    for label in EXPECTED_CLASSES:
        members = [sample for sample in cls_train if sample.label == label]
        train_part, val_part = _split_train_val(
            members,
            val_fraction=val_fraction,
            seed=seed + EXPECTED_CLASSES.index(label),
            stratify=lambda sample: sample.label or "",
        )
        cls_train_out.extend(train_part)
        cls_val_out.extend(val_part)
        class_counts[label] = {
            "eligible_after_duplicate_filter": len(members),
            "train": len(train_part),
            "val": len(val_part),
            "test_unchanged": sum(1 for sample in cls_test if sample.label == label),
        }

    _copy_task(cls_train_out, task="classification", split="train", output_root=output, manifest_rows=manifest_rows)
    _copy_task(cls_val_out, task="classification", split="val", output_root=output, manifest_rows=manifest_rows)
    _copy_task(cls_test, task="classification", split="test", output_root=output, manifest_rows=manifest_rows)

    cls_test_hashes = {sample.digest for sample in cls_test}
    cls_train_hashes = {sample.digest for sample in cls_train_out}
    cls_val_hashes = {sample.digest for sample in cls_val_out}
    if cls_train_hashes & cls_test_hashes or cls_val_hashes & cls_test_hashes:
        raise RuntimeError("Classification duplicate filtering failed: prepared train/val still overlap test hashes.")

    report["tasks"]["classification"] = {
        **cls_clean_report,
        "original_train_images": len(cls_train_original),
        "official_test_images_copied_unchanged": len(cls_test),
        "prepared_train_images": len(cls_train_out),
        "prepared_validation_images": len(cls_val_out),
        "counts_by_class": class_counts,
        "prepared_train_test_hash_overlap": len(cls_train_hashes & cls_test_hashes),
        "prepared_validation_test_hash_overlap": len(cls_val_hashes & cls_test_hashes),
    }

    # Segmentation: use image hashes for duplicate checks and carry the paired
    # mask with each image. A conflicting target for identical images is
    # excluded as ambiguous. The official test image/mask files stay intact.
    seg_train_original = _segmentation_samples(root, "train")
    seg_test = _segmentation_samples(root, "test")
    seg_train, seg_clean_report = _clean_training_samples(
        seg_train_original, seg_test, task="segmentation"
    )
    seg_train_out, seg_val_out = _split_train_val(
        seg_train,
        val_fraction=val_fraction,
        seed=seed,
        stratify=lambda sample: _stratum(sample.image),
    )

    _copy_task(seg_train_out, task="segmentation", split="train", output_root=output, manifest_rows=manifest_rows)
    _copy_task(seg_val_out, task="segmentation", split="val", output_root=output, manifest_rows=manifest_rows)
    _copy_task(seg_test, task="segmentation", split="test", output_root=output, manifest_rows=manifest_rows)

    seg_test_hashes = {sample.digest for sample in seg_test}
    seg_train_hashes = {sample.digest for sample in seg_train_out}
    seg_val_hashes = {sample.digest for sample in seg_val_out}
    if seg_train_hashes & seg_test_hashes or seg_val_hashes & seg_test_hashes:
        raise RuntimeError("Segmentation duplicate filtering failed: prepared train/val still overlap test hashes.")

    report["tasks"]["segmentation"] = {
        **seg_clean_report,
        "original_train_pairs": len(seg_train_original),
        "official_test_pairs_copied_unchanged": len(seg_test),
        "prepared_train_pairs": len(seg_train_out),
        "prepared_validation_pairs": len(seg_val_out),
        "prepared_train_test_hash_overlap": len(seg_train_hashes & seg_test_hashes),
        "prepared_validation_test_hash_overlap": len(seg_val_hashes & seg_test_hashes),
        "validation_stratification": "Filename tumor code when parseable; otherwise unknown. This is not patient-level splitting.",
    }

    manifest_path = output / "prepared_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "task", "split", "label", "image_path", "mask_path",
            "source_image", "source_mask", "image_sha256", "mask_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    report["manifest_path"] = str(manifest_path)
    report["prepared_manifest_rows"] = len(manifest_rows)
    report_path = output / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a leakage-aware BRISC copy for the repository's baseline training scripts."
    )
    parser.add_argument("--root", type=Path, required=True, help="Source folder containing classification_task and segmentation_task.")
    parser.add_argument("--output", type=Path, required=True, help="Separate output folder for prepared train/val/test data.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    args = parser.parse_args()

    report = prepare_dataset(
        args.root,
        args.output,
        val_fraction=args.val_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
