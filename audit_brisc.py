from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_CLASSES = {"glioma", "meningioma", "pituitary", "no_tumor"}
MASK_THRESHOLD = 128


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cross_split_exact_duplicates(paths_by_split: dict[str, list[Path]]) -> dict:
    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split, paths in paths_by_split.items():
        for path in paths:
            hashes[sha256(path)].append({"split": split, "path": str(path)})

    groups = []
    for entries in hashes.values():
        split_names = sorted({entry["split"] for entry in entries})
        if len(split_names) > 1:
            groups.append({"splits": split_names, "files": entries})

    return {
        "exact_duplicate_groups": len(groups),
        "examples": groups[:20],
        "note": "Byte-identical files only; this does not detect near-duplicates or prove patient-level isolation.",
    }


def audit_classification(root: Path) -> dict:
    task_root = root / "classification_task"
    result = {"root": str(task_root), "class_counts": {}, "class_sets_match": True}
    classes_by_split = {}

    for split in ("train", "test"):
        split_root = task_root / split
        if not split_root.is_dir():
            result.setdefault("errors", []).append(f"Missing directory: {split_root}")
            continue

        counts = {}
        all_images = []
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            files = image_files(class_dir)
            counts[class_dir.name] = len(files)
            all_images.extend(files)

        result["class_counts"][split] = counts
        classes_by_split[split] = set(counts)
        result.setdefault("_files", {})[split] = all_images

    if len(classes_by_split) == 2:
        result["class_sets_match"] = classes_by_split["train"] == classes_by_split["test"]
        result["unexpected_classes"] = sorted(
            (classes_by_split["train"] | classes_by_split["test"]) - EXPECTED_CLASSES
        )
        result["missing_expected_classes"] = sorted(
            EXPECTED_CLASSES - (classes_by_split["train"] & classes_by_split["test"])
        )

    if "train" in result.get("_files", {}) and "test" in result.get("_files", {}):
        result["cross_split_exact_duplicates"] = cross_split_exact_duplicates({
            "train": result["_files"]["train"],
            "test": result["_files"]["test"],
        })
    result.pop("_files", None)
    return result


def audit_segmentation(root: Path, sample_masks: int = 60) -> dict:
    task_root = root / "segmentation_task"
    result = {"root": str(task_root), "splits": {}, "mask_pixel_audit": {}}
    image_files_by_split: dict[str, list[Path]] = {}
    pair_errors = False

    for split in ("train", "test"):
        image_dir = task_root / split / "images"
        mask_dir = task_root / split / "masks"

        if not image_dir.is_dir():
            result["splits"][split] = {
                "error": f"Missing image directory: {image_dir}",
                "images": 0,
                "masks": len(image_files(mask_dir)),
                "paired_by_filename_stem": 0,
                "images_missing_masks": [],
                "image_missing_mask_count": 0,
                "masks_without_images": [],
                "orphan_mask_count": len(image_files(mask_dir)),
            }
            pair_errors = True
            continue

        # A missing masks directory is audited as an empty mask set. This
        # preserves actionable counts instead of exiting before reporting
        # that every image in the split is missing its annotation.
        images = image_files(image_dir)
        masks = image_files(mask_dir)
        mask_by_stem = {path.stem: path for path in masks}
        image_stems = {path.stem for path in images}
        mask_stems = set(mask_by_stem)
        missing_masks = sorted(image_stems - mask_stems)
        orphan_masks = sorted(mask_stems - image_stems)

        result["splits"][split] = {
            "images": len(images),
            "masks": len(masks),
            "paired_by_filename_stem": len(image_stems & mask_stems),
            "images_missing_masks": missing_masks[:30],
            "image_missing_mask_count": len(missing_masks),
            "masks_without_images": orphan_masks[:30],
            "orphan_mask_count": len(orphan_masks),
        }
        image_files_by_split[split] = images
        pair_errors |= bool(missing_masks or orphan_masks or not images)

        # Sample masks across the sorted file list to check encoding without
        # loading every full-resolution pixel into memory at once.
        if masks:
            step = max(1, len(masks) // max(1, sample_masks))
            selected = masks[::step][:sample_masks]
            bins = {
                "pixels_0_to_7": 0,
                "pixels_8_to_127": 0,
                "pixels_128_to_247": 0,
                "pixels_248_to_255": 0,
            }
            total_pixels = 0
            occupied_fractions = []
            sampled_values = set()
            size_mismatches = []

            for mask_path in selected:
                with Image.open(mask_path) as opened:
                    mask = np.asarray(opened.convert("L"), dtype=np.uint8)
                    mask_size = opened.size
                total_pixels += int(mask.size)
                bins["pixels_0_to_7"] += int(np.count_nonzero(mask <= 7))
                bins["pixels_8_to_127"] += int(np.count_nonzero((mask >= 8) & (mask <= 127)))
                bins["pixels_128_to_247"] += int(np.count_nonzero((mask >= 128) & (mask <= 247)))
                bins["pixels_248_to_255"] += int(np.count_nonzero(mask >= 248))
                occupied_fractions.append(float(np.mean(mask >= MASK_THRESHOLD)))
                sampled_values.update(int(v) for v in np.unique(mask)[:32])

                image_path = image_dir / f"{mask_path.stem}.jpg"
                if not image_path.is_file():
                    candidates = [p for p in images if p.stem == mask_path.stem]
                    image_path = candidates[0] if candidates else None
                if image_path is not None and image_path.is_file():
                    with Image.open(image_path) as opened:
                        if opened.size != mask_size:
                            size_mismatches.append({
                                "image": image_path.name,
                                "image_size": list(opened.size),
                                "mask_size": list(mask_size),
                            })

            result["mask_pixel_audit"][split] = {
                "sampled_masks": len(selected),
                "sampled_pixels": total_pixels,
                "intensity_bins": bins,
                "sampled_unique_values_first_32_per_mask_union": sorted(sampled_values),
                "fraction_foreground_using_threshold_128": {
                    "mean": float(np.mean(occupied_fractions)) if occupied_fractions else None,
                    "min": float(np.min(occupied_fractions)) if occupied_fractions else None,
                    "max": float(np.max(occupied_fractions)) if occupied_fractions else None,
                },
                "image_mask_size_mismatch_count_in_sample": len(size_mismatches),
                "image_mask_size_mismatch_examples": size_mismatches[:10],
                "threshold_note": "Threshold 128 treats 0–7 as background and 248–255 as foreground in the inspected sample; review the intensity bins before finalizing this choice.",
            }

    if len(image_files_by_split) == 2:
        result["cross_split_exact_duplicates"] = cross_split_exact_duplicates(image_files_by_split)

    result["pairing_check_pass"] = not pair_errors
    return result


def audit_manifest(root: Path) -> dict:
    path = root / "manifest.csv"
    if not path.is_file():
        return {"available": False, "note": "manifest.csv not found; no metadata split audit was possible."}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)

    lowered = {field.lower(): field for field in fields}
    id_candidates = ("patient_id", "patient", "subject_id", "subject", "study_id", "case_id")
    id_column = next((lowered[key] for key in id_candidates if key in lowered), None)
    relative_column = lowered.get("relative_path")
    hash_column = lowered.get("sha256")
    task_column = lowered.get("task")
    split_column = lowered.get("split")
    mask_column = lowered.get("is_mask")

    result = {
        "available": True,
        "path": str(path),
        "row_count": len(rows),
        "columns": fields,
        "patient_or_subject_id_column": id_column,
    }

    if id_column:
        id_splits: dict[str, set[str]] = defaultdict(set)
        if split_column:
            for row in rows:
                identifier = (row.get(id_column) or "").strip()
                split = (row.get(split_column) or "").strip().lower()
                if identifier and split in {"train", "test", "val", "validation"}:
                    id_splits[identifier].add(split)
        overlaps = [
            {"id": identifier, "splits": sorted(splits)}
            for identifier, splits in id_splits.items()
            if len(splits) > 1
        ]
        result["patient_or_subject_ids_crossing_splits"] = len(overlaps)
        result["patient_or_subject_overlap_examples"] = overlaps[:20]
    else:
        result["patient_level_isolation"] = (
            "Cannot be confirmed from manifest.csv because no explicit patient/subject/study ID column was found."
        )

    if relative_column:
        missing_files = []
        checked_files = 0
        checksum_mismatches = []
        manifest_image_hashes: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row_number, row in enumerate(rows, start=2):
            raw_relative = (row.get(relative_column) or "").strip()
            if not raw_relative:
                missing_files.append({"row": row_number, "relative_path": ""})
                continue

            # The Kaggle manifest stores relative paths with Windows
            # separators (backslashes), even when audited on Linux/Colab.
            # Normalize separators before constructing a platform-native Path.
            normalized_relative = raw_relative.replace("\\\\", "/").replace("\\", "/")
            relative = Path(normalized_relative)
            candidates = (
                [relative] if relative.is_absolute()
                else [root / relative, root.parent / relative]
            )
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
            if resolved is None:
                missing_files.append({"row": row_number, "relative_path": raw_relative})
                continue

            checked_files += 1
            actual_hash = sha256(resolved)
            expected_hash = (row.get(hash_column) or "").strip().lower() if hash_column else ""
            if expected_hash and actual_hash.lower() != expected_hash:
                checksum_mismatches.append({
                    "row": row_number,
                    "relative_path": raw_relative,
                    "manifest_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                })

            # Compare image rows only (not masks), grouping inside each task.
            is_mask = (row.get(mask_column) or "").strip().lower() if mask_column else ""
            if hash_column and task_column and split_column and is_mask not in {"true", "1", "yes"}:
                task = (row.get(task_column) or "").strip()
                split = (row.get(split_column) or "").strip().lower()
                if task and split in {"train", "test"}:
                    # Group by bytes actually present on disk. The manifest
                    # checksum is separately compared to actual_hash above.
                    manifest_image_hashes[(task, actual_hash.lower())].append({
                        "split": split,
                        "relative_path": raw_relative,
                    })

        duplicate_groups = []
        for (task, digest), entries in manifest_image_hashes.items():
            split_names = sorted({entry["split"] for entry in entries})
            if len(split_names) > 1:
                duplicate_groups.append({
                    "task": task,
                    "sha256": digest,
                    "splits": split_names,
                    "files": entries,
                })

        result["file_path_validation"] = {
            "rows_with_relative_path": len(rows),
            "files_found_and_hashed": checked_files,
            "missing_or_unresolved_files": len(missing_files),
            "missing_file_examples": missing_files[:20],
        }
        if hash_column:
            result["sha256_validation"] = {
                "rows_with_comparable_hashes": sum(
                    1 for row in rows if (row.get(hash_column) or "").strip()
                ),
                "files_with_hash_mismatches": len(checksum_mismatches),
                "mismatch_examples": checksum_mismatches[:20],
                "note": "Compares each manifest SHA-256 against the file resolved from relative_path.",
            }
        if manifest_image_hashes:
            result["manifest_image_hashes_crossing_train_test"] = {
                "duplicate_hash_groups": len(duplicate_groups),
                "examples": duplicate_groups[:20],
                "note": "Groups manifest SHA-256 values by task and checks whether image hashes occur in both train and test. This is an additional check independent of the file-hash scan.",
            }

    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight audit for the downloaded BRISC 2025 dataset."
    )
    parser.add_argument("--root", type=Path, required=True, help="Folder containing classification_task, segmentation_task, and optionally manifest.csv")
    parser.add_argument("--output", type=Path, help="Optional path for a JSON audit report")
    parser.add_argument("--sample-masks", type=int, default=60, help="Number of masks to sample per split for intensity checks")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"Dataset root does not exist: {args.root}")

    report = {
        "dataset_root": str(args.root),
        "classification": audit_classification(args.root),
        "segmentation": audit_segmentation(args.root, sample_masks=args.sample_masks),
        "manifest": audit_manifest(args.root),
        "scope_warning": "Exact file hashes do not find near-duplicates. No patient-level separation should be claimed unless patient identifiers or source-level provenance establish it.",
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if not report["segmentation"]["pairing_check_pass"]:
        raise SystemExit("Dataset audit failed: segmentation images and masks need attention.")


if __name__ == "__main__":
    main()
