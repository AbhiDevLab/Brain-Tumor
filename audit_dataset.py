from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_classification(root: Path) -> dict:
    summary = {"kind": "classification", "root": str(root), "splits": {}, "identical_files_crossing_splits": []}
    hashes = defaultdict(list)
    known_splits = [s for s in ("train", "val", "test") if (root / s).is_dir()]
    for split in known_splits:
        split_dir = root / split
        classes = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            files = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
            classes[class_dir.name] = len(files)
            for path in files:
                hashes[digest(path)].append({"split": split, "class": class_dir.name, "path": str(path)})
        summary["splits"][split] = {"classes": classes, "total_images": sum(classes.values())}
    summary["class_sets_match"] = len({tuple(sorted(x["classes"])) for x in summary["splits"].values()}) <= 1
    for same_hash_files in hashes.values():
        if len({x["split"] for x in same_hash_files}) > 1:
            summary["identical_files_crossing_splits"].append(same_hash_files)
    summary["exact_duplicate_groups_across_splits"] = len(summary["identical_files_crossing_splits"])
    summary["note"] = "Hash checks detect exact binary duplicates only; they do not prove patient-wise isolation. Validate source patient IDs separately."
    if not known_splits:
        summary["error"] = "No train/val/test directories were found."
    return summary


def audit_segmentation(root: Path) -> dict:
    summary = {"kind": "segmentation", "root": str(root), "splits": {}, "cross_split_exact_duplicates": []}
    seen_hashes = defaultdict(list)
    for split in ("train", "val", "test"):
        image_dir, mask_dir = root / "images" / split, root / "masks" / split
        if not image_dir.is_dir() or not mask_dir.is_dir():
            summary["splits"][split] = {"error": f"Missing {image_dir} or {mask_dir}"}
            continue
        images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS}
        missing_masks = sorted(set(images) - set(masks))
        extra_masks = sorted(set(masks) - set(images))
        for img in images.values():
            seen_hashes[digest(img)].append({"split": split, "kind": "image", "path": str(img)})
        summary["splits"][split] = {"images": len(images), "masks": len(masks),
                                    "missing_masks": missing_masks, "masks_without_images": extra_masks,
                                    "paired_images": len(set(images) & set(masks))}
    for group in seen_hashes.values():
        if len({x["split"] for x in group}) > 1:
            summary["cross_split_exact_duplicates"].append(group)
    summary["exact_duplicate_groups_across_splits"] = len(summary["cross_split_exact_duplicates"])
    summary["note"] = "Only exact image-byte duplicates are tested. Mask content, patient IDs, and annotation quality require separate review."
    return summary


def main():
    p = argparse.ArgumentParser(description="Audit dataset structure and flag basic split leakage risks")
    p.add_argument("--classification-dir", type=Path)
    p.add_argument("--segmentation-dir", type=Path)
    p.add_argument("--patient-manifest", type=Path, help="Optional CSV with columns path,patient_id,split for patient-wise split audit")
    p.add_argument("--output", type=Path, help="Optional JSON output file")
    args = p.parse_args()
    if not args.classification_dir and not args.segmentation_dir:
        p.error("Provide --classification-dir and/or --segmentation-dir")
    results = []
    if args.classification_dir:
        results.append(audit_classification(args.classification_dir))
    if args.segmentation_dir:
        results.append(audit_segmentation(args.segmentation_dir))
    if args.patient_manifest:
        manifest = args.patient_manifest
        if not manifest.is_file():
            p.error(f"Patient manifest not found: {manifest}")
        patients_by_split = defaultdict(set)
        rows, invalid_rows = 0, []
        with manifest.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"path", "patient_id", "split"}
            if not required.issubset(set(reader.fieldnames or [])):
                p.error("Patient manifest must have columns: path,patient_id,split")
            for row_number, row in enumerate(reader, start=2):
                rows += 1
                patient_id = (row.get("patient_id") or "").strip()
                split = (row.get("split") or "").strip()
                path = (row.get("path") or "").strip()
                if not patient_id or split not in {"train", "val", "test"} or not path:
                    invalid_rows.append(row_number)
                    continue
                patients_by_split[split].add(patient_id)
        pair_overlaps = {}
        split_names = ("train", "val", "test")
        for i, first in enumerate(split_names):
            for second in split_names[i + 1:]:
                shared = sorted(patients_by_split[first] & patients_by_split[second])
                if shared:
                    pair_overlaps[f"{first}__{second}"] = shared
        results.append({"kind": "patient_split_manifest", "path": str(manifest), "rows": rows,
                        "patients_per_split": {k: len(v) for k, v in patients_by_split.items()},
                        "overlap_between_splits": pair_overlaps, "invalid_rows": invalid_rows,
                        "patient_isolation_pass": not pair_overlaps and not invalid_rows})
    payload = results[0] if len(results) == 1 else {"audits": results}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
