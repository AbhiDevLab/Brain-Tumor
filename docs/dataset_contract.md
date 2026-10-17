# Dataset contract and acceptance checklist

## Classification

**Input:** a de-identified 2D MRI slice as PNG, JPG, or TIFF. **Label:** one mutually exclusive class, such as `glioma`, `meningioma`, `pituitary`, or `no_tumor`.

Before training:

- Record the exact source URL, dataset version, license, modality/sequence, plane, patient count, and label definitions.
- Confirm whether labels are patient-, study-, volume-, or slice-level.
- Keep every patient/study in exactly one of train, validation, or test; include all related slices and modalities in the same split.
- Check class balance, image dimensions, corrupted files, duplicates, and near-duplicates.
- Leave the test set untouched until model choices are locked.
- If multiple datasets are used, retain source identifiers and report results by source.

## Segmentation

**Input:** 2D MRI slice. **Target:** binary mask where non-zero denotes tumor. Do not collapse a multi-class mask unless that is the intended target and is documented.

- Inspect image/mask alignment and orientation before preprocessing.
- Resize masks with nearest-neighbor interpolation, not bilinear interpolation.
- Check for missing masks, orphan masks, and filename-stem mismatches.
- Record how masks were produced and who annotated them, where known.
- Report per-case Dice and IoU, with tumor-present and empty-mask cases handled explicitly.
- Inspect small lesions, boundaries, false positives, and false negatives.

## 2D and 3D data

The current code works with 2D image folders. BraTS-style NIfTI volumes and multimodal T1/T1ce/T2/FLAIR data need a separate volumetric loader, consistent orientation/resampling, modality handling, and patient-level splits. Do not discard volume metadata if 3D analysis is part of the research goal.
