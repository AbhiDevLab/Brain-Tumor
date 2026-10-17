# Experiment roadmap

## Phase 0 — Check the data

- Choose a dataset for each task and avoid mixing incompatible labels by default.
- Record provenance, license, patient/study counts, acquisition sequence, annotations, and limitations.
- Create patient-wise train/validation/test manifests whenever identifiers exist.
- Run `audit_dataset.py` and inspect image-mask alignment.

**Exit criteria:** split manifest reviewed, known patient overlap resolved, masks checked, and class counts recorded.

## Phase 1 — Classification baseline

- Train EfficientNet-B0 from scratch, then repeat with ImageNet-pretrained weights.
- Report accuracy, macro-F1, weighted-F1, per-class precision/recall, and a confusion matrix.
- Save the seed, configuration, class order, training history, and checkpoint hash.

**Exit criteria:** a baseline evaluated once on a held-out test set, with errors reviewed by class and source.

## Phase 2 — Segmentation baseline

- Train a compact U-Net with BCE-only loss.
- Compare it with BCE + soft-Dice loss while keeping the architecture and data split fixed.
- Report per-image Dice/IoU, tumor-present Dice, and false positives on empty masks.
- Review false positives, missed regions, and boundary errors.

**Exit criteria:** held-out metrics and qualitative examples, with masks checked at original resolution.

## Phase 3 — Application

- Support image upload, input checks, classification and segmentation output, overlays, and a downloadable JSON report.
- Avoid saving uploaded scans by default.
- Show a clear status when weights are missing or a format is unsupported.

**Exit criteria:** no predictions without a trained checkpoint; outputs identify the model/checkpoint used.

## Phase 4 — Model comparison

Add one change at a time and keep patient-level splits fixed:

1. EfficientNet-B0 from scratch versus ImageNet-pretrained weights.
2. YOLO segmentation/detection, if annotation conversion and the official framework setup can be verified.
3. U-Net variants or nnU-Net if the dataset and compute allow it.
4. A Swin-HAFNet-inspired transformer/attention track with ablations for each component.
5. Grad-CAM or another saliency method, plus checks that explanations are stable under small perturbations.

**Exit criteria:** same splits, preprocessing and test protocol; compute cost, ablations and errors recorded.

## Phase 5 — Robustness

- Evaluate on an independent dataset/site when feasible.
- Stress-test resolution, intensity changes, artifacts, missing sequences, and small tumors.
- Measure uncertainty and calibration; add an abstain/cannot-assess path for unsuitable inputs.
- Seek qualified review of errors and overlays before making any clinical-use claim.

## Research questions

1. Does ImageNet transfer learning improve macro-F1 under patient-wise splits?
2. Does a stronger segmentation model improve Dice for small tumors, not just the overall mean?
3. Do shared classification/segmentation features help compared with separate models?
4. How much does performance decline on a different dataset or source?
5. Are visual explanations stable under clinically irrelevant image perturbations?
