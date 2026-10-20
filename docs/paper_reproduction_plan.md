# Paper Reproduction and Claim-Verification Plan

**Status:** planning and audit stage. No paper result has been reproduced in this repository yet.

This plan is designed to distinguish (1) a paper's reported result, (2) an implementation that appears to follow the method, and (3) a controlled experimental result actually produced by our own run.

## 1. Research sources already reviewed

### A. Published BRISC dataset paper: baseline claims to reproduce first

- Fateh et al., **“BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification,”** *Scientific Data* 13, Article 361 (published 5 February 2026). https://www.nature.com/articles/s41597-026-06753-y
- Dataset: https://www.kaggle.com/datasets/briscdataset/brisc2025/
- Maintainer's notebook: https://www.kaggle.com/code/briscdataset/getting-started-with-brisc-2025

The published paper reports that the custom code for its baseline models is available at the BRISC Kaggle repository. Use that official code/notebook as the first reference implementation and inspect its exact split, preprocessing, training settings and metric definitions.

**Published results described in the paper:**

| Task | Model | Paper-reported metric |
|---|---|---:|
| Classification | EfficientNet-B0 | Accuracy 0.9920 ± 0.0000; weighted F1 0.9920 ± 0.0000 |
| Binary tumor segmentation, aggregated by tumor type | U-Net | Weighted mIoU 75.7% |
| Binary tumor segmentation, aggregated by tumor type | SaberNet | Weighted mIoU 80.6% |

The paper says each classification baseline was trained/tested three times; these results are the paper's values, not reproduced results from this repository. Confirm details from the full tables and the authors' code before making a strict comparison.

**Critical data integrity discrepancy:** the paper states that exact and near-duplicate images were removed before its train/test split, but our SHA-256 audit of the Kaggle download found seven exact duplicate groups crossing train/test in each task. Our downloaded manifest also has no explicit patient/subject identifier. The audit and the paper therefore do not yet support a claim of patient-independent separation. Cross-check each manifest SHA-256 against the downloaded file hashes, confirm the release snapshot, and inspect the authors' implementation before drawing conclusions. The cleaned benchmark in this project is a separate, exact-duplicate-decontaminated variant; it must not be described as the untouched official split.

**Comparison protocol:** our current EfficientNet-B0 and U-Net scripts are independent baselines. Even when trained on the same dataset, they do not count as a faithful reproduction of the published results unless preprocessing, splits, architecture, training schedule, and metric definitions match. Report raw metrics, deviations, and whether results are reproduced or only comparable.

### B. BRISC dataset and Swin-HAFNet / Swin-HAFUNet line### A. BRISC dataset and Swin-HAFNet / Swin-HAFUNet line

- Dataset paper: Fateh et al., **“BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification,”** *Scientific Data* (2026). https://doi.org/10.1038/s41597-026-06753-y
- Related method manuscript: Fateh et al., **“BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification with Swin-HAFNet,”** arXiv:2506.14318. https://arxiv.org/abs/2506.14318
- Dataset: https://www.kaggle.com/datasets/briscdataset/brisc2025
- Maintainer's getting-started notebook: https://www.kaggle.com/code/briscdataset/getting-started-with-brisc-2025

**Claims captured in the supplied research review:** weighted segmentation mIoU 82.4%, with reported tumor-type mIoU values of 76.0% (glioma), 85.0% (meningioma) and 85.3% (pituitary). A manuscript version also reports 99.63% classification accuracy. These numbers are paper-reported reference points, not reproduced results.

**Verification status:** not yet reproduced. The public BRISC release links code for baseline models and a getting-started notebook, but our preliminary source check has not established that a complete, author-maintained Swin-HAFNet implementation with all training configuration and checkpoint details is openly available. Before calling a run a *faithful reproduction*, verify the exact manuscript version, architecture, preprocessing, loss, split, checkpoints, and hyperparameters. A from-scratch implementation should be labelled an *independent implementation inspired by the paper*, not an exact reproduction.

**Candidate architecture points to verify:** hierarchical shifted-window Swin encoder; contextual bottleneck enhancement; hierarchical attention fusion and any deformable-convolution branches; decoder; multi-scale classification branch; BCE + Dice segmentation objective; and ablation results for individual modules. Implement and test one module at a time.

### B. YOLOv5-seg versus YOLOv7-seg

Almufareh et al., **“Automated Brain Tumor Segmentation and Classification in MRI Using YOLO-Based Deep Learning,”** *IEEE Access* 12 (2024), 16189–16207. https://doi.org/10.1109/ACCESS.2024.3359418

**Paper-reported figures captured in the supplied research review:**

| Model | Task / metric | Reported value |
|---|---|---:|
| YOLOv5 | Box precision | 0.940 |
| YOLOv5 | Box recall | 0.905 |
| YOLOv5 | Box mAP@0.5 | 0.947 |
| YOLOv5 | Mask recall | 0.906 |
| YOLOv5 | Mask mAP@0.5 | 0.947 |
| YOLOv7 | Box precision | 0.936 |
| YOLOv7 | Box recall | 0.904 |
| YOLOv7 | Box mAP@0.5 | 0.940 |
| YOLOv7 | Mask recall | 0.903 |
| YOLOv7 | Mask mAP@0.5 | 0.941 |

Confirm values and terminology against the paper's tables before quoting them in a report.

**Important comparability limitations:** the paper uses three tumor classes and a different dataset/annotation conversion pipeline than our BRISC four-class classification plus binary-mask setup. Its reported split counts in our notes (2,144 train, 827 validation, 91 test) do not sum to its reported 3,064 images, and the results discussion appears to describe evaluation on 827 images. Resolve these details from the full paper before attempting a strict reproduction. The current preliminary review did not identify a clearly linked official author code repository.

**Verification status:** not yet reproduced. A later YOLO comparison on BRISC would be a controlled *new benchmark*, not a faithful reproduction of the paper's original results. A faithful reproduction would need the original dataset, labels, preprocessing, model versions, split, hyperparameters and evaluation procedure.

### C. Comprehensive deep-learning review

Dudhe, Burade and Nikose, **“Comprehensive Review of Brain Tumor Segmentation and Classification Using Deep Learning Techniques,”** IEEE SCEECS (2025). https://doi.org/10.1109/SCEECS64059.2025.10940343

This is a survey of methods, not one executable model or one consistent test benchmark. It cannot be reproduced as a single experiment. Use it to define method families, metrics and limitations; do not copy a number from its table as though it were a baseline measured on BRISC.

### D. Project synopsis

The supplied synopsis is a proposal, not a validated result paper. Its target accuracy and feature list are aspirations, not evidence. We should test the actual features individually and report results without assuming the targets will be reached.

## 2. Verification language

Use these labels in notebooks, JSON reports and the final project report:

1. **Paper-reported:** quoted from a named paper/version; not our result.
2. **Implementation inspected:** architecture/code reviewed, but no matching experiment run.
3. **Independent implementation:** built from the method description; differences from official code are documented.
4. **Reproduced under matching protocol:** same dataset/version, split, labels, preprocessing, metric definition and material training settings; results fall within an explained tolerance.
5. **Comparable benchmark:** a method is implemented on our agreed BRISC split, but it does not reproduce the original paper's dataset or protocol.
6. **Not verified:** code, weights, split, hyperparameters or annotation conversion needed to support the claim are unavailable or unresolved.

Never label a result “reproduced” merely because the code runs or a metric is high.

## 3. First experiment sequence

### Phase 0 — Data integrity (current phase)

- Record dataset source/version and license.
- Inspect the manifest, class counts, filenames, image dimensions and masks.
- Inspect mask grayscale intensity values. Samples reviewed during setup show values near 0–7 and 248–255; this repository now uses a configurable threshold (default 128) rather than treating every nonzero grayscale value as tumor. Confirm the intensity distribution across samples before finalizing this preprocessing choice.
- Check exact-byte duplicates across official train/test splits separately for the classification images and segmentation images. Exact hash matches do not detect near-duplicates.
- Look for patient/subject/study identifiers. If none are provided, document that patient-level isolation cannot be confirmed; do not claim it has been established by filename or image-hash checks alone.
- Preserve the official test set. Make validation data from the official training set only.

**Exit criteria:** no unmatched image/mask pairs; mask encoding understood; class counts recorded; exact duplicate results reviewed; split limitations documented.

### Phase 1 — Reproducible baselines

1. EfficientNet-B0 classification, first with ImageNet-pretrained weights and then (optionally) from scratch. Report accuracy, macro-F1, per-class precision/recall and confusion matrix.
2. U-Net binary segmentation using BCE, then BCE + soft-Dice under an otherwise identical configuration. Report per-image Dice/IoU, tumor-present Dice, and false-positive behaviour.
3. Fix random seeds and record Python/PyTorch/torchvision versions, GPU, batch size, image size, optimizer, learning rate, epochs, early stopping, dataset manifest hash and checkpoint hash.
4. Use validation data to choose checkpoints. Evaluate the official test data only after the configuration is locked.

BRISC has fewer segmentation pairs than classification images. The no-tumor classification images do not appear in the paired-mask counts. Do not assume empty-mask negatives exist in the segmentation task; if using no-tumor images to measure or train against false positives, add them as an explicit, documented experiment with all-zero masks rather than silently treating them as annotated segmentation samples.

### Phase 2 — Paper-method comparisons

1. Reproduce the BRISC authors' public baseline notebook as closely as its code permits.
2. Implement Swin-HAFNet/Swin-HAFUNet only after identifying an authoritative version of its architecture and training procedure. If exact code or settings are unavailable, label our implementation as independent and report architectural deviations.
3. Convert BRISC masks into YOLO segmentation labels only after testing polygon extraction/contour conversion visually and recording the conversion method. Compare YOLO models on the same BRISC split as a new benchmark, not against the original paper's mAP as if the dataset/protocol were identical.
4. Run one-variable-at-a-time ablations (BCE vs BCE+Dice; then attention/fusion additions). Record compute cost and difficult-case error analysis.

### Phase 3 — Claim verification report

For every paper/method, create a row with: citation/version; repository/code URL; commit/tag; dataset and split; preprocessing and annotation format; architecture components; loss/optimizer/epochs; reported metric and evaluation set; our metric and evaluation set; deviations; run status; and conclusion about the claim.

A final conclusion should be one of: *reproduced within tolerance*, *partially reproduced*, *not reproduced under our setup*, or *not verifiable from available artifacts*. A mismatch does not alone prove the paper claim false; first rule out protocol differences and implementation errors.

## 4. Relevant source links

- BRISC dataset paper and data/code availability: https://www.nature.com/articles/s41597-026-06753-y
- BRISC/Swin-HAFNet arXiv manuscript: https://arxiv.org/abs/2506.14318
- BRISC Kaggle dataset: https://www.kaggle.com/datasets/briscdataset/brisc2025
- BRISC maintainer notebook: https://www.kaggle.com/code/briscdataset/getting-started-with-brisc-2025
- YOLOv5/YOLOv7 segmentation and classification paper: https://doi.org/10.1109/ACCESS.2024.3359418
- 2025 review: https://doi.org/10.1109/SCEECS64059.2025.10940343
