# BRISC 2025 Audit Findings

**Audit date:** 20 October 2026  
**Dataset source:** [BRISC 2025 on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)  
**Related dataset paper:** Fateh et al., [BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification](https://www.nature.com/articles/s41597-026-06753-y).

## Observed folder counts

| Task | Official train | Official test | Total |
|---|---:|---:|---:|
| Classification images | 5,000 | 1,000 | 6,000 |
| Segmentation image-mask pairs | 3,933 | 860 | 4,793 |

Classification has four expected folders: glioma, meningioma, pituitary and no_tumor. All four were present in both splits. Segmentation image-mask filename stems matched for all 3,933 training images and 860 test images, with zero missing and zero orphan masks in the observed audit.

## Critical finding: exact duplicate images cross train/test

The audit found **seven groups of byte-identical image hashes crossing train and test in the classification task**, and seven corresponding groups in the segmentation task. One group contained three training filenames that matched one test image; in total, nine training paths in each task were affected by cross-split duplicate hashes.

Examples from the source filenames:

- `brisc2025_train_01210_me_ax_t1.jpg` and `brisc2025_test_00344_me_ax_t1.jpg`
- `brisc2025_train_01304_me_ax_t1.jpg` and `brisc2025_test_00352_me_ax_t1.jpg`
- Training images `brisc2025_train_04176_pi_co_t1.jpg`, `brisc2025_train_04177_pi_co_t1.jpg`, and `brisc2025_train_04178_pi_co_t1.jpg` matched `brisc2025_test_00825_pi_co_t1.jpg`.

These exact copies contaminate an otherwise unseen-test evaluation if both copies are allowed to reach training and testing. The original download should be preserved for traceability, but do not train the first benchmark from the original train folders unchanged.

The `prepare_brisc.py` utility makes a separate copy, leaves the supplied test files unchanged, and removes from the training side any exact image hashes also found in test. It also deduplicates identical training samples when their labels/targets are consistent and excludes ambiguous duplicate-image groups when labels or segmentation masks conflict. The preparation report records removed paths and hashes.

**Scope limitation:** an exact hash check only finds byte-identical files. It does not find resized/augmented/near-duplicate images, and it cannot establish patient-wise isolation.

## Patient-level independence is unverified

The downloaded `manifest.csv` contained 15,586 rows and these columns: `relative_path`, `filename`, `task`, `split`, `index`, `tumor_code`, `tumor_label`, `plane_code`, `plane_label`, `sequence`, `is_mask`, `linked_image`, `width`, `height`, `file_size_bytes`, and `sha256`. It did **not** contain an explicit patient, subject, or study identifier.

Therefore this audit cannot verify that all slices/views belonging to one person are confined to one split. Do not describe the resulting split as patient-independent. The cleaned output is only *exact-duplicate-decontaminated*; broader data independence remains unknown.

## Mask encoding observations

The audit sampled 60 masks from each official split (not every mask pixel in the entire dataset):

| Split | Sampled masks | Pixels 0–7 | Pixels 8–127 | Pixels 128–247 | Pixels 248–255 |
|---|---:|---:|---:|---:|---:|
| Train | 60 | 15,235,989 | 1 | 3 | 278,867 |
| Test | 60 | 14,375,104 | 3 | 0 | 270,862 |

Together with the visual examples, this supports thresholding the grayscale masks at 128 as a sensible initial binary conversion: near-black values become background and near-white values become foreground. This choice must remain configurable and be checked with overlays and the audit report; it is not an independent validation of annotation quality.

Image and mask dimensions matched for all sampled masks in this audit. Mask resizing must use nearest-neighbor interpolation.

## Next experimental protocol

1. Keep the raw Kaggle download untouched.
2. Run `prepare_brisc.py` to generate a new prepared folder with a seeded validation split drawn only from eligible official training images.
3. Run `audit_brisc.py` on the raw source and verify the preparation report records zero exact train/validation-to-test hash overlap.
4. Train EfficientNet-B0 and U-Net on the prepared training folders, choose checkpoints using validation only, and run final metrics once on the untouched supplied test files.
5. Report the split as not verified patient-wise. Do not claim that this procedure resolves near-duplicate or subject leakage.
6. For paper verification, reproduce the BRISC authors' publicly linked baseline notebook/code as closely as possible and clearly distinguish it from our independent EfficientNet/U-Net baseline. A model being implemented or trained successfully is not evidence that a paper's numeric results have been reproduced.

## Public research artifacts

- [BRISC 2025 dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025)
- [BRISC maintainer's getting-started notebook](https://www.kaggle.com/code/briscdataset/getting-started-with-brisc-2025)
- [BRISC dataset paper and code availability statement](https://www.nature.com/articles/s41597-026-06753-y)
- [Swin-HAFNet / BRISC preprint](https://arxiv.org/abs/2506.14318)
- [YOLOv5-seg vs. YOLOv7-seg paper](https://doi.org/10.1109/ACCESS.2024.3359418)

Status: data-quality warning documented; baseline training not yet run.