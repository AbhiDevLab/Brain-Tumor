# Previous Run — Historical Baseline Reference

**Recorded on:** 2026-10-20  
**Purpose:** Preserve prior-run measurements and recovery status for the clean restart. These are historical results and should not automatically be treated as reproducible on a regenerated dataset.

## Repository

- Repository: `AbhiDevLab/Brain-Tumor`
- Default branch: `main`
- Code and reusable audit/preparation/training/evaluation scripts are versioned in GitHub.

## Classification — EfficientNet-B0

Previously reported settings:
- Pretrained EfficientNet-B0
- Epochs: 15; batch size: 16; learning rate: 0.0001
- Image size: 224; seed: 42; early-stopping patience: 5

Reported original test-set results:
- 1,000 entries; accuracy 0.994
- Macro-F1 0.993944785
- Weighted-F1 0.994000496

Reported unique-image results:
- 995 unique hashes after excluding five redundant entries
- Accuracy 0.993969849
- Macro-F1 0.993907941
- Weighted-F1 0.993970358

Previously saved in Drive:
- `BrainTumorOutputs/classifier.pt`
- Classification evaluation outputs, including unique-image metrics

## Segmentation — U-Net with BCE

Previously reported settings:
- Epochs: up to 30; batch size: 4; learning rate: 0.001
- Image size: 256; seed: 42; early-stopping patience: 6
- Checkpoint selected by validation loss; recorded minimum validation loss was about 0.01516 at epoch 20.

Clean-test results:
- Test images: 854
- Mean Dice: 0.8004178215 (80.04%)
- Mean IoU: 0.7177789213 (71.78%)

Previously saved in Drive:
- `BrainTumorOutputs/segmenter_bce.pt`
- `BrainTumorOutputs/segmentation_bce_metrics.json`
- `BrainTumorOutputs/segmentation_bce_unique_metrics.json`

## Segmentation — BCE + Dice (incomplete)

Training was interrupted after three epochs with `KeyboardInterrupt`.

| Epoch | Training loss | Validation loss | Validation Dice | Validation IoU |
|---|---:|---:|---:|---:|
| 1 | 0.757431 | 0.572630 | 0.515455 | 0.399384 |
| 2 | 0.464363 | 0.436029 | 0.631498 | 0.535908 |
| 3 | 0.399230 | 0.484087 | 0.584188 | 0.505637 |

A file named `BrainTumorOutputs/segmenter_bce_dice.pt` was found in Drive, but this was only an interim checkpoint. No final clean-test evaluation was completed. Do not cite it as a completed result.

## Dataset audit notes

- Original classification dataset: 5,000 train and 1,000 official test images.
- Original segmentation dataset: 3,933 train pairs and 860 official test pairs.
- The segmentation test entries had 856 unique image hashes; two duplicate-image pairs had conflicting masks.
- The documented clean-evaluation protocol kept one image from groups with identical masks and excluded every item from conflicting duplicate groups, producing 854 images.
- Patient-level independence was not confirmed; SHA-256 checks do not rule out near-duplicates.

Previously prepared data counts, to be re-verified on the fresh run:
- Classification: 4,210 train, 745 validation, 1,000 official test.
- Segmentation: 3,301 train pairs, 584 validation pairs, 860 original official-test entries before clean-evaluation exclusions.

## Restart guidance

1. Preserve `AbhiDevLab/Brain-Tumor`; inspect current source code before adding or duplicating scripts.
2. Re-download and re-audit BRISC 2025 if raw data is unavailable.
3. Recreate prepared data and the clean segmentation evaluation set; verify copied files by counts and hashes.
4. Before deleting old Drive outputs, ensure the historical model checkpoints are intentionally archived or retained. GitHub's text reference alone is **not** a backup of binary model weights.
5. In the new workflow, back up checkpoints to Google Drive during training and mirror completed checkpoint binaries to GitHub using Git LFS or a versioned Release asset when configured and verified.
6. Keep the raw MRI dataset in Drive or retrieve it from Kaggle under the dataset's terms; do not commit raw data to ordinary Git history.
7. Save run configurations, epoch histories, metrics, reports, and a SHA-256 inventory to both Drive and GitHub. Verify that a remote copy can be listed or retrieved before considering it backed up.
