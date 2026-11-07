# Brain Tumor MRI Analysis

Research prototype for four-class classification of 2D brain MRI images and binary foreground segmentation, using the project's existing trained EfficientNet-B0 classifier and U-Net checkpoints. The Streamlit interface is a research and educational demonstration, not a medical device.

> **Research-use warning:** The models have not been clinically validated. Scores and masks can be wrong. Do not use them for diagnosis, treatment, triage, or other medical decisions. Upload only de-identified 2D images, and have real MRI studies reviewed by qualified medical professionals.

## Current status — 7 November 2026

The research experiments are complete according to the run results supplied by the researcher, and this branch adds an inference pipeline and online-platform UI. The actual model files remain excluded from ordinary Git history and are attached to the repository's published pre-release. The pinned asset IDs, sizes and SHA-256 metadata have been verified. GitHub Actions successfully downloaded all three best checkpoints using the repository's read-only `BRAIN_MRI_GITHUB_TOKEN` secret, verified their SHA-256 hashes, loaded the actual trained weights, and passed a synthetic software-contract inference test on commit `51322230fc261572f1e328d0d67d6c0cd84d085e`. This smoke test is not an MRI performance test. Live hosting and end-to-end testing on an approved de-identified MRI remain separate acceptance gates.

- [x] Existing EfficientNet-B0 and U-Net architecture/preprocessing inspected.
- [x] Feature branch `feature/online-mri-research-platform` created from `research/colab-baseline-2026-10-21`.
- [x] Checkpoint path configuration, strict architecture/state-dictionary loading, private HTTPS and pinned GitHub Release retrieval with SHA-256 verification added.
- [x] Image validation, in-memory inference functions, model-score reporting, mask/overlay alignment, preliminary consistency rules, JSON report construction, and unit tests added.
- [x] Streamlit interface updated with de-identified upload guidance, result dashboard, Grad-CAM distinction, error handling, and research warnings.
- [x] GitHub Actions unit-test suite passed on the updated CPU-only inference dependency setup.
- [x] Published the checkpoint pre-release and configured the repository Actions secret `BRAIN_MRI_GITHUB_TOKEN` with repository-scoped Contents: Read-only permission.
- [x] Real-checkpoint smoke test passed: downloaded classifier and both U-Net best checkpoints, verified SHA-256, loaded the actual trained weights, and validated software inference outputs on a synthetic pattern.
- [x] Created isolated branch `final-review-enhancements` from `feature/online-mri-research-platform` for the final project review.
- [x] Added separate Overview, MRI Analysis, Classification, Segmentation, Explainable AI, and Research & Metrics views.
- [x] Added same-image BCE U-Net vs BCE + soft Dice comparison; BCE remains the default.
- [x] Added probability-map-backed threshold exploration and adjustable Grad-CAM overlay intensity without rerunning model inference.
- [x] Added measured inference-path timing, PDF research report export, and expanded JSON output.
- [x] Added a research metrics view that reads raw evaluation JSON when present and otherwise shows only documented aggregate results.
- [ ] Run CI on `final-review-enhancements`, address any failures, and verify all application pages locally.
- [ ] Deploy this branch and verify both private checkpoints plus a permitted de-identified MRI on the live app.
- [ ] Deploy the app, configure the token in the hosting provider's server-side secrets, and verify the live URL.
- [ ] Complete an end-to-end check using an approved de-identified MRI; this is separate from the synthetic smoke test.
- [ ] Configure private artifact access in the hosting provider, deploy, and verify the live URL.

### Reported experiment results

The metrics in this table were supplied by the researcher and have not been recomputed in the current runtime.

| Model | Evaluation set | Reported metrics |
|---|---|---|
| EfficientNet-B0 classifier | 1,000-image classification test set | Accuracy **99.40%**; macro-F1 **0.993945**; weighted-F1 **0.994000** |
| BCE U-Net | 854 cleaned segmentation image-mask pairs | Mean Dice **0.799568**; mean IoU **0.718709** |
| BCE + soft Dice U-Net | Same 854 pairs | Mean Dice **0.799782**; mean IoU **0.716567** |

The paired bootstrap comparison reportedly found no convincing overall improvement from BCE + Dice; both 95% confidence intervals included zero. The platform therefore defaults to the BCE U-Net. These evaluation metrics are not performance guarantees on arbitrary MRI images and do not establish external, patient-independent, or clinical validation.

## Project contents

- `app.py` — multi-section Streamlit research dashboard, inference, threshold/Grad-CAM controls, model comparison, measured timing, and PDF/JSON exports.
- `src/brain_tumor/models.py` — existing EfficientNet-B0 and U-Net architecture definitions.
- `src/brain_tumor/data.py` — existing classifier and segmenter preprocessing.
- `src/brain_tumor/explain.py` — classifier Grad-CAM attribution visualization.
- `src/brain_tumor/inference.py` — checkpoint validation/loading, secure private artifact retrieval, input validation, prediction, preliminary consistency checks, and JSON-ready report.
- `src/brain_tumor/reports.py` — in-memory PDF research report generation from the current inference outputs.
- `train_classifier.py`, `train_segmenter.py` — training workflows (not invoked by the web app).
- `evaluate_classifier.py`, `evaluate_segmenter.py` — evaluation scripts.
- `tests/` — architecture, preprocessing, loss, and inference-contract tests.
- `docs/online_platform_deployment.md` — artifact and hosting instructions.
- `docs/brisc_audit_findings.md` — dated data-audit findings and split caveats.
- `docs/paper_reproduction_plan.md` — research claim and reproduction checklist.
- `docs/research_gap_and_proposed_contribution.md` — methods and proposed experiments.
- `docs/project_status_summary.md` — consolidated status; its original 20 October snapshot is retained as history.

## Setup and local execution

Python 3.11 is recommended. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

On Linux/macOS, create and activate a virtual environment using the platform's usual `python3 -m venv .venv` and activation commands.

Without checkpoints, the app should clearly report that the relevant model is unavailable; it must not generate a fake prediction. Configure the paths or private artifact retrieval as described in [Online platform deployment](docs/online_platform_deployment.md).

### Local checkpoint paths

If the checkpoints are available as files that the application process can read, set these environment variables before launching Streamlit:

- `BRAIN_MRI_CLASSIFIER_CHECKPOINT` — existing EfficientNet-B0 checkpoint.
- `BRAIN_MRI_SEGMENTER_CHECKPOINT` — existing BCE U-Net checkpoint.

For the existing published pre-release in the private GitHub repository, the hosted app can use a fine-grained, repository-scoped `Contents: Read-only` token in the server-side secret `BRAIN_MRI_GITHUB_TOKEN`. The release asset IDs and SHA-256 checks are pinned in the inference module. See [online platform deployment](docs/online_platform_deployment.md) for the redirect security model and exact Streamlit configuration.

The original research paths are:

- `/content/drive/MyDrive/BrainTumorResearch/models/classifier/efficientnet_b0_best_20261020T192345Z.pt`
- `/content/drive/MyDrive/BrainTumorResearch/models/segmenter/unet_bce_best_20261020T195800Z.pt`
- Optional research comparison, not default: `/content/drive/MyDrive/BrainTumorResearch/models/segmenter/unet_bce_dice_best_20261020T205900Z.pt`

These are Colab/Drive paths, not automatically available to an online host. For deployment, use a private HTTPS artifact endpoint and the variables documented in [Online platform deployment](docs/online_platform_deployment.md). The endpoint must use a separate read-only token and expected SHA-256 hash; do not put raw model binaries, credentials, or secrets in ordinary Git history. The app verifies the downloaded artifact's hash before loading it.

## Data contract and limitations

This version accepts 2D PNG, JPG/JPEG, BMP and TIFF/TIF images up to 12 MiB. It validates extension/content, decodability, pixel dimensions and file size. For a multi-frame TIFF, the first frame is used.

The classifier follows its saved checkpoint class order and the repository's ImageNet-normalized preprocessing (generally 224 × 224). Its displayed softmax values are **uncalibrated model scores**, not validated probabilities.

The U-Net follows its training-compatible RGB scaling to [0, 1] (generally 256 × 256), without adding ImageNet normalization. Its default mask threshold is 0.50, which has not yet been calibrated through the planned validation-only threshold experiment. Foreground pixels/fraction are image-grid measures, not physical tumor measurements.

Grad-CAM is a classifier attribution visualization and is not the U-Net segmentation mask. The predicted mask is model-generated and is not a clinically confirmed tumor boundary. Preliminary classifier/segmenter consistency rules may flag disagreements, but have not been validated, are not a reliability gate, and do not suppress either output.

**Not supported:** DICOM, NIfTI, full 3D volumes, multiple MRI sequences, physical voxel measurements, patient records, clinical interpretation, and external clinical validation.

## Research history and data audit

The original research documents preserve the earlier planning status and dated audit findings. In particular, the 20 October 2026 BRISC audit recorded seven groups of byte-identical images crossing train/test in each task and noted that the manifest did not provide explicit patient/subject identifiers. See [BRISC audit findings](docs/brisc_audit_findings.md) and [paper reproduction plan](docs/paper_reproduction_plan.md). Do not claim that these splits establish patient-wise independence. Exact hashes do not detect near-duplicates or identify the same patient across different images.

The earlier proposed contribution remains a set of research hypotheses. The small reported mean Dice difference between BCE and BCE + Dice is not convincing evidence of an improvement. Attention-gated U-Net changes, reliability gating, threshold calibration, model calibration and external validation remain planned work—not features or findings already validated by this platform.

## Deployment

See [docs/online_platform_deployment.md](docs/online_platform_deployment.md) for Streamlit Community Cloud setup, private artifact retrieval, server-side secret configuration, local paths, resource limitations and pre-deployment checks.

Do not call the application deployed until the hosting provider returns a real URL and a smoke test confirms that the live process can load both intended checkpoints and run inference. The final-review enhancement branch has not yet been deployed or live-verified.
