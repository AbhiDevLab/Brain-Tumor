# Brain Tumor MRI Project — Consolidated Status

**Last updated:** 20 October 2026  
**Repository:** https://github.com/AbhiDevLab/Brain-Tumor

## Objective

Build an academic research prototype that classifies 2D brain MRI scans and segments suspected tumor regions where valid expert annotations are available. This is not a clinically validated diagnostic system.

## Research basis and baseline plan

The project review covered four supplied research documents spanning YOLO-based detection/segmentation, a broad deep-learning review, the Swin-HAFNet multi-task transformer direction, and the project synopsis.

First establish separate baselines:
- **EfficientNet-B0 classifier:** glioma, meningioma, pituitary tumor, and no tumor.
- **U-Net segmenter:** binary tumor/background mask, trained only on images with valid expert masks.

Prior research already includes CNN transfer-learning classification, U-Net-style segmentation, YOLO-based localization, and attention/transformer/multi-task approaches. These methods are not claimed as new inventions by this project.

## Planned experiments

1. **A — Baseline:** train U-Net with binary cross-entropy (BCE).
2. **B — Loss comparison:** test BCE + Dice while keeping the split and other settings fixed.
3. **C — Attention:** test attention-gated skip connections when baseline error analysis justifies the added complexity.
4. **D — Difficult cases:** separately analyze small, irregular, low-contrast tumors and inspect missed regions and false positives.

Use a fixed, leakage-aware held-out evaluation protocol. Primary segmentation target is Dice Similarity Coefficient (DSC), with tumor sensitivity/recall as an important safeguard; also report IoU and false-positive behavior. For classification, report macro-F1 and per-class recall. Retain an improvement only if held-out results support it without unacceptable recall loss or unjustified computational cost.

## Repository implementation published

The repository currently includes:
- EfficientNet-B0 and U-Net model components.
- Separate classifier and segmenter training scripts.
- Held-out evaluation scripts.
- BCE and BCE + Dice segmentation-loss options.
- Dataset structure, duplicate and image-mask pairing checks.
- Research review, experiment roadmap and dataset contract.
- Streamlit application scaffold and Grad-CAM attribution visualization.
- JSON result reporting, pytest tests and a GitHub Actions workflow.
- .gitignore safeguards for local environments, secrets, data and generated artifacts.

Earlier package smoke tests recorded five passing tests, but the current published code's full local test result and the latest GitHub Actions outcome were not yet confirmed in the last project summary. Check these again locally before training.

## What has not yet been demonstrated

- Current published code's full test suite in the target Windows environment.
- Streamlit launch.
- BRISC dataset audit and confirmation of the license/version.
- Real-data training.
- Real classification and segmentation metrics.
- Any improvement in tumor-detection accuracy, segmentation quality or clinical utility.

Synthetic tests are not medical-image validation. Do not describe proposed gains as achieved results.

## Immediate next steps (Windows)

From PowerShell:

```powershell
cd C:\Dev\Web
git clone https://github.com/AbhiDevLab/Brain-Tumor.git
cd Brain-Tumor
git status
git log -8 --oneline

py -0p
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python --version

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m compileall -q app.py audit_dataset.py train_classifier.py train_segmenter.py evaluate_classifier.py evaluate_segmenter.py src tests
python -m pytest -q
```

Only after the environment checks pass, launch `streamlit run app.py`. Missing trained checkpoints are expected at this stage.

Then download BRISC 2025 from https://www.kaggle.com/datasets/briscdataset/brisc2025/ and audit its actual layout, license, labels, mask availability/alignment, duplicates and patient-level split integrity before adapting the repository's expected folder structure. Keep the dataset and model weights outside Git.

## PDF and speech deliverables

The updated local deliverables created on 20 October 2026 are:
- `Brain_Tumor_Flowchart_and_Presentation_Speech_UPDATED.pdf`
- `Brain_Tumor_Base_Model_and_Implementation_Roadmap_UPDATED.pdf`
- `Brain_Tumor_Research_Synthesis_and_Prototype_Strategy_UPDATED.pdf`
- `Brain_Tumor_Research_Gap_and_Proposed_Contribution_Flowchart.pdf`
- `Brain_Tumor_Research_Gap_and_Proposed_Contribution_Flowchart.png`
- `Brain_Tumor_Updated_PDFs_and_Flowchart.zip`

The updated flowchart was checked in the PDF renderings. It clearly separates established prior methods, our planned baseline, controlled experiments, evaluation criteria and the unvalidated status of the proposed contribution. A separate Markdown version of the flowchart and presentation speech is maintained in the repository's `docs/` directory.
