# Brain Tumor MRI Research Prototype

A small research project for classifying 2D brain MRI images and segmenting tumor regions. The first goal is to establish reproducible baselines and verify the dataset before testing architectural changes.

## Current status

- EfficientNet-B0 classifier for four image classes: glioma, meningioma, pituitary tumor, and no tumor.
- U-Net baseline for binary tumor masks.
- Separate training and evaluation scripts.
- Dataset checks for class distribution, image duplicates across splits, and image/mask pairing.
- A BRISC 2025 preflight auditor for class counts, mask pairing, sampled mask pixel intensities, exact cross-split duplicates, and available manifest identifiers.
- A Streamlit interface scaffold for local experiments.
- `prepare_brisc.py` to remove exact train/test image-hash overlap from the training side, create a reproducible validation split, and preserve official test files.

No model has been trained on the project dataset yet. The repository does not contain trained weights or validated performance results. This is a research prototype, not a clinical diagnostic tool.

## Research documents and presentation

- [Research vs. proposed contribution flowchart](docs/research_gap_and_proposed_contribution.md) — distinguishes established methods from the baseline and experiments we plan to test.
- [Paper reproduction and claim-verification plan](docs/paper_reproduction_plan.md) — maps the previously reviewed studies to what can and cannot be reproduced, with comparison rules.
- [BRISC audit findings and split limitations](docs/brisc_audit_findings.md) — records the observed exact duplicate overlap and missing patient identifiers.
- [Presentation speech](docs/presentation_speech.md) — approx. two-minute explanation of the flowchart.
- [Consolidated project status and next steps](docs/project_status_summary.md) — current implementation, evaluation status, and Windows setup checklist.
- [Research documentation index](docs/README.md).

The proposed improvement is **not yet demonstrated**. We will only report gains if controlled, held-out evaluation supports them.

## Setup

Python 3.11 is recommended. From the repository directory:

\`\`\`powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
\`\`\`

Run the tests:

\`\`\`powershell
python -m pytest -q
\`\`\`

Start the interface:

\`\`\`powershell
streamlit run app.py
\`\`\`

## Data

The project training scripts expect two separate data layouts. Keep patients in only one split whenever patient identifiers are available.

Classification images:

\`\`\`text
data/classification/{train,val,test}/{glioma,meningioma,pituitary,no_tumor}/
\`\`\`

Segmentation images and their binary masks:

\`\`\`text
data/segmentation/images/{train,val,test}/
data/segmentation/masks/{train,val,test}/
\`\`\`

The matching image and mask files must share the same filename stem. Review the dataset contract and run the audit scripts before training. Do not use class-labelled images as if they had segmentation masks.

### Audit the raw BRISC 2025 download

After downloading and extracting BRISC 2025, run:

\`\`\`bash
python audit_brisc.py --root /path/to/brisc2025 --output artifacts/brisc_audit.json
\`\`\`

In Colab, for the current notebook path:

\`\`\`bash
python audit_brisc.py --root /content/kaggle_data/brisc2025 --output /content/brisc_audit.json
\`\`\`

The audit checks folder/class counts, image-mask filename pairing, a sample of mask grayscale values, exact-byte duplicates across the dataset's train/test splits, and whether the manifest provides patient/subject identifiers. It does **not** detect near-duplicates or prove patient-level separation when identifiers are unavailable. The mask binarization threshold defaults to 128 because reviewed examples contained near-black values (0–7) and near-white values (248–255); inspect the full audit output before locking preprocessing.

Keep datasets and trained weights outside Git.

### Audit and prepare BRISC 2025

After downloading and extracting the dataset, audit the raw files first:

```bash
python audit_brisc.py --root /path/to/brisc2025 --output /path/to/brisc_audit.json
```

Then create a separately prepared dataset. The script retains the supplied test images and masks, removes from the training side any exact image hashes also present in test, removes safe training-only exact duplicates, and takes validation samples from the remaining training set:

```bash
python prepare_brisc.py --root /path/to/brisc2025 --output /path/to/processed_brisc --val-fraction 0.15 --seed 42
```

In Colab, use:

```bash
python audit_brisc.py --root /content/kaggle_data/brisc2025 --output /content/brisc_audit.json
python prepare_brisc.py --root /content/kaggle_data/brisc2025 --output /content/processed_brisc --val-fraction 0.15 --seed 42
```

The output includes `prepared_manifest.csv` and `preparation_report.json`. Patient-level independence cannot be confirmed if the dataset provides no patient/subject IDs. Exact hashes do not detect near-duplicates.

The audit run captured on 20 October 2026 found seven groups of byte-identical images crossing train/test in each task; nine training file paths were affected because one duplicate group included three training copies. Re-audit the prepared output before training.

Sampled masks contain near-black background values (0–7) and near-white foreground values (mostly 248–255). The current loader thresholds at 128 and resizes masks using nearest-neighbor interpolation. This preprocessing choice still needs review against the full audit and visual examples; it is not a claim of complete annotation validation.

Keep raw/prepared datasets and model weights outside Git.

## Experiments

The first segmentation comparison keeps the U-Net and data split fixed while changing the loss from binary cross-entropy to binary cross-entropy plus Dice loss. Dice score is the primary overlap metric; tumor recall is tracked to guard against missed tumor pixels. More complex attention or detector models come later only if the baseline results justify them.

See \`docs/experiment_plan.md\`, \`docs/implementation_status.md\`, and \`docs/paper_reproduction_plan.md\` for the working plan and current limitations.

## Scope and limitations

The current implementation targets 2D images and binary masks. DICOM/NIfTI volume handling, multi-region segmentation, YOLO replication, transformer models, calibration, and independent external validation are not yet implemented. Grad-CAM is an attribution visualization, not a segmentation mask or proof of clinical reasoning.
