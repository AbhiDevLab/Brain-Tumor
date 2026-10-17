# Brain Tumor MRI Research Prototype

A small research project for classifying 2D brain MRI images and segmenting tumor regions. The first goal is to establish reproducible baselines and verify the dataset before testing architectural changes.

## Current status

- EfficientNet-B0 classifier for four image classes: glioma, meningioma, pituitary tumor, and no tumor.
- U-Net baseline for binary tumor masks.
- Separate training and evaluation scripts.
- Dataset checks for class distribution, image duplicates across splits, and image/mask pairing.
- A Streamlit interface scaffold for local experiments.

No model has been trained on the project dataset yet. The repository does not contain trained weights or validated performance results. This is a research prototype, not a clinical diagnostic tool.

## Setup

Python 3.11 is recommended. From the repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the tests:

```powershell
python -m pytest
```

Start the interface:

```powershell
streamlit run app.py
```

## Data

The project expects two separate data layouts. Keep patients in only one split whenever patient identifiers are available.

Classification images:

```text
data/classification/{train,val,test}/{glioma,meningioma,pituitary,no_tumor}/
```

Segmentation images and their binary masks:

```text
data/segmentation/images/{train,val,test}/
data/segmentation/masks/{train,val,test}/
```

The matching image and mask files must share the same filename stem. Review the dataset contract and run the audit scripts before training. Do not use class-labelled images as if they had segmentation masks.

## Experiments

The first segmentation comparison keeps the U-Net and data split fixed while changing the loss from binary cross-entropy to binary cross-entropy plus Dice loss. Dice score is the primary overlap metric; tumor recall is tracked to guard against missed tumor pixels. More complex attention or detector models come later only if the baseline results justify them.

See `docs/experiment_plan.md` and `docs/implementation_status.md` for the working plan and current limitations.

## Scope and limitations

The current implementation targets 2D images and binary masks. DICOM/NIfTI volume handling, multi-region segmentation, YOLO replication, transformer models, calibration, and independent external validation are not yet implemented. Grad-CAM is an attribution visualization, not a segmentation mask or proof of clinical reasoning.
