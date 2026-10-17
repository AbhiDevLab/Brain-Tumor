# Implementation status

## In the initial scaffold

- EfficientNet-B0 classifier with a separate training and test-evaluation path.
- Compact U-Net for binary tumor/background masks.
- BCE-only and BCE + soft-Dice segmentation loss options.
- Dataset audit for class counts, exact duplicate images across splits, and image/mask pairing.
- Grad-CAM overlay, prediction overlays, JSON reporting, and a local Streamlit interface scaffold.
- Research notes and a staged experiment plan.

## Not yet done or validated

- No dataset or trained checkpoint is included.
- No real-data training results have been recorded.
- No YOLOv5/v7 or Swin-HAFNet result has been reproduced.
- DICOM/NIfTI input, 3D volumes, multi-region tumor masks, calibration, and external validation are out of scope for the current baseline.
- The prototype has not been clinically or regulatorily validated.

## Next step

Confirm the current BRISC package, license, annotations, and official splits; make patient/source-aware manifests where identifiers allow it; inspect image-mask alignment; and complete the data audit before training.
