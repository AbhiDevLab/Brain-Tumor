# Research review and design decisions

**Review cut-off:** 17 October 2026. This document distinguishes the four supplied files from additional web-verified literature. Reported metrics below are paper-reported, not reproduced in this repository. They are not directly comparable unless dataset, label, split, task, and metric are identical.

## A. Supplied document: YOLOv5/YOLOv7 segmentation and classification (IEEE Access, 2024)

**Reference:** M. F. Almufareh et al., “Automated Brain Tumor Segmentation and Classification in MRI Using YOLO-Based Deep Learning,” *IEEE Access*, vol. 12, pp. 16189–16207, 2024. DOI: https://doi.org/10.1109/ACCESS.2024.3359418

### Problem and data

- Targets three classes: meningioma, glioma and pituitary tumor.
- Reports 3,064 MRI slices from 233 patients, with 512 × 512 images.
- Source annotations were MATLAB `.mat` files. The paper converts images/masks to PNG and notes tumor border coordinates were misaligned with image content.
- Its preprocessing creates a binary mask, applies morphological closing, extracts contours, approximates polygons and converts the polygon coordinates into YOLO labels.
- Its stated split is 2,144 training images (69.97%), 827 validation images (26.99%) and 91 test images (2.97%). The results section also describes evaluation on 827 images, so the exact status of reported metrics as test-set versus validation-set results should be checked before trying to reproduce them. The test split is small.

### Model and paper-reported results

- Compares YOLOv5-seg and YOLOv7-seg, both producing class/localization outputs and masks in a single-stage framework.
- Paper-reported YOLOv5: box precision 0.940, recall 0.905, mAP@0.5 0.947, mAP@0.5:0.95 0.666; mask precision 0.936, recall 0.906, mAP@0.5 0.947, mAP@0.5:0.95 0.657.
- Paper-reported YOLOv7: box precision/accuracy is described around 0.936 and recall 0.904, mAP@0.5 0.940, mAP@0.5:0.95 0.677; mask precision/accuracy is described around 0.935, recall 0.903, mAP@0.5 0.941, mAP@0.5:0.95 0.659. Exact terminology should be checked against the paper's result tables before quoting.
- The paper describes 100 epochs and learning rate 0.001 for training, and a substantial Colab/A100 GPU environment for evaluation.

### Reusable ideas

1. The quality of mask alignment can be as important as model architecture.
2. Preserve detection/localization metrics such as precision, recall and mAP in a YOLO replication track.
3. Inspect original image, raw mask, corrected mask, and contour overlay before training.

### Limitations / cautions

- The split is described at image-slice level, though the dataset contains many slices from 233 patients; the document does not establish patient-wise isolation. Patient-level leakage must be ruled out before relying on the results.
- The reported test set is only 91 images, while results prose also mentions 827-image evaluation. This split/evaluation description needs reconciliation for reproducibility.
- It is a 2D three-tumor-class dataset and should not be conflated with volumetric BraTS subregion segmentation.

## B. Supplied document: Comprehensive review (IEEE SCEECS 2025)

**Reference:** A. R. Dudhe, P. G. Burade and M. Nikose, “Comprehensive Review of Brain Tumor Segmentation and Classification Using Deep Learning Techniques,” 2025 IEEE International Students' Conference on Electrical, Electronics and Computer Science. DOI: https://doi.org/10.1109/SCEECS64059.2025.10940343

### Main ideas

- Frames the field as a progression from region-growing/thresholding to classical machine learning (e.g. SVM, random forest), then CNNs, U-Net variants, transfer learning, attention/hybrid models and transformers.
- Highlights preprocessing, feature selection, postprocessing and dataset selection as part of the method, not incidental steps.
- Contrasts segmentation metrics such as Dice/IoU with classification metrics such as sensitivity/recall, specificity, precision, F1 and AUC.
- Notes common issues: label scarcity, noise, interpretability, class imbalance, computational cost and generalization beyond the dataset.

### How to use this review

Use it as a map of method families and metrics. It is not a single controlled benchmark: its cited methods use different datasets, labels and experimental protocols. Avoid interpreting the review's tables as a fair head-to-head ranking or a pooled meta-analysis for our dataset.

## C. Supplied document: Swin-HAFNet manuscript and the BRISC dataset line

The uploaded file is formatted as a manuscript “submitted to Elsevier”; its abstract reports weighted segmentation mIoU 82.4% and classification accuracy 99.63% on BRISC. The conclusion discusses future work to extend the model to joint segmentation/classification, which conflicts with the abstract/introduction's description of a joint multi-task method. Treat the exact manuscript/version and the reported classification score cautiously until checked against the authors' latest code and arXiv version.

Related public version: A. Fateh et al., “BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification with Swin-HAFNet,” arXiv:2506.14318 (2025), https://arxiv.org/abs/2506.14318. The currently indexed arXiv version describes BRISC as 6,000 contrast-enhanced T1-weighted 2D images across glioma, meningioma, pituitary and no-tumor categories, with masks and axial/coronal/sagittal views. It describes 5,000 training and 1,000 test images. The preprint's model naming appears as Swin-HAFUNet for segmentation and it describes a multi-scale classification pipeline; therefore do not assume this matches every detail in the uploaded file.

### Architectural features reported

- Hierarchical Swin Transformer feature encoder with shifted-window attention and multi-scale representations.
- Contextual Bottleneck Enhancer (CBE): shifted MLP/token mixing along width and height, depth-wise convolution, gating/residual processing to enrich long-range directional context.
- Hierarchical Attention Fusion (HAF): attention-based fusion of encoder/decoder features with deformable-convolution branches to better align semantic and geometric detail than naive skip concatenation.
- Decoder reconstructs a high-resolution segmentation map.
- Separate multi-scale classification branch reduces/alines feature map dimensions, fuses levels and produces four-class output.
- Segmentation uses a BCE + Dice compound objective.

### Paper-reported outcomes in the uploaded version

- BRISC segmentation weighted mIoU: 82.4%; per-type reported mIoU: glioma 76.0%, meningioma 85.0%, pituitary 85.3%.
- Additional Kaggle Brain Tumor Segmentation benchmark: IoU 0.7142 and Dice 0.8294.
- Abstract reports BRISC classification accuracy 99.63%; compare this only after verifying the exact data split/protocol for that manuscript version.
- Ablation table reports baseline weighted mIoU 79.8%, with HAF 81.1%, CBE 81.3%, both 82.4%. This is the strongest reusable research lesson: run ablations and demonstrate each module's contribution rather than adding complex components all at once.

### Limitations / cautions

The uploaded manuscript acknowledges small/irregular lesions, acquisition-protocol changes, class imbalance and transformer computational cost. The public arXiv dataset paper positions itself primarily as an introduction to a dataset/benchmark; its text calls for further benchmarking and clinically informed evaluation. Verify dataset release, annotation policy, license and official split before use.

## D. Supplied document: September 2026 project synopsis

This is the project proposal rather than a results paper. It proposes: accepting MRI/CT uploads; resizing, intensity normalization, denoising, optional skull stripping and augmentation; a VGG/ResNet transfer-learning classifier; U-Net mask segmentation; Grad-CAM explanation; metrics including accuracy, precision, recall, F1, ROC-AUC, Dice and IoU; and a Flask/Streamlit interface with reports/history.

It proposes a 70/15/15 split and >90% classification accuracy as an expected target, not as a verified result. We should not hard-code a >90% target as success before checking patient-wise test performance. The synopsis itself correctly flags dataset bias, limited 2D analysis and lack of clinical validation. Its flowchart on page 4 depicts upload → preprocessing → feature extraction/classification → segmentation → result/report/history → radiologist review.

### Decisions for the first prototype

- First deliverable is the smallest reproducible baseline, not a paper reproduction: EfficientNet-B0 classification + compact U-Net binary segmentation with separate pipelines and metrics.
- The app accepts common 2D image files first; it does not imply support for DICOM/NIfTI until a dedicated medical-imaging pipeline is implemented.
- Save no patient history by default; output a downloadable JSON with model outputs and a clear research disclaimer.
- Do not run inference without trained checkpoints.

## E. Additional literature checked through 17 October 2026

### 1. 2026 systematic review: generalization and evaluation gaps

“Systematic Review for Detecting & Diagnosing Brain Tumor Using Deep Learning: Current Trends and Future Research Directions,” *International Journal of Computational Intelligence Systems* (published 26 March 2026), DOI: https://doi.org/10.1007/s44196-026-01283-2.

The review reports recurring benchmark-comparability problems due to heterogeneous labels and limited multicentre testing, as well as incomplete uncertainty/calibration reporting and domain shift across sites/scanners. It also emphasizes the instability of preprocessing effects and the importance of checking explanations across perturbations. This strengthens the decision to measure external generalization and uncertainty later, instead of treating a single benchmark score as clinical readiness.

### 2. 2026 multimodal feature-fusion/Transformer paper

M. Pajany et al., “Multimodal deep feature fusion with transformer for brain tumor classification from magnetic resonance imaging,” *Scientific Reports* 16, 17041 (published 11 April 2026). DOI: https://doi.org/10.1038/s41598-026-44957-9.

Its pipeline uses CLAHE and normalization, fuses features from CapsNet/ResNet50/AlexNet, uses a Transformer/BiConvLSTM-related classification block, and uses nnU-Net for segmentation. The authors report 98.91% accuracy on the selected MRI dataset. It is useful as a future model-comparison candidate, not as our starting architecture: it is substantially more complex, and the result is dataset/protocol specific.

### 3. 2026 contrastive learning and adaptive feature fusion

Y. Peng, S. He and L. Chang, “A contrastive learning framework with adaptive feature fusion for brain tumor classification,” *Scientific Reports* (published 22 March 2026), DOI: https://doi.org/10.1038/s41598-026-44887-6.

This paper explores adaptive feature fusion and contrastive learning and reports 99.35% accuracy. Importantly, its limitations section notes random slice-level splitting on a 233-patient Figshare dataset may put slices from the same patient into train and test and thereby inflate results; it calls for patient-isolated validation. This is direct support for prioritizing split integrity over headline accuracy.

### 4. 2026 transformer/MRI segmentation review

C. Jin et al., “Transformer-based architectures in MRI brain tumor segmentation: A review,” *Computerized Medical Imaging and Graphics* 129 (2026), article 102729; DOI: https://doi.org/10.1016/j.compmedimag.2026.102729. Its existence confirms transformer segmentation remains an active research direction. The broader newer literature also examines handling missing MRI modalities, domain-shift robustness and deployment; those are later-stage goals once our simple baseline and data contracts are solid.

### 5. BRISC reference and data access

BRISC arXiv paper: https://arxiv.org/abs/2506.14318. Dataset repository: https://www.kaggle.com/datasets/briscdataset/brisc2025/ . The public arXiv manuscript (v3 indexed online) supplies a 6,000-image dataset description and associated masks/code. Confirm actual available files, data license, train/test manifests and current version at download time.

### 6. Official BraTS ecosystem

Official challenge repository: https://github.com/brats . The BraTS 2025 challenge expands beyond classic segmentation to longitudinal tumor response, generalizability, rare tumor entities and missing MRI modality synthesis/inpainting. For this project's second research stage, BraTS is better suited to a true volumetric/multi-modal segmentation track than a 2D image-folder baseline.

## F. Synthesis: the research gap worth addressing

Many published models advertise very high accuracy, but the most reusable gap for this student prototype is **reliable evaluation across patients and datasets with interpretable, spatially checked outputs**. A defensible contribution later could be a controlled comparison of (i) transfer-learning classification, (ii) U-Net/nnU-Net segmentation, and (iii) YOLO or Swin-inspired joint learning, with identical patient-wise splits, ablations, error analysis, and external evaluation.

A novel architecture is not necessary for the first milestone. Reproducibility, clean labels, fair comparison, and honest failure reporting are prerequisites for a credible improvement claim.

## Initial implementation decision

The first runnable implementation uses EfficientNet-B0 for four-class classification and a compact U-Net for binary tumor/background segmentation. The first planned segmentation ablation is BCE-only versus BCE + soft-Dice loss with the same data split and architecture. These are implementation baselines; no performance claim is made until they are trained and evaluated on audited data.
