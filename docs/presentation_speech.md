# Presentation Speech: Brain Tumor MRI Research Flowchart

**Approximate duration:** 2 minutes

Good morning. This flowchart separates established research, the baseline we will implement, and the improvements we plan to test. The key point is that we will build on published methods and then measure whether our changes actually help.

The established research includes transfer-learning CNNs for classifying MRI images, U-Net for pixel-level tumor segmentation, YOLO for localization and mask prediction, and attention or transformer models for contextual feature fusion. These are prior techniques, so we are not presenting them as inventions of our team.

Our Version 1 will start with audited two-dimensional, contrast-enhanced T1 MRI images. After label and mask checks, duplicate auditing, consistent resizing and normalization, EfficientNet-B0 will classify supported images as glioma, meningioma, pituitary tumor or no tumor. A separate U-Net will predict a binary tumor mask, and it will be trained only on examples with valid expert-annotated masks.

Our primary improvement target is the Dice score, which measures the overlap between our predicted tumor mask and the expert mask. Tumor sensitivity or recall is a critical guardrail because we do not want a better average overlap to hide more missed tumor regions. We will also report IoU for segmentation and macro-F1 plus per-class recall for classification. We will inspect false-positive masks, especially on no-tumor scans, and track inference time.

The experiments will proceed one change at a time. First, establish the standard U-Net with binary cross-entropy loss. Second, keep the architecture and split the same but test a combined binary cross-entropy and Dice loss. Third, add attention-gated skip connections if boundary errors justify it. We will benchmark YOLO separately, and only later consider a context-enhancement module or shared multi-task model. Every candidate uses the same audited evaluation protocol.

We will retain an improvement only if its Dice gain is repeatable, sensitivity does not fall unacceptably, false positives are examined, and the extra computation is justified. The proposed contribution at this stage is a reproducible, reliability-aware evaluation workflow and a tested improvement hypothesis, not a claim that a new architecture or clinical accuracy has already been achieved. Thank you.

## Quick explanation: what exists vs. what is ours?

- **Already exists in prior work:** CNN transfer learning, U-Net segmentation, YOLO detection/masks, transformer attention and multi-task learning.
- **What we will implement first:** a data-audited EfficientNet-B0 classifier plus a separate U-Net segmenter, with independent metrics.
- **What we will test later:** controlled loss comparisons, attention-gated feature fusion, a YOLO benchmark, and potentially context enhancement or a shared multi-task model.
- **Proposed contribution:** a reproducible and sensitivity-aware experimental comparison. It is not yet demonstrated; real-data training and held-out metrics remain pending.

## Suggested speaker emphasis

Avoid saying that our method is already more accurate. Explain that the project contribution will be established by controlled, reproducible evidence if the experiments confirm the hypothesis.
