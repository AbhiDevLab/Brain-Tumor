# Final Review Enhancement Branch

**Status date:** 7 November 2026  
**Branch:** final-review-enhancements  
**Base:** feature/online-mri-research-platform  
**Entry point:** app.py

## Objective

Improve the existing Streamlit research prototype for the final academic project review without changing the trained model architectures, weights, or default segmentation policy.

## Implemented interface

The application has six navigation areas:

- **Overview:** model stack, workflow, and clearly labelled historical experiment snapshot.
- **MRI Analysis:** validated de-identified 2D upload, original-image preview, inference, run-time measurements, review flags, and report downloads.
- **Classification:** EfficientNet-B0 prediction and uncalibrated per-class scores.
- **Segmentation:** U-Net binary masks and threshold exploration.
- **Explainable AI:** adjustable Grad-CAM overlay intensity.
- **Research & Metrics:** BRISC split distribution, recorded aggregate metrics, BCE vs BCE + soft Dice comparison, and optional raw evaluation artifacts.

## Segmentation comparison and threshold explorer

The BCE U-Net remains the default. An optional checkbox runs the BCE + soft Dice checkpoint on the same image to show a side-by-side comparison. Both use the same display threshold selected in the interface.

The inference function retains its normalized probability grid in session memory. Changing the threshold rebuilds the binary mask and overlay from that grid; it does not run the neural network again. The probability grid is excluded from the JSON report.

A visual comparison does not establish segmentation accuracy. The uploaded image has no paired ground-truth annotation, so the application does not calculate Dice or IoU for an individual upload.

## Explainable AI

The classifier inference now retains the normalized Grad-CAM map in session memory. The interface lets the reviewer adjust overlay intensity without recomputing classification. The map is excluded from JSON export.

Grad-CAM is a classifier attribution method. It is not a U-Net mask, a causal explanation, or proof of correctness.

## PDF and JSON research reports

src/brain_tumor/reports.py generates an in-memory PDF containing:

- input dimensions and image format (not the source filename);
- classifier output, uncalibrated class scores, and checkpoint integrity metadata;
- per-model segmentation threshold and foreground pixel statistics;
- processing device and measured inference-path durations;
- preliminary review flags;
- original MRI, Grad-CAM overlay, and available segmentation masks/overlays;
- an explicit research-only and not-clinically-validated warning.

The PDF does not assert per-upload accuracy, Dice, IoU, physical tumor dimensions, or diagnostic certainty. The JSON report excludes the uploaded image and image-sized attribution/probability arrays.

The PDF generator uses Matplotlib, which is already included in requirements.txt; no additional package is required.

## Research metrics policy

The aggregate metrics shown on the Research & Metrics page are documented researcher-reported results:

| Model | Evaluation set | Reported result |
|---|---|---|
| EfficientNet-B0 | 1,000 classification test images | Accuracy 99.40%; macro-F1 0.993945; weighted-F1 0.994000 |
| BCE U-Net | 854 cleaned segmentation pairs | Mean Dice 0.799568; mean IoU 0.718709 |
| BCE + soft Dice U-Net | Same 854 pairs | Mean Dice 0.799782; mean IoU 0.716567 |

The UI labels these as reported historical results, not freshly recomputed values. It optionally reads:

- artifacts/evaluation/classification/metrics.json
- artifacts/evaluation/segmentation.json

When those actual evaluation artifacts are absent, the UI does not invent confusion-matrix, per-class, or per-image metrics.

The BRISC split chart uses the documented prepared classification counts. The app explicitly notes that exact duplicate hashes were addressed in the training-side preparation, but patient-wise independence and near-duplicate isolation remain unverified.

## Verification

From the repository root:

    python -m pip install -r requirements-dev.txt
    python -m compileall -q app.py src tests
    python -m pytest -q
    streamlit run app.py

The regression suite should be run on this branch before deployment. Existing tests that use controlled model doubles verify software contracts only; they are not MRI accuracy or clinical validation.

For the deployment smoke test:

1. Select branch final-review-enhancements and app.py in the hosting provider.
2. Configure the existing fine-grained, repository-scoped BRAIN_MRI_GITHUB_TOKEN in server-side secrets only.
3. Verify that the classifier and default BCE U-Net checkpoint load and pass SHA-256 verification.
4. Upload an approved de-identified 2D MRI and check classification, segmentation, Grad-CAM, report downloads, and logs.
5. Enable the optional BCE + soft Dice comparison and verify it loads its own pinned checkpoint.
6. Check the live app URL in a separate browser session and confirm resource use and cold-start behaviour.

Do not commit checkpoints, private tokens, MRI images, or patient information. Do not change the published release assets or trained checkpoints as part of this UI enhancement branch.
