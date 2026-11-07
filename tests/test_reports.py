from __future__ import annotations

from PIL import Image

from src.brain_tumor.reports import build_pdf_report


def test_pdf_report_is_generated_and_contains_pdf_header():
    report = {
        "created_utc": "2026-11-07T00:00:00+00:00",
        "input": {"width_px": 96, "height_px": 96, "format": "PNG", "file_bytes": 500},
        "classification": {
            "predicted_class": "glioma",
            "score_not_calibrated_probability": 0.91,
            "scores_not_calibrated_probabilities": {"glioma": 0.91, "meningioma": 0.03},
            "checkpoint_name": "classifier.pt",
            "checkpoint_sha256": "a" * 64,
        },
        "segmentation_models": {
            "BCE U-Net": {
                "loss_mode": "bce",
                "threshold": 0.5,
                "foreground_pixel_fraction": 0.12,
                "predicted_foreground_pixels": 120,
                "checkpoint_name": "segmenter.pt",
            }
        },
        "processing": {
            "device": "cpu",
            "seconds": {"classification_seconds": 0.12, "bce_unet_seconds": 0.08},
            "total_seconds": 0.20,
        },
        "review_flags": [{"level": "info", "message": "Example review note"}],
    }
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    pdf = build_pdf_report(
        report=report,
        original_image=image,
        gradcam_image=image,
        segmentation_views={
            "BCE U-Net": {
                "mask_display": Image.new("L", (96, 96), 255),
                "overlay": image,
            }
        },
    )
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 2_000
