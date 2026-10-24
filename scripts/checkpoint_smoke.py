"""Load the exact pinned private GitHub Release checkpoints and smoke-test inference.

This CI check uses a deterministic synthetic software-test image solely to prove
that the real saved weights load and emit contract-valid outputs. It is not an
MRI, a performance evaluation, or evidence of medical-image accuracy.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# Make package imports work when invoked as `python scripts/checkpoint_smoke.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.brain_tumor.inference import (
    GITHUB_RELEASE_ASSETS,
    _sha256,
    build_result_report,
    configured_checkpoint_path,
    input_quality_flags,
    load_classifier_checkpoint,
    load_segmenter_checkpoint,
    predict_classification,
    predict_segmentation,
    preliminary_consistency_flags,
    validate_uploaded_image,
)


def make_software_smoke_input() -> tuple[Image.Image, dict, bytes]:
    """Create a reproducible non-medical RGB pattern in memory."""
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    red = (x * 3 + y) % 256
    green = (y * 5 + 32) % 256
    blue = ((x // 2 + y // 3) * 7) % 256
    image = Image.fromarray(np.stack([red, green, blue], axis=-1).astype(np.uint8), mode="RGB")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    decoded, metadata = validate_uploaded_image(payload, "software-smoke-input.png")
    return decoded, metadata, payload


def main() -> None:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")

    classifier_path = configured_checkpoint_path("classifier")
    segmenter_path = configured_checkpoint_path("segmenter")
    classifier_expected = GITHUB_RELEASE_ASSETS["classifier"]["sha256"]
    segmenter_expected = GITHUB_RELEASE_ASSETS["segmenter"]["sha256"]

    assert _sha256(classifier_path) == classifier_expected, "Classifier checkpoint SHA-256 mismatch"
    assert _sha256(segmenter_path) == segmenter_expected, "BCE U-Net checkpoint SHA-256 mismatch"

    classifier, classifier_meta = load_classifier_checkpoint(classifier_path, device)
    segmenter, segmenter_meta = load_segmenter_checkpoint(segmenter_path, device)
    comparison_path = configured_checkpoint_path("segmenter_bce_dice")
    comparison_expected = GITHUB_RELEASE_ASSETS["segmenter_bce_dice"]["sha256"]
    assert _sha256(comparison_path) == comparison_expected, "BCE + Dice checkpoint SHA-256 mismatch"
    comparison_model, comparison_meta = load_segmenter_checkpoint(comparison_path, device)

    assert classifier_meta["sha256"] == classifier_expected
    assert segmenter_meta["sha256"] == segmenter_expected
    assert comparison_meta["sha256"] == comparison_expected
    assert classifier_meta["checkpoint_name"] == GITHUB_RELEASE_ASSETS["classifier"]["name"]
    assert segmenter_meta["checkpoint_name"] == GITHUB_RELEASE_ASSETS["segmenter"]["name"]
    assert comparison_meta["checkpoint_name"] == GITHUB_RELEASE_ASSETS["segmenter_bce_dice"]["name"]
    assert len(classifier_meta["classes"]) == 4
    assert segmenter_meta["loss_mode"] == "bce"
    assert comparison_meta["loss_mode"] == "bce_dice"

    image, input_metadata, _payload = make_software_smoke_input()
    classification = predict_classification(
        classifier, image, classifier_meta, device, include_gradcam=True
    )
    assert classification["predicted_class"] in classifier_meta["classes"]
    scores = classification["scores_not_calibrated_probabilities"]
    assert list(scores) == classifier_meta["classes"]
    assert len(scores) == 4 and all(np.isfinite(v) for v in scores.values())
    assert abs(sum(scores.values()) - 1.0) < 1e-5
    assert classification["gradcam_status"] == "generated"
    assert classification["gradcam_overlay"].size == image.size

    segmentation = predict_segmentation(
        segmenter, image, segmenter_meta, device, threshold=0.5
    )
    assert set(np.unique(segmentation["mask_native"])).issubset({0, 1})
    assert segmentation["mask_display"].size == image.size
    assert segmentation["overlay"].size == image.size
    assert segmentation["checkpoint_sha256"] == segmenter_expected
    assert segmentation["loss_mode"] == "bce"

    comparison_segmentation = predict_segmentation(
        comparison_model, image, comparison_meta, device, threshold=0.5
    )
    assert comparison_segmentation["checkpoint_sha256"] == comparison_expected
    assert set(np.unique(comparison_segmentation["mask_native"])).issubset({0, 1})
    assert comparison_segmentation["mask_display"].size == image.size
    assert comparison_segmentation["overlay"].size == image.size
    assert comparison_segmentation["loss_mode"] == "bce_dice"

    flags = input_quality_flags(image)
    flags.extend(preliminary_consistency_flags(
        classification["predicted_class"], segmentation["foreground_pixel_fraction"]
    ))
    report = build_result_report(
        input_metadata=input_metadata,
        classification=classification,
        segmentation=segmentation,
        flags=flags,
        errors=[],
        threshold=0.5,
    )
    # The JSON report is the same serialization contract used by the dashboard.
    serialized = json.dumps(report, allow_nan=False)
    parsed = json.loads(serialized)
    assert parsed["classification"]["checkpoint_sha256"] == classifier_expected
    assert parsed["segmentation"]["checkpoint_sha256"] == segmenter_expected
    assert parsed["segmentation"]["threshold"] == 0.5
    assert parsed["research_only"] is True
    assert "not clinically validated" in parsed["disclaimer"].lower()

    print("REAL_CHECKPOINT_SMOKE=PASS")
    print(f"CLASSIFIER_SHA256={classifier_expected}")
    print(f"BCE_UNET_SHA256={segmenter_expected}")
    print(f"BCE_DICE_UNET_SHA256={comparison_expected}")
    print("CLASSIFIER_OUTPUT=valid_four_class_scores_and_gradcam")
    print("SEGMENTATION_OUTPUT=valid_binary_mask_and_aligned_overlay")
    print("COMPARISON_OUTPUT=valid_optional_bce_dice_mask_and_overlay")
    print("REPORT_OUTPUT=json_serialization_passed")
    print("INPUT_KIND=synthetic_software_test_pattern_not_an_MRI")
    print("MEDICAL_PERFORMANCE_CLAIM=none")


if __name__ == "__main__":
    main()
