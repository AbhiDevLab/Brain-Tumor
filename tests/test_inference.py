from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import src.brain_tumor.inference as inference


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([1.0, 3.0, -1.0, 0.0]))

    def forward(self, x):
        return self.logits.unsqueeze(0).expand(x.shape[0], -1)


class TinySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        return x[:, :1] * 0 + self.logit


def png_bytes(size=(80, 64), value=128):
    image = Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8), mode="RGB")
    stream = __import__("io").BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_validate_supported_png_and_omits_filename_from_metadata():
    image, metadata = inference.validate_uploaded_image(png_bytes(), "private-name.png")
    assert image.mode == "RGB"
    assert image.size == (80, 64)
    assert metadata["format"] == "PNG"
    assert metadata["file_bytes"] > 0
    assert "filename" not in metadata


@pytest.mark.parametrize(
    ("payload", "filename", "message"),
    [
        (b"", "scan.png", "empty"),
        (b"not an image", "scan.png", "corrupted"),
        (png_bytes(), "scan.exe", "Unsupported file type"),
        (png_bytes(), "scan.jpg", "extension does not match"),
        (png_bytes(size=(20, 20)), "tiny.png", "at least 32"),
    ],
)
def test_validate_uploaded_image_rejects_unsupported_inputs(payload, filename, message):
    with pytest.raises(inference.InputValidationError, match=message):
        inference.validate_uploaded_image(payload, filename)


def test_validate_uploaded_image_enforces_size_limit():
    with pytest.raises(inference.InputValidationError, match="12 MB"):
        inference.validate_uploaded_image(png_bytes(), "scan.png", max_bytes=2)


def test_validate_uploaded_image_rejects_overlarge_dimensions():
    with pytest.raises(inference.InputValidationError, match="pixel limit"):
        inference.validate_uploaded_image(png_bytes(size=(40, 40)), "scan.png", max_pixels=1000)


def test_preprocessing_contracts_are_distinct():
    image = Image.fromarray(np.full((64, 64, 3), 64, dtype=np.uint8), mode="RGB")
    classifier_tensor = inference.classify_image_tensor(image, 32)
    segmenter_tensor = inference.segmentation_image_tensor(image, 32)
    assert tuple(classifier_tensor.shape) == (1, 3, 32, 32)
    assert tuple(segmenter_tensor.shape) == (1, 3, 32, 32)
    # Classifier input is ImageNet-normalized, while the segmenter is scaled to [0, 1].
    assert classifier_tensor.min().item() < 0
    assert segmenter_tensor.min().item() > 0
    assert segmenter_tensor.max().item() <= 1


def test_classifier_checkpoint_loads_compatible_payload_and_preserves_class_order(tmp_path, monkeypatch):
    model = TinyClassifier()
    path = tmp_path / "classifier.pt"
    classes = ["glioma", "meningioma", "no_tumor", "pituitary"]
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": 32,
        "architecture": "efficientnet_b0",
    }, path)
    monkeypatch.setattr(inference, "build_classifier", lambda num_classes, pretrained=False: TinyClassifier())
    loaded, meta = inference.load_classifier_checkpoint(path)
    assert loaded.training is False
    assert meta["classes"] == classes
    assert meta["image_size"] == 32
    assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    result = inference.predict_classification(loaded, image, meta, "cpu", include_gradcam=False)
    assert result["predicted_class"] == "meningioma"
    assert set(result["scores_not_calibrated_probabilities"]) == set(classes)
    assert "probabilities" not in result


def test_segmenter_checkpoint_loads_and_returns_binary_aligned_mask(tmp_path, monkeypatch):
    model = TinySegmenter()
    path = tmp_path / "segmenter.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "image_size": 32,
        "architecture": "unet_binary",
        "loss_mode": "bce",
    }, path)
    monkeypatch.setattr(inference, "UNet", TinySegmenter)
    loaded, meta = inference.load_segmenter_checkpoint(path)
    image = Image.fromarray(np.zeros((64, 80, 3), dtype=np.uint8), mode="RGB")
    result = inference.predict_segmentation(loaded, image, meta, "cpu", threshold=0.5)
    assert result["mask_native"].shape == (32, 32)
    assert set(np.unique(result["mask_native"])).issubset({0, 1})
    assert result["predicted_foreground_pixels"] == 32 * 32
    assert result["foreground_pixel_fraction"] == 1.0
    assert result["mask_display"].size == image.size
    assert result["overlay"].size == image.size
    assert result["loss_mode"] == "bce"


def test_missing_and_corrupt_checkpoints_fail_closed(tmp_path):
    with pytest.raises(inference.CheckpointError, match="missing or inaccessible"):
        inference.load_classifier_checkpoint(tmp_path / "missing.pt")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(inference.CheckpointError, match="corrupt or unreadable"):
        inference.load_segmenter_checkpoint(corrupt)


def test_unsupported_checkpoint_architecture_is_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"model_state_dict": {}, "architecture": "random_model"}, path)
    with pytest.raises(inference.CheckpointError, match="architecture is incompatible"):
        inference.load_segmenter_checkpoint(path)


@pytest.mark.parametrize(
    ("predicted_class", "fraction", "expected_code"),
    [
        ("no_tumor", 0.25, "classifier_segmenter_disagreement"),
        ("glioma", 0.0005, "classifier_segmenter_disagreement"),
        ("glioma", 0.20, "preliminary_checks_no_flag"),
        (None, 0.20, "incomplete_model_outputs"),
    ],
)
def test_preliminary_consistency_flags(predicted_class, fraction, expected_code):
    flags = inference.preliminary_consistency_flags(predicted_class, fraction)
    assert flags[0]["code"] == expected_code


def test_report_is_json_ready_and_does_not_include_uploaded_filename():
    image, metadata = inference.validate_uploaded_image(png_bytes(), "patient_very_private_name.png")
    flags = inference.input_quality_flags(image)
    report = inference.build_result_report(
        input_metadata=metadata,
        classification=None,
        segmentation=None,
        flags=flags,
        errors=["model unavailable"],
    )
    import json
    encoded = json.dumps(report)
    assert "patient_very_private_name" not in encoded
    assert report["classification"] is None
    assert report["segmentation"] is None
    assert report["research_only"] is True
    assert "not clinically validated" in report["disclaimer"]



def test_private_artifact_download_uses_checksum_and_reuses_verified_cache(tmp_path, monkeypatch):
    import io

    payload = b"research-checkpoint-bytes"
    expected = hashlib.sha256(payload).hexdigest()
    cache_dir = tmp_path / "private-cache"
    monkeypatch.setenv("BRAIN_MRI_MODEL_CACHE_DIR", str(cache_dir))

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def geturl(self):
            return "https://artifacts.example.org/models/classifier.pt"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    calls = {"count": 0}

    def fake_open(request, timeout):
        calls["count"] += 1
        assert request.get_header("Authorization") == "Bearer read-only-token"
        assert timeout == 90
        return FakeResponse(payload)

    monkeypatch.setattr(inference, "build_opener", lambda handler: type(
        "FakeOpener", (), {"open": staticmethod(fake_open)}
    )())
    artifact = inference._download_private_artifact(
        "classifier",
        "https://artifacts.example.org/models/classifier.pt",
        "read-only-token",
        expected,
    )
    assert artifact.read_bytes() == payload
    assert inference._download_private_artifact(
        "classifier",
        "https://artifacts.example.org/models/classifier.pt",
        "read-only-token",
        expected,
    ) == artifact
    assert calls["count"] == 1


@pytest.mark.parametrize(
    ("url", "token", "checksum"),
    [
        ("http://artifacts.example.org/model.pt", "read-only-token", "a" * 64),
        ("https://artifacts.example.org/model.pt?token=embedded", "read-only-token", "a" * 64),
        ("https://artifacts.example.org/model.pt", "", "a" * 64),
        ("https://artifacts.example.org/model.pt", "read-only-token", "not-a-checksum"),
    ],
)
def test_private_artifact_download_rejects_insecure_configuration(url, token, checksum):
    with pytest.raises(inference.CheckpointError):
        inference._download_private_artifact("classifier", url, token, checksum)


def test_private_artifact_checksum_mismatch_fails_closed(tmp_path, monkeypatch):
    import io

    monkeypatch.setenv("BRAIN_MRI_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    payload = b"not-the-expected-checkpoint"

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def geturl(self):
            return "https://artifacts.example.org/models/model.pt"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(inference, "build_opener", lambda handler: type(
        "FakeOpener", (), {"open": staticmethod(lambda request, timeout: FakeResponse(payload))}
    )())
    with pytest.raises(inference.CheckpointError, match="SHA-256 integrity check"):
        inference._download_private_artifact(
            "classifier",
            "https://artifacts.example.org/models/model.pt",
            "read-only-token",
            hashlib.sha256(b"different bytes").hexdigest(),
        )



def test_checkpoint_source_status_does_not_trigger_download_or_expose_secrets(tmp_path, monkeypatch):
    for name in (
        "BRAIN_MRI_CLASSIFIER_CHECKPOINT", "BRAIN_MRI_CLASSIFIER_URL",
        "BRAIN_MRI_CLASSIFIER_TOKEN", "BRAIN_MRI_CLASSIFIER_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(inference, "DEFAULT_CLASSIFIER_PATH", tmp_path / "not-installed.pt")
    assert inference.checkpoint_source_status("classifier") == "unavailable"

    local = tmp_path / "classifier.pt"
    monkeypatch.setenv("BRAIN_MRI_CLASSIFIER_CHECKPOINT", str(local))
    assert inference.checkpoint_source_status("classifier") == "local_missing"
    local.write_bytes(b"private")
    assert inference.checkpoint_source_status("classifier") == "local_available"

    monkeypatch.delenv("BRAIN_MRI_CLASSIFIER_CHECKPOINT")
    monkeypatch.setenv("BRAIN_MRI_CLASSIFIER_URL", "https://artifacts.example.org/models/classifier.pt")
    monkeypatch.setenv("BRAIN_MRI_CLASSIFIER_TOKEN", "do-not-display-this")
    monkeypatch.setenv("BRAIN_MRI_CLASSIFIER_SHA256", "a" * 64)
    assert inference.checkpoint_source_status("classifier") == "private_configured"
    assert "do-not-display-this" not in inference.checkpoint_source_status("classifier")

    monkeypatch.delenv("BRAIN_MRI_CLASSIFIER_TOKEN")
    assert inference.checkpoint_source_status("classifier") == "private_incomplete"
