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
    assert result["probability_map"].shape == (32, 32)
    assert result["probability_map"].dtype == np.float32
    assert float(result["probability_map"].min()) >= 0.0
    assert float(result["probability_map"].max()) <= 1.0
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



def test_github_release_asset_download_follows_redirect_without_forwarding_token(tmp_path, monkeypatch):
    import io
    from email.message import Message
    from urllib.error import HTTPError

    payload = b"real-checkpoint-download-contract"
    expected = hashlib.sha256(payload).hexdigest()
    asset = {
        "asset_id": 424242,
        "name": "efficientnet_b0_best_20261020T192345Z.pt",
        "size_bytes": len(payload),
        "sha256": expected,
        "architecture": "efficientnet_b0",
    }
    monkeypatch.setitem(inference.GITHUB_RELEASE_ASSETS, "classifier", asset)
    calls = []

    class FakeResponse(io.BytesIO):
        def __init__(self, body, url):
            super().__init__(body)
            self.headers = {"Content-Length": str(len(body)), "Content-Type": "application/octet-stream"}
            self._url = url

        def geturl(self):
            return self._url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    class FakeOpener:
        def open(self, request, timeout):
            calls.append(request)
            assert timeout == 90
            if len(calls) == 1:
                assert request.full_url == "https://api.github.com/repos/AbhiDevLab/Brain-Tumor/releases/assets/424242"
                assert request.get_header("Authorization") == "Bearer read-only-secret"
                assert request.get_header("Accept") == "application/octet-stream"
                headers = Message()
                headers["Location"] = "https://release-assets.githubusercontent.com/download/checkpoint.pt?sig=signed"
                raise HTTPError(request.full_url, 302, "Found", headers, io.BytesIO())
            assert request.full_url.startswith("https://release-assets.githubusercontent.com/")
            assert not any(k.lower() == "authorization" for k in request.headers)
            return FakeResponse(payload, request.full_url)

    monkeypatch.setattr(inference, "build_opener", lambda *_args, **_kwargs: FakeOpener())
    result = inference._download_github_release_asset(
        "classifier", "read-only-secret", cache_dir=tmp_path
    )
    assert result.name == asset["name"]
    assert result.read_bytes() == payload
    assert inference._sha256(result) == expected
    assert len(calls) == 2


def test_github_release_asset_rejects_unapproved_redirect_host(tmp_path, monkeypatch):
    import io
    from email.message import Message
    from urllib.error import HTTPError

    asset = dict(inference.GITHUB_RELEASE_ASSETS["classifier"])
    monkeypatch.setitem(inference.GITHUB_RELEASE_ASSETS, "classifier", asset)
    calls = []

    class FakeOpener:
        def open(self, request, timeout):
            calls.append(request)
            headers = Message()
            headers["Location"] = "https://attacker.example.com/checkpoint.pt?token=anything"
            raise HTTPError(request.full_url, 302, "Found", headers, io.BytesIO())

    monkeypatch.setattr(inference, "build_opener", lambda *_args, **_kwargs: FakeOpener())
    with pytest.raises(inference.CheckpointError, match="unapproved release asset redirect host"):
        inference._download_github_release_asset("classifier", "read-only-secret", cache_dir=tmp_path)
    assert len(calls) == 1
    assert not (tmp_path / asset["name"]).exists()


def test_github_release_asset_checksum_mismatch_fails_closed(tmp_path, monkeypatch):
    import io
    from email.message import Message
    from urllib.error import HTTPError

    payload = b"wrong bytes"
    asset = {
        "asset_id": 989898,
        "name": "unet_bce_best_20261020T195800Z.pt",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(b"the right bytes").hexdigest(),
        "architecture": "unet_binary",
    }
    monkeypatch.setitem(inference.GITHUB_RELEASE_ASSETS, "segmenter", asset)

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(payload)), "Content-Type": "application/octet-stream"}

        def geturl(self):
            return "https://api.github.com/repos/AbhiDevLab/Brain-Tumor/releases/assets/989898"

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse(payload)

    monkeypatch.setattr(inference, "build_opener", lambda *_args, **_kwargs: FakeOpener())
    with pytest.raises(inference.CheckpointError, match="SHA-256 integrity check"):
        inference._download_github_release_asset("segmenter", "read-only-secret", cache_dir=tmp_path)
    assert not (tmp_path / asset["name"]).exists()


def test_checkpoint_source_status_supports_private_github_release(monkeypatch):
    monkeypatch.delenv("BRAIN_MRI_CLASSIFIER_CHECKPOINT", raising=False)
    monkeypatch.delenv("BRAIN_MRI_CLASSIFIER_URL", raising=False)
    monkeypatch.setenv("BRAIN_MRI_GITHUB_TOKEN", "secret-must-not-be-returned")
    assert inference.checkpoint_source_status("classifier") == "github_configured"
    assert "secret-must-not-be-returned" not in inference.checkpoint_source_status("classifier")
    monkeypatch.delenv("BRAIN_MRI_GITHUB_TOKEN")



def test_github_release_redirect_allowlist_accepts_only_known_github_asset_hosts():
    assert inference._allowed_github_asset_redirect("release-assets.githubusercontent.com")
    assert inference._allowed_github_asset_redirect("objects.githubusercontent.com")
    assert inference._allowed_github_asset_redirect("github-production-release-asset-abc123.s3.amazonaws.com")
    assert not inference._allowed_github_asset_redirect("attacker.example.com")
    assert not inference._allowed_github_asset_redirect("github.com.attacker.example.com")


def test_optional_bce_dice_release_asset_metadata_is_pinned():
    asset = inference.GITHUB_RELEASE_ASSETS["segmenter_bce_dice"]
    assert asset["asset_id"] == 661318090
    assert asset["name"] == "unet_bce_dice_best_20261020T205900Z.pt"
    assert asset["sha256"] == "346c701760b56d27def9d779f2d6e96f740579454fcc8a2db623ccfc28edc008"
    assert asset["architecture"] == "unet_binary"


def test_checkpoint_source_status_supports_comparison_model(monkeypatch, tmp_path):
    monkeypatch.delenv("BRAIN_MRI_SEGMENTER_BCE_DICE_CHECKPOINT", raising=False)
    monkeypatch.delenv("BRAIN_MRI_SEGMENTER_BCE_DICE_URL", raising=False)
    monkeypatch.delenv("BRAIN_MRI_GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(inference.Path, "is_file", lambda self: False)
    assert inference.checkpoint_source_status("segmenter_bce_dice") == "unavailable"
    monkeypatch.setenv("BRAIN_MRI_GITHUB_TOKEN", "read-only")
    assert inference.checkpoint_source_status("segmenter_bce_dice") == "github_configured"



def test_github_release_asset_403_explains_missing_draft_release_access(tmp_path, monkeypatch):
    import io
    from email.message import Message
    from urllib.error import HTTPError

    class FakeOpener:
        def open(self, request, timeout):
            headers = Message()
            raise HTTPError(request.full_url, 403, "Forbidden", headers, io.BytesIO(b"private response body"))

    monkeypatch.setattr(inference, "build_opener", lambda *_args, **_kwargs: FakeOpener())
    with pytest.raises(inference.CheckpointError, match="HTTP 403") as caught:
        inference._download_github_release_asset("classifier", "top-secret-token", cache_dir=tmp_path)
    assert "top-secret-token" not in str(caught.value)
    assert "private response body" not in str(caught.value)
