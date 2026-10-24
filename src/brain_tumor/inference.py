from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import warnings
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.brain_tumor.data import classify_image_tensor, segmentation_image_tensor
from src.brain_tumor.explain import generate_gradcam, overlay_gradcam
from src.brain_tumor.models import UNet, build_classifier

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_FORMATS = {"PNG", "JPEG", "BMP", "TIFF"}
DEFAULT_CLASSIFIER_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "classifier.pt"
DEFAULT_SEGMENTER_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "segmenter.pt"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MIN_IMAGE_EDGE = 32
DEFAULT_MASK_THRESHOLD = 0.5
# Preliminary UI consistency rules only; not a validated reliability gate.
SUBSTANTIAL_MASK_FRACTION = 0.05
NEGLIGIBLE_MASK_FRACTION = 0.001
EXPECTED_CLASSES = {"glioma", "meningioma", "pituitary", "pituitary_tumor", "no_tumor"}


class InputValidationError(ValueError):
    """An uploaded image failed a safe, user-facing input check."""


class CheckpointError(RuntimeError):
    """A configured model artifact could not be loaded safely."""


MAX_REMOTE_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024

# Immutable IDs and integrity metadata discovered from the private draft release
# baseline-checkpoints-2026-10-21 (release ID 418777578).
GITHUB_RELEASE_REPO = "AbhiDevLab/Brain-Tumor"
GITHUB_RELEASE_ASSETS: dict[str, dict[str, Any]] = {
    "classifier": {
        "asset_id": 660318425,
        "name": "efficientnet_b0_best_20261020T192345Z.pt",
        "size_bytes": 16_364_125,
        "sha256": "c0775f865d8a4811f10b50818fb44ce62d8b08c1ecb366cf317f8242eb38f9d8",
        "architecture": "efficientnet_b0",
    },
    "segmenter": {
        "asset_id": 660464605,
        "name": "unet_bce_best_20261020T195800Z.pt",
        "size_bytes": 7_754_627,
        "sha256": "7b7cac96ceebdccf892ecfd7bfbe36a14fdda1bb5ce8899716a1db6ed2624357",
        "architecture": "unet_binary",
    },
    "segmenter_bce_dice": {
        "asset_id": 661318090,
        "name": "unet_bce_dice_best_20261020T205900Z.pt",
        "size_bytes": 7_756_333,
        "sha256": "346c701760b56d27def9d779f2d6e96f740579454fcc8a2db623ccfc28edc008",
        "architecture": "unet_binary",
    },
}
GITHUB_ASSET_REDIRECT_HOSTS = {
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}


def _allowed_github_asset_redirect(host: str) -> bool:
    """Permit GitHub's documented release CDN and its release-specific S3 host."""
    if host in GITHUB_ASSET_REDIRECT_HOSTS:
        return True
    return bool(re.fullmatch(
        r"github-production-release-asset-[a-z0-9-]+\.s3\.amazonaws\.com",
        host.lower(),
    ))


def _download_github_release_asset(
    kind: str,
    token: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Fetch one pinned private Release asset without forwarding auth on redirects.

    The API request is authenticated at api.github.com with a read-only token.
    Every redirect is followed manually; redirect hosts are allow-listed, and
    subsequent requests contain no Authorization header. Bytes are cached only
    after the asset size and pinned SHA-256 match the release metadata.
    """
    if kind not in GITHUB_RELEASE_ASSETS:
        raise ValueError("Unsupported GitHub release model kind.")
    if not token or not token.strip():
        raise CheckpointError("Private GitHub model access requires a server-side read-only token.")

    asset = GITHUB_RELEASE_ASSETS[kind]
    expected_sha256 = asset["sha256"]
    expected_size = int(asset["size_bytes"])
    asset_id = int(asset["asset_id"])
    api_url = (
        f"https://api.github.com/repos/{GITHUB_RELEASE_REPO}"
        f"/releases/assets/{asset_id}"
    )
    root = Path(cache_dir or os.environ.get(
        "BRAIN_MRI_MODEL_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "brain-mri-model-artifacts"),
    )).expanduser()

    try:
        root.mkdir(parents=True, exist_ok=True)
        try:
            root.chmod(0o700)
        except OSError:
            pass
        destination = root / asset["name"]
        if destination.is_file() and _sha256(destination) == expected_sha256:
            return destination
        if destination.exists():
            destination.unlink()

        class _RejectRedirects(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = build_opener(_RejectRedirects)
        current_url = api_url
        authenticated_request = True
        temp_path: Path | None = None
        response = None
        for _redirect_count in range(4):
            try:
                parsed = urlparse(current_url)
            except ValueError as exc:
                raise CheckpointError("The GitHub release asset URL is invalid.") from exc
            host = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment:
                raise CheckpointError("GitHub supplied an unsafe release asset URL.")
            if authenticated_request:
                if host != "api.github.com" or parsed.query:
                    raise CheckpointError("The private GitHub asset API endpoint is invalid.")
            elif not _allowed_github_asset_redirect(host):
                raise CheckpointError("GitHub supplied an unapproved release asset redirect host.")

            headers = {
                "Accept": "application/octet-stream",
                "User-Agent": "BrainTumorMRIResearch/1.0",
            }
            if authenticated_request:
                headers["Authorization"] = f"Bearer {token}"
                headers["X-GitHub-Api-Version"] = "2022-11-28"
            request = Request(current_url, headers=headers, method="GET")
            try:
                response = opener.open(request, timeout=90)
                final_url = urlparse(response.geturl())
                final_host = (final_url.hostname or "").lower()
                if final_url.scheme != "https" or final_host != host or final_url.username or final_url.password:
                    response.close()
                    raise CheckpointError("The GitHub release asset response URL did not match its validated host.")
                break
            except HTTPError as exc:
                status = int(exc.code)
                location = exc.headers.get("Location") if exc.headers else None
                exc.close()
                if status not in {301, 302, 303, 307, 308} or not location:
                    raise CheckpointError("The private GitHub release asset request failed.")
                current_url = urljoin(current_url, location)
                # All following requests are unauthenticated, including cross-host redirects.
                authenticated_request = False
        else:
            raise CheckpointError("The GitHub release asset redirected too many times.")

        try:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "application/json" in content_type:
                raise CheckpointError("GitHub returned asset metadata instead of binary checkpoint bytes.")
            advertised_size = response.headers.get("Content-Length")
            if advertised_size:
                try:
                    if int(advertised_size) != expected_size:
                        raise CheckpointError("The downloaded GitHub asset size does not match the pinned release metadata.")
                except ValueError as exc:
                    raise CheckpointError("The GitHub release asset size header is invalid.") from exc

            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{kind}-", suffix=".download",
                dir=root, delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                byte_count = 0
                digest = hashlib.sha256()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > MAX_REMOTE_ARTIFACT_BYTES or byte_count > expected_size:
                        raise CheckpointError("The private GitHub model asset exceeds its expected size.")
                    digest.update(chunk)
                    temp_file.write(chunk)

            if byte_count != expected_size:
                raise CheckpointError("The downloaded GitHub asset size does not match the pinned release metadata.")
            if digest.hexdigest() != expected_sha256:
                raise CheckpointError("The downloaded GitHub asset failed its SHA-256 integrity check.")
            try:
                temp_path.chmod(0o600)
            except OSError:
                pass
            temp_path.replace(destination)
            return destination
        finally:
            response.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    except CheckpointError:
        raise
    except (HTTPError, URLError, OSError, ValueError) as exc:
        # Never include the token, signed redirect URL, response body, or local path in an error.
        raise CheckpointError(f"The private GitHub {kind} checkpoint could not be retrieved.") from exc


def _download_private_artifact(kind: str, url: str, token: str, expected_sha256: str) -> Path:
    """Download a private checkpoint over HTTPS into an ephemeral server-side cache.

    The URL is an endpoint, not a signed URL: credentials belong in a separate
    read-only Bearer token. A trusted SHA-256 value is required before loading.
    """
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise CheckpointError("The private artifact endpoint URL is invalid.") from exc
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CheckpointError("The private artifact endpoint must be a clean HTTPS URL without embedded credentials or query parameters.")
    if not token or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256 or ""):
        raise CheckpointError("Private artifact access requires a read-only token and an expected SHA-256 checksum.")
    expected_sha256 = expected_sha256.lower()
    cache_root = Path(os.environ.get(
        "BRAIN_MRI_MODEL_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "brain-mri-model-artifacts"),
    )).expanduser()
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        try:
            cache_root.chmod(0o700)
        except OSError:
            pass
        destination = cache_root / f"{kind}-{expected_sha256[:20]}.pt"
        if destination.is_file() and _sha256(destination) == expected_sha256:
            return destination
        if destination.exists():
            destination.unlink()

        request = Request(
            url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "BrainTumorMRIResearch/1.0"},
            method="GET",
        )
        class _RejectRedirects(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener = build_opener(_RejectRedirects)
        temp_path: Path | None = None
        try:
            # Never forward the Bearer credential through an HTTP redirect.
            with opener.open(request, timeout=90) as response:
                final_url = urlparse(response.geturl())
                if final_url.scheme != "https" or final_url.username or final_url.password or final_url.query:
                    raise CheckpointError("The private artifact endpoint redirected to an unsafe URL.")
                advertised_size = response.headers.get("Content-Length")
                if advertised_size and int(advertised_size) > MAX_REMOTE_ARTIFACT_BYTES:
                    raise CheckpointError("The private model artifact exceeds the allowed size.")
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{kind}-", suffix=".download",
                    dir=cache_root, delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    byte_count = 0
                    digest = hashlib.sha256()
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        byte_count += len(chunk)
                        if byte_count > MAX_REMOTE_ARTIFACT_BYTES:
                            raise CheckpointError("The private model artifact exceeds the allowed size.")
                        digest.update(chunk)
                        temp_file.write(chunk)
            if digest.hexdigest() != expected_sha256:
                raise CheckpointError("The downloaded model artifact failed its SHA-256 integrity check.")
            try:
                temp_path.chmod(0o600)
            except OSError:
                pass
            temp_path.replace(destination)
            return destination
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    except CheckpointError:
        raise
    except (HTTPError, URLError, OSError, ValueError) as exc:
        # Do not echo endpoint, token, local path, response body or provider details.
        raise CheckpointError(f"The private {kind} checkpoint could not be retrieved.") from exc


@lru_cache(maxsize=2)
def configured_checkpoint_path(kind: str) -> Path:
    """Resolve local files, explicit private HTTPS endpoints, or the pinned GitHub Release.

    Priority: per-model local path, explicit HTTPS endpoint, authenticated GitHub
    Release asset (when BRAIN_MRI_GITHUB_TOKEN is configured), then the local
    generic path for clear missing-checkpoint handling. Restart after changing secrets.
    """
    if kind == "classifier":
        prefix, default = "BRAIN_MRI_CLASSIFIER", DEFAULT_CLASSIFIER_PATH
    elif kind == "segmenter":
        prefix, default = "BRAIN_MRI_SEGMENTER", DEFAULT_SEGMENTER_PATH
    else:
        raise ValueError("kind must be 'classifier' or 'segmenter'")
    configured = os.environ.get(f"{prefix}_CHECKPOINT")
    if configured:
        return Path(configured).expanduser()
    url = os.environ.get(f"{prefix}_URL")
    if url:
        return _download_private_artifact(
            kind,
            url,
            os.environ.get(f"{prefix}_TOKEN", ""),
            os.environ.get(f"{prefix}_SHA256", ""),
        )
    github_token = os.environ.get("BRAIN_MRI_GITHUB_TOKEN", "")
    if github_token:
        return _download_github_release_asset(kind, github_token)
    return default


def checkpoint_source_status(kind: str) -> str:
    """Describe configuration without triggering downloads or revealing paths/tokens.

    States: local_available, local_missing, private_configured, private_incomplete,
    github_configured, or unavailable.
    """
    if kind == "classifier":
        prefix, default = "BRAIN_MRI_CLASSIFIER", DEFAULT_CLASSIFIER_PATH
    elif kind == "segmenter":
        prefix, default = "BRAIN_MRI_SEGMENTER", DEFAULT_SEGMENTER_PATH
    else:
        raise ValueError("kind must be 'classifier' or 'segmenter'")
    local = os.environ.get(f"{prefix}_CHECKPOINT")
    if local:
        return "local_available" if Path(local).expanduser().is_file() else "local_missing"
    url = os.environ.get(f"{prefix}_URL")
    if url:
        token = os.environ.get(f"{prefix}_TOKEN", "")
        checksum = os.environ.get(f"{prefix}_SHA256", "")
        try:
            parsed = urlparse(url)
            url_valid = (
                parsed.scheme == "https" and bool(parsed.netloc)
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
            )
        except ValueError:
            url_valid = False
        complete = url_valid and bool(token) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", checksum))
        return "private_configured" if complete else "private_incomplete"
    if os.environ.get("BRAIN_MRI_GITHUB_TOKEN", "").strip():
        return "github_configured"
    return "local_available" if default.is_file() else "unavailable"


def validate_uploaded_image(
    file_bytes: bytes,
    filename: str,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
    min_edge: int = MIN_IMAGE_EDGE,
) -> tuple[Image.Image, dict[str, Any]]:
    """Validate extension, byte size, content format, decode and dimensions.

    The returned RGB image stays in memory. This function never writes the upload
    to disk and does not retain the supplied filename in its return metadata.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InputValidationError("Unsupported file type. Upload PNG, JPG/JPEG, BMP, or TIFF.")
    if not file_bytes:
        raise InputValidationError("The uploaded file is empty.")
    if len(file_bytes) > max_bytes:
        raise InputValidationError("The image exceeds the 12 MB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(file_bytes)) as probe:
                actual_format = (probe.format or "").upper()
                width, height = probe.size
                if actual_format not in SUPPORTED_FORMATS:
                    raise InputValidationError("The file contents are not a supported image format.")
                extension_formats = {
                    ".png": {"PNG"}, ".jpg": {"JPEG"}, ".jpeg": {"JPEG"},
                    ".bmp": {"BMP"}, ".tif": {"TIFF"}, ".tiff": {"TIFF"},
                }
                if actual_format not in extension_formats[suffix]:
                    raise InputValidationError("The file extension does not match its image content.")
                if width < min_edge or height < min_edge:
                    raise InputValidationError(f"Image dimensions must be at least {min_edge} × {min_edge} pixels.")
                if width * height > max_pixels:
                    raise InputValidationError("The image dimensions exceed the supported pixel limit.")
                # Verify decodability without relying on file headers alone.
                probe.verify()
            with Image.open(io.BytesIO(file_bytes)) as decoded:
                decoded.seek(0)  # TIFF inputs use the first 2D frame only.
                decoded.load()
                rgb = decoded.convert("RGB").copy()
    except InputValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise InputValidationError("The image is corrupted, malformed, or could not be decoded.") from exc

    array = np.asarray(rgb, dtype=np.uint8)
    info = {
        "width_px": int(width),
        "height_px": int(height),
        "channels": 3,
        "format": actual_format,
        "file_bytes": len(file_bytes),
        "mean_intensity_0_to_255": round(float(array.mean()), 2),
        "intensity_std_0_to_255": round(float(array.std()), 2),
    }
    return rgb, info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint(path: str | Path, expected_architecture: str) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise CheckpointError(f"The configured {expected_architecture} checkpoint is missing or inaccessible.")
    try:
        digest = _sha256(checkpoint_path)
        # These are the repository's own trusted training artifacts. Latest recovery
        # checkpoints contain RNG state objects, so weights_only=False is required.
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"The configured {expected_architecture} checkpoint is corrupt or unreadable.") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise CheckpointError(f"The configured {expected_architecture} checkpoint has an unsupported format.")
    declared_architecture = payload.get("architecture")
    if declared_architecture is not None and declared_architecture != expected_architecture:
        raise CheckpointError(f"The checkpoint architecture is incompatible with {expected_architecture}.")
    meta = {
        "checkpoint_name": checkpoint_path.name,
        "sha256": digest,
        "architecture": expected_architecture,
        "image_size": payload.get("image_size"),
        "checkpoint_metadata_architecture": declared_architecture,
    }
    return payload, meta


def load_classifier_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload, meta = _read_checkpoint(path, "efficientnet_b0")
    classes = payload.get("classes")
    if (
        not isinstance(classes, (list, tuple))
        or len(classes) != 4
        or not all(isinstance(name, str) and name.strip() for name in classes)
        or len({name.lower().strip() for name in classes}) != len(classes)
    ):
        raise CheckpointError("The classifier checkpoint must declare four unique class labels.")
    normalized = {name.lower().replace(" ", "_").strip() for name in classes}
    if not normalized.issubset(EXPECTED_CLASSES):
        raise CheckpointError("The classifier checkpoint contains unexpected class labels.")
    image_size = payload.get("image_size", 224)
    if not isinstance(image_size, int) or not 32 <= image_size <= 1024:
        raise CheckpointError("The classifier checkpoint declares an invalid input size.")
    try:
        model = build_classifier(num_classes=len(classes), pretrained=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise CheckpointError("The classifier checkpoint weights do not match the repository model.") from exc
    meta.update({"image_size": image_size, "classes": list(classes)})
    return model, meta


def load_segmenter_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload, meta = _read_checkpoint(path, "unet_binary")
    image_size = payload.get("image_size", 256)
    if not isinstance(image_size, int) or not 32 <= image_size <= 1024:
        raise CheckpointError("The segmenter checkpoint declares an invalid input size.")
    try:
        model = UNet()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise CheckpointError("The segmenter checkpoint weights do not match the repository model.") from exc
    meta.update({
        "image_size": image_size,
        "loss_mode": payload.get("loss_mode", "unspecified"),
        "threshold_default": DEFAULT_MASK_THRESHOLD,
    })
    return model, meta


def predict_classification(
    model: torch.nn.Module,
    image: Image.Image,
    metadata: dict[str, Any],
    device: torch.device | str,
    *,
    include_gradcam: bool = True,
) -> dict[str, Any]:
    classes = metadata["classes"]
    size = int(metadata["image_size"])
    tensor = classify_image_tensor(image, size).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape != (1, len(classes)):
            raise RuntimeError("Classifier returned an unexpected output shape.")
        scores = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    order = np.argsort(scores)[::-1]
    result: dict[str, Any] = {
        "predicted_class": classes[int(order[0])],
        "score_not_calibrated_probability": float(scores[int(order[0])]),
        "scores_not_calibrated_probabilities": {
            classes[index]: float(scores[index]) for index in range(len(classes))
        },
        "input_tensor_shape": [1, 3, size, size],
        "checkpoint_name": metadata["checkpoint_name"],
        "checkpoint_sha256": metadata["sha256"],
        "architecture": metadata["architecture"],
        "image_size": size,
    }
    if include_gradcam:
        try:
            cam = generate_gradcam(model, tensor, int(order[0]), output_size=(size, size))
            result["gradcam_overlay"] = overlay_gradcam(image, cam)
            result["gradcam_status"] = "generated"
        except Exception:
            # The classification prediction remains valid when attribution
            # visualization fails; never mislabel a segmentation mask as Grad-CAM.
            result["gradcam_overlay"] = None
            result["gradcam_status"] = "unavailable"
    return result


def predict_segmentation(
    model: torch.nn.Module,
    image: Image.Image,
    metadata: dict[str, Any],
    device: torch.device | str,
    *,
    threshold: float = DEFAULT_MASK_THRESHOLD,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Segmentation threshold must be between 0 and 1.")
    size = int(metadata["image_size"])
    tensor = segmentation_image_tensor(image, size).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 4 or logits.shape[:2] != (1, 1):
            raise RuntimeError("Segmenter returned an unexpected output shape.")
        probabilities = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    if probabilities.shape != (size, size):
        raise RuntimeError("Segmenter output dimensions do not match its configured input size.")
    native_mask = (probabilities >= threshold).astype(np.uint8)
    mask_image = Image.fromarray(native_mask * 255, mode="L")
    display_mask = mask_image.resize(image.size, Image.Resampling.NEAREST)
    mask_display_array = np.asarray(display_mask, dtype=np.uint8) > 0
    original_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = original_array.copy()
    tint = np.array([255, 48, 48], dtype=np.float32)
    overlay[mask_display_array] = (
        original_array[mask_display_array].astype(np.float32) * 0.60 + tint * 0.40
    ).astype(np.uint8)
    fraction = float(native_mask.mean())
    return {
        "mask_native": native_mask,
        "mask_display": display_mask,
        "overlay": Image.fromarray(overlay, mode="RGB"),
        "predicted_foreground_pixels": int(native_mask.sum()),
        "foreground_pixel_fraction": fraction,
        "threshold": float(threshold),
        "output_grid_size_px": [size, size],
        "display_size_px": [image.width, image.height],
        "checkpoint_name": metadata["checkpoint_name"],
        "checkpoint_sha256": metadata["sha256"],
        "architecture": metadata["architecture"],
        "loss_mode": metadata.get("loss_mode", "unspecified"),
    }


def _canonical_label(label: str) -> str:
    return label.lower().strip().replace(" ", "_").replace("-", "_")


def preliminary_consistency_flags(
    predicted_class: str | None,
    foreground_fraction: float | None,
    *,
    substantial_fraction: float = SUBSTANTIAL_MASK_FRACTION,
    negligible_fraction: float = NEGLIGIBLE_MASK_FRACTION,
) -> list[dict[str, str]]:
    """Apply simple, unvalidated consistency flags without hiding either output."""
    if predicted_class is None or foreground_fraction is None:
        return [{
            "level": "review",
            "code": "incomplete_model_outputs",
            "message": "One or more model outputs are unavailable; combined review is incomplete.",
        }]
    label = _canonical_label(predicted_class)
    no_tumor = label in {"no_tumor", "notumor", "no_tumour", "none"}
    if no_tumor and foreground_fraction >= substantial_fraction:
        return [{
            "level": "review",
            "code": "classifier_segmenter_disagreement",
            "message": "The classifier predicts no_tumor while the segmentation mask covers at least 5% of its output grid.",
        }]
    if not no_tumor and foreground_fraction <= negligible_fraction:
        return [{
            "level": "review",
            "code": "classifier_segmenter_disagreement",
            "message": "The classifier predicts a tumor category while the segmentation mask covers at most 0.1% of its output grid.",
        }]
    return [{
        "level": "info",
        "code": "preliminary_checks_no_flag",
        "message": "The preliminary size-based consistency rules did not flag this pair of outputs; this is not evidence of reliability.",
    }]


def input_quality_flags(image: Image.Image) -> list[dict[str, str]]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    flags: list[dict[str, str]] = []
    if min(image.size) < 64:
        flags.append({
            "level": "review",
            "code": "small_input_dimensions",
            "message": "The image is unusually small for this workflow; inspect input quality before interpreting model outputs.",
        })
    if float(arr.std()) < 2.0:
        flags.append({
            "level": "review",
            "code": "near_uniform_intensity",
            "message": "The image has near-uniform pixel intensity and may not be suitable for meaningful assessment.",
        })
    return flags


def build_result_report(
    *,
    input_metadata: dict[str, Any],
    classification: dict[str, Any] | None,
    segmentation: dict[str, Any] | None,
    flags: list[dict[str, str]],
    errors: list[str],
    threshold: float = DEFAULT_MASK_THRESHOLD,
) -> dict[str, Any]:
    """Build a non-identifying JSON-ready summary, excluding input filenames and pixels."""
    classification_report = None
    if classification is not None:
        classification_report = {
            key: value for key, value in classification.items()
            if key not in {"gradcam_overlay"}
        }
    segmentation_report = None
    if segmentation is not None:
        segmentation_report = {
            "predicted_foreground_pixels": segmentation["predicted_foreground_pixels"],
            "foreground_pixel_fraction": segmentation["foreground_pixel_fraction"],
            "threshold": segmentation["threshold"],
            "output_grid_size_px": segmentation["output_grid_size_px"],
            "display_size_px": segmentation["display_size_px"],
            "checkpoint_name": segmentation["checkpoint_name"],
            "checkpoint_sha256": segmentation["checkpoint_sha256"],
            "architecture": segmentation["architecture"],
            "loss_mode": segmentation["loss_mode"],
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "input": {
            "width_px": input_metadata["width_px"],
            "height_px": input_metadata["height_px"],
            "format": input_metadata["format"],
            "file_bytes": input_metadata["file_bytes"],
        },
        "classification": classification_report,
        "segmentation": segmentation_report,
        "segmentation_threshold": float(threshold),
        "review_flags": flags,
        "inference_errors": list(errors),
        "disclaimer": (
            "Research and education only. These models are not clinically validated. "
            "Predictions can be wrong and must not guide medical decisions or treatment. "
            "Have MRI studies reviewed by qualified medical professionals."
        ),
    }
