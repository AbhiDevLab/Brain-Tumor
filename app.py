from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.brain_tumor.inference import (
    DEFAULT_MASK_THRESHOLD, CheckpointError, InputValidationError,
    build_result_report, checkpoint_source_status, configured_checkpoint_path,
    input_quality_flags, load_classifier_checkpoint, load_segmenter_checkpoint,
    predict_classification, predict_segmentation, preliminary_consistency_flags,
    validate_uploaded_image,
)
from src.brain_tumor.reports import build_pdf_report


st.set_page_config(
    page_title="Brain MRI Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1420px; padding-top: 1.25rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(120,140,160,.20);}
    .hero {padding: 1.6rem 1.85rem; border-radius: 18px; background: linear-gradient(120deg,#102a43 0%,#176b87 100%); color: #fff; margin-bottom: 1rem;}
    .hero h1 {color:#fff; font-size:2.25rem; line-height:1.16; margin:0 0 .55rem 0;}
    .hero p {color:#e3f2f7; max-width:850px; margin:0; font-size:1.02rem;}
    .eyebrow {letter-spacing:.12em; text-transform:uppercase; font-size:.74rem; font-weight:700; opacity:.88; margin-bottom:.65rem;}
    .warning-note {border:1px solid #dfb35b; border-left:5px solid #d39b2a; padding:.9rem 1rem; border-radius:9px; background:rgba(211,155,42,.08); margin:.8rem 0 1.15rem 0;}
    .section-kicker {letter-spacing:.08em; text-transform:uppercase; font-size:.72rem; font-weight:700; color:#527086; margin-bottom:.2rem;}
    div[data-testid="stMetric"] {border:1px solid rgba(120,140,160,.24); border-radius:12px; padding:.75rem .95rem;}
    </style>
    <div class="hero">
      <div class="eyebrow">Research demonstration · 2D MRI workflow</div>
      <h1>Brain MRI Research Platform</h1>
      <p>Four-class MRI classification, binary tumor-region segmentation, and classifier attribution using the project's saved research checkpoints.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="warning-note"><b>Research use only — not a medical device.</b> '
    'The models have not been clinically validated. Their outputs can be wrong and must not be used for diagnosis, treatment, triage, or medical decisions. '
    'Upload only de-identified 2D images. Real MRI studies must be reviewed by qualified medical professionals.</div>',
    unsafe_allow_html=True,
)

_SECRET_KEYS = (
    "BRAIN_MRI_CLASSIFIER_CHECKPOINT", "BRAIN_MRI_CLASSIFIER_URL", "BRAIN_MRI_CLASSIFIER_TOKEN",
    "BRAIN_MRI_CLASSIFIER_SHA256", "BRAIN_MRI_SEGMENTER_CHECKPOINT", "BRAIN_MRI_SEGMENTER_URL",
    "BRAIN_MRI_SEGMENTER_TOKEN", "BRAIN_MRI_SEGMENTER_SHA256",
    "BRAIN_MRI_SEGMENTER_BCE_DICE_CHECKPOINT", "BRAIN_MRI_SEGMENTER_BCE_DICE_URL",
    "BRAIN_MRI_SEGMENTER_BCE_DICE_TOKEN", "BRAIN_MRI_SEGMENTER_BCE_DICE_SHA256",
    "BRAIN_MRI_MODEL_CACHE_DIR", "BRAIN_MRI_GITHUB_TOKEN",
)
for _secret_key in _SECRET_KEYS:
    if not os.environ.get(_secret_key):
        try:
            _secret_value = st.secrets.get(_secret_key)
        except Exception:
            _secret_value = None
        if _secret_value:
            os.environ[_secret_key] = str(_secret_value)

DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_NAME)
CHECKPOINT_STATUS_LABELS = {
    "local_available": "Server-side checkpoint available",
    "local_missing": "Configured local checkpoint unavailable",
    "private_configured": "Private artifact configured; retrieval starts during analysis",
    "private_incomplete": "Private artifact settings incomplete or invalid",
    "github_configured": "Private GitHub Release configured; retrieval starts during analysis",
    "unavailable": "Checkpoint unavailable",
}

# Counts from the prepared BRISC 2025 split recorded in the project documentation.
CLASSIFICATION_SPLITS = pd.DataFrame(
    {
        "Training": [974, 1113, 1224, 899],
        "Validation": [173, 197, 216, 159],
        "Testing": [254, 306, 300, 140],
    },
    index=["Glioma", "Meningioma", "Pituitary", "No tumor"],
)
CLASSIFICATION_SPLITS.index.name = "Class"
REPORTED_RESULTS = pd.DataFrame(
    [
        {"Model": "EfficientNet-B0", "Metric": "Accuracy", "Value": 0.994000},
        {"Model": "EfficientNet-B0", "Metric": "Macro-F1", "Value": 0.993945},
        {"Model": "EfficientNet-B0", "Metric": "Weighted-F1", "Value": 0.994000},
        {"Model": "BCE U-Net", "Metric": "Mean Dice", "Value": 0.799568},
        {"Model": "BCE U-Net", "Metric": "Mean IoU", "Value": 0.718709},
        {"Model": "BCE + soft Dice U-Net", "Metric": "Mean Dice", "Value": 0.799782},
        {"Model": "BCE + soft Dice U-Net", "Metric": "Mean IoU", "Value": 0.716567},
    ]
)


@st.cache_resource(show_spinner=False)
def cached_classifier(path: str, mtime_ns: int, size_bytes: int, device_label: str):
    return load_classifier_checkpoint(path, torch.device(device_label))


@st.cache_resource(show_spinner=False)
def cached_segmenter(path: str, mtime_ns: int, size_bytes: int, device_label: str):
    return load_segmenter_checkpoint(path, torch.device(device_label))


def segmentation_view(image: Image.Image, result: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Rebuild the binary display mask from stored probabilities without another model pass."""
    probabilities = np.asarray(result["probability_map"], dtype=np.float32)
    native_mask = (probabilities >= float(threshold)).astype(np.uint8)
    mask_image = Image.fromarray(native_mask * 255, mode="L")
    display_mask = mask_image.resize(image.size, Image.Resampling.NEAREST)
    mask_array = np.asarray(display_mask, dtype=np.uint8) > 0
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = original.copy()
    tint = np.array([255, 48, 48], dtype=np.float32)
    overlay[mask_array] = (
        original[mask_array].astype(np.float32) * 0.60 + tint * 0.40
    ).astype(np.uint8)
    return {
        "mask_display": display_mask,
        "overlay": Image.fromarray(overlay, mode="RGB"),
        "predicted_foreground_pixels": int(native_mask.sum()),
        "foreground_pixel_fraction": float(native_mask.mean()),
        "threshold": float(threshold),
    }


def gradcam_overlay(image: Image.Image, cam: np.ndarray, alpha: float) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = np.asarray(cam, dtype=np.float32)
    if heat.shape != base.shape[:2]:
        heat_image = Image.fromarray(
            np.uint8(np.clip(heat, 0.0, 1.0) * 255), mode="L"
        ).resize(image.size, Image.Resampling.BILINEAR)
        heat = np.asarray(heat_image, dtype=np.float32) / 255.0
    strength = np.clip(heat[..., None], 0.0, 1.0)
    red = np.zeros_like(base)
    red[..., 0] = 255
    merged = np.clip(base * (1.0 - alpha * strength) + red * alpha * strength, 0, 255).astype(np.uint8)
    return Image.fromarray(merged, mode="RGB")


def export_report(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    baseline = result["segmentations"].get("BCE U-Net")
    baseline_view = segmentation_view(result["image"], baseline, threshold) if baseline else None
    flags = list(result["input_flags"])
    if baseline_view is not None:
        flags.extend(preliminary_consistency_flags(
            result["classification"].get("predicted_class") if result["classification"] else None,
            baseline_view["foreground_pixel_fraction"],
        ))
    else:
        flags.extend(preliminary_consistency_flags(None, None))

    # build_result_report selects approved metadata fields and excludes image-sized arrays.
    report = build_result_report(
        input_metadata=result["input_metadata"],
        classification=result["classification"],
        segmentation=baseline,
        flags=flags,
        errors=result["errors"],
        threshold=float(threshold),
    )
    if baseline_view and report.get("segmentation"):
        report["segmentation"].update({
            "threshold": float(threshold),
            "predicted_foreground_pixels": baseline_view["predicted_foreground_pixels"],
            "foreground_pixel_fraction": baseline_view["foreground_pixel_fraction"],
        })
    summaries = {}
    for name, model_result in result["segmentations"].items():
        view = segmentation_view(result["image"], model_result, threshold)
        summaries[name] = {
            "checkpoint_name": model_result["checkpoint_name"],
            "checkpoint_sha256": model_result["checkpoint_sha256"],
            "loss_mode": model_result["loss_mode"],
            "threshold": float(threshold),
            "predicted_foreground_pixels": view["predicted_foreground_pixels"],
            "foreground_pixel_fraction": view["foreground_pixel_fraction"],
            "output_grid_size_px": model_result["output_grid_size_px"],
        }
    report["segmentation_models"] = summaries
    report["processing"] = {
        "device": result["device"],
        "seconds": result["timings"],
        "total_seconds": round(sum(result["timings"].values()), 4),
    }
    report["review_flags"] = flags
    report["disclaimer"] += (
        " Per-upload masks are predictions without ground-truth labels; no Dice or IoU "
        "accuracy score is calculated for this uploaded image."
    )
    return report


def current_result() -> dict[str, Any] | None:
    return st.session_state.get("analysis_result")


def show_model_availability() -> None:
    status = {
        "classifier": checkpoint_source_status("classifier"),
        "segmenter": checkpoint_source_status("segmenter"),
        "segmenter_bce_dice": checkpoint_source_status("segmenter_bce_dice"),
    }
    with st.expander("Model availability and runtime", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**EfficientNet-B0 classifier**")
            st.write(CHECKPOINT_STATUS_LABELS[status["classifier"]])
            st.caption("Glioma · Meningioma · Pituitary · No tumor")
            st.markdown("**BCE U-Net baseline**")
            st.write(CHECKPOINT_STATUS_LABELS[status["segmenter"]])
            st.caption("Default segmentation model · threshold 0.50")
        with right:
            st.markdown("**BCE + soft Dice comparison**")
            st.write(CHECKPOINT_STATUS_LABELS[status["segmenter_bce_dice"]])
            st.caption("Optional research comparison; not established as superior.")
            st.markdown("**Inference device**")
            st.write("CUDA GPU" if DEVICE_NAME == "cuda" else "CPU")
        st.caption("Private paths, tokens, and artifact URLs are not displayed.")


def show_overview() -> None:
    st.markdown('<div class="section-kicker">Workspace overview</div>', unsafe_allow_html=True)
    st.header("Research workspace")
    st.write("A model-aware interface for exploring the project's trained classification and segmentation baselines.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classifier", "EfficientNet-B0")
    c2.metric("Segmenter", "U-Net · BCE default")
    c3.metric("Explainability", "Grad-CAM")
    c4.metric("Classification test set", "1,000 images")
    st.subheader("Analysis workflow")
    steps = st.columns(3)
    steps[0].markdown("**01 · Inspect**")
    steps[0].caption("Upload a de-identified 2D MRI and review format, dimensions, and intensity.")
    steps[1].markdown("**02 · Analyze**")
    steps[1].caption("Run EfficientNet-B0 and BCE U-Net; optionally compare BCE + Dice.")
    steps[2].markdown("**03 · Review and export**")
    steps[2].caption("Explore masks, threshold effects, classifier attribution, runtime, and PDF reports.")
    st.subheader("Reported experiment snapshot")
    a, b, c = st.columns(3)
    a.metric("Classifier accuracy", "99.40%")
    b.metric("BCE U-Net mean Dice", "0.799568")
    c.metric("BCE + Dice mean Dice", "0.799782")
    st.caption(
        "Historical researcher-reported results documented in the project README. "
        "They are not recomputed during this session or guarantees of performance."
    )
    st.info("Open **MRI Analysis** to upload an image. Supported inputs are de-identified 2D PNG, JPG/JPEG, BMP, or TIFF/TIF.")
    show_model_availability()


def show_mri_analysis() -> None:
    st.markdown('<div class="section-kicker">Image-level inference</div>', unsafe_allow_html=True)
    st.header("MRI analysis")
    show_model_availability()
    uploaded = st.file_uploader(
        "Upload a de-identified 2D brain MRI image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Maximum 12 MiB. Do not upload names, reports, or identifying information.",
        key="mri_upload",
    )
    if uploaded is None:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("upload_key", None)
        st.info("Upload an image to preview it and validate its dimensions before inference.")
        return
    file_bytes = uploaded.getvalue()
    try:
        original, metadata = validate_uploaded_image(file_bytes, uploaded.name)
    except InputValidationError as exc:
        st.error(str(exc))
        return

    upload_key = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("analysis_result", None)
        st.session_state["input_image"] = original
        st.session_state["input_metadata"] = metadata
    else:
        original = st.session_state.get("input_image", original)
        metadata = st.session_state.get("input_metadata", metadata)

    preview, details = st.columns([1.2, 1])
    with preview:
        st.markdown("**Original image preview**")
        st.image(original, use_container_width=True)
    with details:
        st.markdown("**Input validation**")
        x1, x2 = st.columns(2)
        x1.metric("Width", f'{metadata["width_px"]:,} px')
        x2.metric("Height", f'{metadata["height_px"]:,} px')
        st.write(f'**Format:** {metadata["format"]}')
        st.write(f'**File size:** {metadata["file_bytes"] / (1024 * 1024):.2f} MiB')
        st.write(f'**Intensity variation:** {metadata["intensity_std_0_to_255"]:.2f} / 255')
        st.caption("Image processing remains in session memory; the application does not create a permanent image history.")

    compare_enabled = st.checkbox(
        "Run BCE U-Net vs BCE + soft Dice side-by-side",
        value=False,
        help="Runs the optional comparison checkpoint on the same image. It does not replace the BCE default.",
    )
    threshold = st.slider(
        "Segmentation threshold explorer",
        min_value=0.05, max_value=0.95,
        value=float(st.session_state.get("mask_threshold", DEFAULT_MASK_THRESHOLD)),
        step=0.05, key="mask_threshold",
        help="Rebuilds the binary mask from retained probabilities without rerunning the neural network.",
    )
    st.caption("The threshold is an exploratory display setting, not a calibrated clinical cutoff. Default: 0.50.")

    if st.button("Run MRI analysis", type="primary", use_container_width=True):
        classification = None
        segmentations: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        timings = {"classification_seconds": 0.0, "bce_unet_seconds": 0.0, "bce_dice_seconds": 0.0}
        progress = st.progress(0, text="Preparing model inference…")

        try:
            started = time.perf_counter()
            path = configured_checkpoint_path("classifier")
            if not path.is_file():
                errors.append("Classifier checkpoint unavailable. Configure its local path or the private GitHub Release token on the server.")
            else:
                stat = path.stat()
                model, model_meta = cached_classifier(str(path), stat.st_mtime_ns, stat.st_size, DEVICE_NAME)
                classification = predict_classification(model, original, model_meta, DEVICE)
                timings["classification_seconds"] = round(time.perf_counter() - started, 4)
        except CheckpointError:
            errors.append("The private classifier checkpoint could not be retrieved or loaded. Check server-side artifact configuration.")
        except Exception:
            errors.append("Classifier inference failed. No classification output is available.")
        progress.progress(35, text="Classifier stage completed")

        jobs = [("BCE U-Net", "segmenter", "bce_unet_seconds")]
        if compare_enabled:
            jobs.append(("BCE + soft Dice U-Net", "segmenter_bce_dice", "bce_dice_seconds"))
        for index, (label, kind, timing_key) in enumerate(jobs):
            try:
                started = time.perf_counter()
                path = configured_checkpoint_path(kind)
                if not path.is_file():
                    errors.append(f"{label} checkpoint unavailable. Configure its server-side checkpoint or private artifact access.")
                else:
                    stat = path.stat()
                    model, model_meta = cached_segmenter(str(path), stat.st_mtime_ns, stat.st_size, DEVICE_NAME)
                    segmentations[label] = predict_segmentation(
                        model, original, model_meta, DEVICE, threshold=DEFAULT_MASK_THRESHOLD
                    )
                    timings[timing_key] = round(time.perf_counter() - started, 4)
            except CheckpointError:
                errors.append(f"The {label} checkpoint could not be retrieved or loaded.")
            except Exception:
                errors.append(f"{label} inference failed. No mask is available.")
            progress.progress(65 + int(30 * (index + 1) / len(jobs)), text=f"{label} stage completed")
        progress.progress(100, text="Analysis complete")
        st.session_state["analysis_result"] = {
            "classification": classification,
            "segmentations": segmentations,
            "errors": errors,
            "input_flags": input_quality_flags(original),
            "timings": timings,
            "device": DEVICE_NAME,
            "input_metadata": metadata,
            "image": original,
        }

    result = current_result()
    if result is None:
        st.info("Run analysis to generate outputs. The trained checkpoints remain private and are not stored with the uploaded image.")
        return
    show_analysis_results(result, threshold)


def show_analysis_results(result: dict[str, Any], threshold: float) -> None:
    st.divider()
    st.subheader("Analysis summary")
    for message in result["errors"]:
        st.error(message)
    if result["classification"] is None and not result["segmentations"]:
        st.warning("No model output was generated. Check the server-side checkpoint configuration.")
        return

    timings = result["timings"]
    total_seconds = sum(timings.values())
    s1, s2, s3 = st.columns(3)
    s1.metric("Measured inference time", f"{total_seconds:.3f} s")
    s2.metric("Device", "CUDA GPU" if result["device"] == "cuda" else "CPU")
    s3.metric("Models executed", str(int(result["classification"] is not None) + len(result["segmentations"])))
    st.caption("Runtime includes preprocessing and classifier Grad-CAM generation. Timings vary by host, hardware, and cache state.")

    prediction = result["classification"]
    if prediction:
        st.markdown('<div class="section-kicker">EfficientNet-B0</div>', unsafe_allow_html=True)
        st.subheader("Classification")
        left, right = st.columns([0.8, 1.4])
        with left:
            st.metric("Predicted category", prediction["predicted_class"].replace("_", " ").title())
            st.metric("Top uncalibrated score", f'{prediction["score_not_calibrated_probability"]:.4f}')
            st.caption(f'Checkpoint: {prediction["checkpoint_name"]} · SHA-256: {prediction["checkpoint_sha256"][:12]}…')
        with right:
            frame = pd.DataFrame.from_dict(
                prediction["scores_not_calibrated_probabilities"], orient="index", columns=["Model score"]
            )
            frame.index.name = "Class"
            st.bar_chart(frame)
            st.caption("Softmax scores are uncalibrated and are not validated probabilities or clinical certainty.")

    st.markdown('<div class="section-kicker">U-Net</div>', unsafe_allow_html=True)
    st.subheader("Segmentation")
    if not result["segmentations"]:
        st.warning("No segmentation model output is available.")
    elif len(result["segmentations"]) == 2:
        columns = st.columns(2)
        for column, (name, model_result) in zip(columns, result["segmentations"].items()):
            view = segmentation_view(result["image"], model_result, threshold)
            with column:
                st.markdown(f"**{name}**")
                st.caption(
                    f'Loss: {model_result["loss_mode"]} · foreground: {view["foreground_pixel_fraction"] * 100:.2f}% · '
                    f'{view["predicted_foreground_pixels"]:,} pixels'
                )
                image_columns = st.columns(2)
                image_columns[0].image(view["mask_display"], caption="Binary mask", use_container_width=True)
                image_columns[1].image(view["overlay"], caption="Overlay", use_container_width=True)
                st.caption(f'Checkpoint SHA-256: {model_result["checkpoint_sha256"][:12]}…')
        st.info(
            "Both checkpoints processed the same image at the same display threshold. "
            "Visual similarity or difference does not establish accuracy without a ground-truth mask."
        )
    else:
        name, model_result = next(iter(result["segmentations"].items()))
        view = segmentation_view(result["image"], model_result, threshold)
        metrics = st.columns(3)
        metrics[0].metric("Foreground pixels", f'{view["predicted_foreground_pixels"]:,}')
        metrics[1].metric("Foreground fraction", f'{view["foreground_pixel_fraction"] * 100:.2f}%')
        metrics[2].metric("Threshold", f"{threshold:.2f}")
        image_columns = st.columns(3)
        image_columns[0].image(result["image"], caption="Original", use_container_width=True)
        image_columns[1].image(view["mask_display"], caption="Predicted mask", use_container_width=True)
        image_columns[2].image(view["overlay"], caption="Mask overlay", use_container_width=True)
        st.caption(f'{name} · Loss: {model_result["loss_mode"]} · Checkpoint SHA-256: {model_result["checkpoint_sha256"][:12]}…')
    st.caption(
        "These are model-generated foreground proposals, not confirmed tumor boundaries. "
        "Pixel counts are image-grid statistics, not physical tumor measurements."
    )

    st.markdown('<div class="section-kicker">Explainable AI</div>', unsafe_allow_html=True)
    st.subheader("Grad-CAM classifier attribution")
    if prediction and prediction.get("gradcam_map") is not None:
        alpha = st.slider("Grad-CAM overlay intensity", 0.10, 0.90, 0.45, 0.05, key="gradcam_alpha")
        overlay = gradcam_overlay(result["image"], prediction["gradcam_map"], alpha)
        g1, g2 = st.columns(2)
        g1.image(result["image"], caption="Original MRI", use_container_width=True)
        g2.image(overlay, caption="Interactive Grad-CAM overlay", use_container_width=True)
        st.caption("Grad-CAM shows regions influencing the selected classifier output. It is not a segmentation mask or proof of correctness.")
    elif prediction:
        st.info("Grad-CAM was unavailable for this run.")
    else:
        st.info("Grad-CAM requires a successful classification result.")

    baseline = result["segmentations"].get("BCE U-Net")
    flags = list(result["input_flags"])
    if baseline:
        baseline_view = segmentation_view(result["image"], baseline, threshold)
        flags.extend(preliminary_consistency_flags(
            prediction.get("predicted_class") if prediction else None,
            baseline_view["foreground_pixel_fraction"],
        ))
    else:
        flags.extend(preliminary_consistency_flags(None, None))
    st.subheader("Preliminary consistency review")
    st.caption("Unvalidated size-based checks only. They are not a reliability gate and do not suppress either model output.")
    for flag in flags:
        if flag["level"] == "review":
            st.warning(flag["message"])
        else:
            st.info(flag["message"])

    report = export_report(result, threshold)
    pdf = build_pdf_report(
        report=report,
        original_image=result["image"],
        gradcam_image=(
            gradcam_overlay(result["image"], prediction["gradcam_map"], 0.45)
            if prediction and prediction.get("gradcam_map") is not None else None
        ),
        segmentation_views={
            name: segmentation_view(result["image"], model_result, threshold)
            for name, model_result in result["segmentations"].items()
        },
    )
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download PDF research report", data=pdf,
        file_name="brain_mri_research_report.pdf", mime="application/pdf",
        use_container_width=True,
    )
    d2.download_button(
        "Download JSON result report", data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="brain_mri_research_result.json", mime="application/json",
        use_container_width=True,
    )


def show_classification_page(result: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-kicker">Classifier details</div>', unsafe_allow_html=True)
    st.header("Classification")
    if result is None or result["classification"] is None:
        st.info("Run an MRI analysis first. The latest classification will appear here.")
        return
    prediction = result["classification"]
    st.metric("EfficientNet-B0 prediction", prediction["predicted_class"].replace("_", " ").title())
    frame = pd.DataFrame.from_dict(
        prediction["scores_not_calibrated_probabilities"], orient="index", columns=["Uncalibrated score"]
    )
    frame.index.name = "Class"
    st.bar_chart(frame)
    st.dataframe(frame.style.format("{:.5f}"), use_container_width=True)
    st.caption(f'Input tensor: {prediction["input_tensor_shape"]} · Checkpoint: {prediction["checkpoint_name"]}')
    st.warning("Softmax values are not calibrated probabilities and do not represent clinical certainty.")


def show_segmentation_page(result: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-kicker">Binary foreground masks</div>', unsafe_allow_html=True)
    st.header("Segmentation")
    if result is None or not result["segmentations"]:
        st.info("Run an MRI analysis first. Segmentation masks will appear here.")
        return
    threshold = st.slider("Threshold explorer", 0.05, 0.95,
                          float(st.session_state.get("mask_threshold", DEFAULT_MASK_THRESHOLD)),
                          0.05, key="segmentation_page_threshold")
    st.caption("Masks are rebuilt from stored probabilities; the neural network is not rerun.")
    for name, model_result in result["segmentations"].items():
        view = segmentation_view(result["image"], model_result, threshold)
        st.markdown(f"**{name}**")
        st.metric("Foreground fraction", f'{view["foreground_pixel_fraction"] * 100:.2f}%')
        columns = st.columns(2)
        columns[0].image(view["mask_display"], caption="Mask", use_container_width=True)
        columns[1].image(view["overlay"], caption="Overlay", use_container_width=True)
        st.caption(f'Threshold {threshold:.2f} · {view["predicted_foreground_pixels"]:,} pixels · loss {model_result["loss_mode"]}')


def show_explainability_page(result: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-kicker">Classifier attribution</div>', unsafe_allow_html=True)
    st.header("Explainable AI")
    if result is None or result["classification"] is None:
        st.info("Run an MRI analysis first to generate Grad-CAM.")
        return
    prediction = result["classification"]
    if prediction.get("gradcam_map") is None:
        st.info("Grad-CAM is unavailable for the latest analysis.")
        return
    alpha = st.slider("Attribution overlay intensity", 0.10, 0.90, 0.45, 0.05)
    overlay = gradcam_overlay(result["image"], prediction["gradcam_map"], alpha)
    columns = st.columns(2)
    columns[0].image(result["image"], caption="Original MRI", use_container_width=True)
    columns[1].image(overlay, caption="Grad-CAM overlay", use_container_width=True)
    st.caption(
        f'Attribution for classifier output: {prediction["predicted_class"]}. '
        "This is not a tumor mask or a validated medical explanation."
    )


def show_research_page() -> None:
    st.markdown('<div class="section-kicker">Dataset and experiment evidence</div>', unsafe_allow_html=True)
    st.header("Research & metrics")
    st.subheader("BRISC 2025 classification split distribution")
    st.bar_chart(CLASSIFICATION_SPLITS)
    totals = CLASSIFICATION_SPLITS.copy()
    totals.loc["Total"] = totals.sum(axis=0)
    totals["Total"] = totals.sum(axis=1)
    st.dataframe(totals, use_container_width=True)
    st.caption("Prepared split counts recorded for this project. Exact-duplicate contamination was addressed on the training side; patient-wise independence is unverified.")

    st.subheader("Recorded model results")
    st.dataframe(REPORTED_RESULTS.style.format({"Value": "{:.6f}"}), use_container_width=True, hide_index=True)
    st.caption("Researcher-reported results: 1,000 classification test images and 854 cleaned segmentation evaluation pairs.")
    comparison = pd.DataFrame(
        [
            {"Mean Dice": 0.799568, "Mean IoU": 0.718709},
            {"Mean Dice": 0.799782, "Mean IoU": 0.716567},
        ],
        index=["BCE U-Net", "BCE + soft Dice U-Net"],
    )
    st.markdown("**Segmentation loss comparison**")
    st.bar_chart(comparison)
    st.info("Paired bootstrap did not establish a convincing overall BCE + soft Dice improvement. BCE U-Net remains the default.")

    st.subheader("Optional saved evaluation artifacts")
    classifier_path = Path("artifacts/evaluation/classification/metrics.json")
    segmenter_path = Path("artifacts/evaluation/segmentation.json")
    any_loaded = False
    if classifier_path.is_file():
        try:
            metrics = json.loads(classifier_path.read_text(encoding="utf-8"))
            matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            classes = metrics["classes"]
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] == len(classes):
                st.markdown("**Saved classifier confusion matrix**")
                st.dataframe(pd.DataFrame(matrix, index=classes, columns=classes), use_container_width=True)
                report = metrics.get("classification_report", {})
                rows = [{"Class": label, **report[label]} for label in classes if isinstance(report.get(label), dict)]
                if rows:
                    st.markdown("**Saved per-class precision, recall and F1**")
                    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)
                any_loaded = True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            st.warning("Classifier metrics JSON could not be parsed; no substitute values were generated.")
    if segmenter_path.is_file():
        try:
            metrics = json.loads(segmenter_path.read_text(encoding="utf-8"))
            keys = (
                "n_test_images", "mean_dice_all_images", "mean_iou_all_images",
                "tumor_present_image_count", "mean_dice_tumor_present_only",
                "mean_iou_tumor_present_only", "empty_ground_truth_mask_count",
                "empty_mask_false_positive_rate",
            )
            st.markdown("**Saved segmentation evaluation summary**")
            st.json({key: metrics.get(key) for key in keys})
            any_loaded = True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            st.warning("Segmentation metrics JSON could not be parsed; no substitute values were generated.")
    if not any_loaded:
        st.caption("Raw confusion-matrix and per-image evaluation files are not in this deployment. Only documented aggregate results are shown; no per-class or per-image charts are invented.")
    st.subheader("Dataset limitation")
    st.write(
        "The audit found exact duplicate image hashes crossing the original train/test boundary. Training-side duplicates were removed for the prepared split, "
        "but explicit patient/subject identifiers were absent. Patient-wise independence and near-duplicate isolation remain unverified."
    )


NAV_ITEMS = ["Overview", "MRI Analysis", "Classification", "Segmentation", "Explainable AI", "Research & Metrics"]
with st.sidebar:
    st.title("🧠 Brain MRI Lab")
    st.caption("Research platform · 2D workflow")
    active_page = st.radio("Workspace", NAV_ITEMS, label_visibility="collapsed")
    st.divider()
    st.markdown("**Model stack**")
    st.caption("EfficientNet-B0 · U-Net · Grad-CAM")
    st.markdown("**Baseline policy**")
    st.caption("BCE U-Net default; BCE + Dice is comparison only.")
    st.divider()
    st.caption("Not clinically validated. No medical decisions.")

latest = current_result()
if active_page == "Overview":
    show_overview()
elif active_page == "MRI Analysis":
    show_mri_analysis()
elif active_page == "Classification":
    show_classification_page(latest)
elif active_page == "Segmentation":
    show_segmentation_page(latest)
elif active_page == "Explainable AI":
    show_explainability_page(latest)
elif active_page == "Research & Metrics":
    show_research_page()

st.divider()
st.caption("Brain MRI Research Platform · Research and education only · Not clinically validated · Do not use for medical decisions.")
