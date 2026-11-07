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
from src.brain_tumor.education import (
    BRAIN_LANDMARKS, CARE_QUESTIONS, EDUCATION_SOURCES, MRI_PATTERN_GUIDE,
    MRI_SEQUENCE_GUIDE, PATIENT_SUPPORT_TIPS, TUMOUR_OVERVIEWS, URGENT_WARNING_SIGNS,
)


st.set_page_config(
    page_title="Brain MRI Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink-0: #07111d;
      --ink-1: #0b1726;
      --ink-2: #101f32;
      --ink-3: #172a40;
      --line: rgba(137, 178, 207, .17);
      --line-strong: rgba(85, 211, 230, .34);
      --text-main: #edf6ff;
      --text-soft: #a9bfd1;
      --text-dim: #7690a7;
      --cyan: #4bd8e8;
      --cyan-soft: #9cecf3;
      --blue: #589cff;
      --amber: #f2c36c;
    }
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
      background: #07111d !important;
      color: var(--text-main) !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(ellipse at 80% 0%, rgba(25, 116, 153, .14), transparent 34%),
        radial-gradient(ellipse at 7% 44%, rgba(29, 86, 132, .10), transparent 32%),
        linear-gradient(180deg, #081321 0%, #07111d 62%, #091523 100%) !important;
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 29, .78) !important; }
    [data-testid="stToolbar"] { background: transparent !important; }
    .block-container {max-width: 1480px; padding: 4.25rem 2.3rem 3.6rem !important; }
    [data-testid="stMainBlockContainer"] { color: var(--text-main); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0b1929 0%, #091522 100%) !important;
      border-right: 1px solid var(--line) !important;
    }
    [data-testid="stSidebar"] > div:first-child { background: transparent !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
      color: var(--text-soft) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
      color: #f2f8ff !important; letter-spacing: -.02em;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stText"], label, .stCaption, [data-testid="stCaptionContainer"] {
      color: var(--text-soft);
    }
    h1, h2, h3, h4 { color: var(--text-main) !important; letter-spacing: -.035em; }
    h1 {font-size: 2rem !important; font-weight: 720 !important;}
    h2 {font-size: 1.48rem !important; font-weight: 680 !important; margin-top: .35rem !important;}
    h3 {font-size: 1.08rem !important; font-weight: 650 !important;}
    strong, b {color: #f4f9ff;}
    a {color: var(--cyan) !important;}
    hr {border-color: var(--line) !important; margin: 1.5rem 0 !important;}

    .hero-shell {
      position: relative; isolation: isolate; overflow: hidden;
      padding: 1.65rem 1.9rem 1.25rem; margin: .1rem 0 1.15rem;
      border: 1px solid rgba(111, 214, 232, .27); border-radius: 22px;
      background:
        radial-gradient(circle at 84% 24%, rgba(73, 220, 238, .19), transparent 24%),
        linear-gradient(118deg, rgba(13, 34, 55, .98) 0%, rgba(12, 57, 79, .96) 54%, rgba(12, 73, 91, .92) 100%);
      box-shadow: 0 18px 52px rgba(0, 0, 0, .24), inset 0 1px 0 rgba(255,255,255,.05);
    }
    .hero-shell:before {
      content: ""; position: absolute; z-index: -1; inset: 0; opacity: .16;
      background-image: linear-gradient(rgba(153,229,239,.16) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(153,229,239,.16) 1px, transparent 1px);
      background-size: 34px 34px; mask-image: linear-gradient(90deg, black, transparent 80%);
    }
    .hero-topline {display:flex; align-items:center; gap:.65rem; color:#a6dce7; font-size:.68rem; font-weight:750; letter-spacing:.16em; text-transform:uppercase;}
    .pulse-dot {width:7px; height:7px; border-radius:50%; background:#57e3d1; box-shadow:0 0 0 4px rgba(87,227,209,.13), 0 0 14px rgba(87,227,209,.7);}
    .hero-grid {display:grid; grid-template-columns:minmax(0,1fr) 190px; gap:1.4rem; align-items:center; margin-top:1.15rem;}
    .hero-shell h1 {color:#f4fbff !important; font-size:2.35rem !important; line-height:1.05 !important; margin:0 0 .7rem !important; letter-spacing:-.055em !important;}
    .hero-shell h1 span {color:#76e3ee;}
    .hero-shell p {color:#c0d9e7 !important; max-width:760px; margin:0; font-size:.98rem; line-height:1.65;}
    .hero-emblem {
      width:160px; height:126px; justify-self:end; border:1px solid rgba(123,226,239,.25);
      border-radius:20px; display:flex; flex-direction:column; justify-content:center; align-items:center;
      background:linear-gradient(145deg,rgba(81,205,226,.12),rgba(11,29,49,.28));
      box-shadow:inset 0 0 35px rgba(76,213,232,.07),0 12px 30px rgba(0,0,0,.16);
    }
    .emblem-core {font-size:2.25rem; line-height:1; color:#79e6f0; text-shadow:0 0 24px rgba(82,224,239,.38);}
    .emblem-caption {font-size:.62rem; letter-spacing:.17em; color:#a6dce7; font-weight:750; margin-top:.65rem;}
    .hero-bottom {display:flex; gap:1.5rem; flex-wrap:wrap; margin-top:1.35rem; padding-top:.9rem; border-top:1px solid rgba(153,220,232,.16);}
    .hero-chip {font-size:.72rem; color:#b4d4e2; letter-spacing:.02em;}
    .hero-chip b {color:#e8fbff; font-weight:700; margin-right:.3rem;}
    .research-warning {
      display:flex; gap:.8rem; align-items:flex-start; padding:.85rem 1rem; margin:0 0 1.55rem;
      border:1px solid rgba(242,195,108,.28); border-left:3px solid var(--amber); border-radius:12px;
      background:linear-gradient(100deg,rgba(91,67,31,.20),rgba(28,34,40,.56));
      color:#e8d9b8; font-size:.82rem; line-height:1.55;
    }
    .warning-icon {color:var(--amber); font-weight:900; font-size:1rem; line-height:1.2;}
    .section-kicker {letter-spacing:.17em; text-transform:uppercase; font-size:.65rem; font-weight:800; color:#65ccdc !important; margin-bottom:.28rem;}
    div[data-testid="stMetric"] {
      background:linear-gradient(145deg,rgba(19,38,58,.88),rgba(12,27,43,.9)) !important;
      border:1px solid var(--line) !important; border-radius:15px !important;
      padding:.9rem 1rem !important; box-shadow:0 8px 22px rgba(0,0,0,.13);
    }
    [data-testid="stMetricLabel"] {color:#9ab5ca !important; font-size:.78rem !important;}
    [data-testid="stMetricValue"] {color:#f2fbff !important; font-weight:720 !important; letter-spacing:-.035em;}
    [data-testid="stMetricDelta"] {font-size:.72rem !important;}
    [data-testid="stExpander"] {
      background:rgba(15,31,49,.78) !important; border:1px solid var(--line) !important;
      border-radius:13px !important; overflow:hidden;
    }
    [data-testid="stExpander"] summary {color:#c8dbea !important; font-weight:620;}
    [data-testid="stFileUploader"] section {
      background:linear-gradient(145deg,rgba(16,36,56,.8),rgba(10,24,39,.8)) !important;
      border:1px dashed rgba(85,211,230,.42) !important; border-radius:15px !important;
    }
    [data-testid="stFileUploader"] section:hover {border-color:var(--cyan) !important; background:rgba(28,67,84,.38) !important;}
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span {color:#a9c4d7 !important;}
    [data-testid="stImage"] img {border-radius:13px; border:1px solid var(--line);}
    [data-testid="stAlert"] {border-radius:12px !important; border:1px solid var(--line) !important;}
    [data-testid="stDataFrame"], [data-testid="stTable"] {
      background:rgba(13,29,46,.86) !important; border:1px solid var(--line) !important; border-radius:13px; overflow:hidden;
    }
    [data-testid="stChart"] {background:rgba(13,29,46,.46); border:1px solid var(--line); border-radius:14px; padding:.45rem;}
    [data-testid="stProgress"] > div > div {background:linear-gradient(90deg,#2db9d0,#76e5ed) !important;}
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {background:#71e2ed !important; border-color:#71e2ed !important;}
    [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
      background:#0b1b2c !important; color:#eff8ff !important; border-color:var(--line-strong) !important;
    }
    [data-baseweb="select"] > div {background:#0d2033 !important; border-color:var(--line) !important; color:#eef8ff !important;}
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
      border-radius:10px !important; font-weight:680 !important; transition:all .18s ease;
      border:1px solid rgba(90,213,230,.28) !important;
    }
    [data-testid="stButton"] button[kind="primary"], [data-testid="stDownloadButton"] button {
      background:linear-gradient(105deg,#167c98,#1ba6b8) !important; color:#f5feff !important;
      box-shadow:0 7px 18px rgba(23,157,180,.18);
    }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
      transform:translateY(-1px); border-color:rgba(104,229,240,.75) !important;
      box-shadow:0 9px 22px rgba(23,157,180,.23);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
      padding:.48rem .65rem !important; border-radius:9px; margin:.12rem 0;
      border:1px solid transparent; transition:all .15s ease;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:rgba(66,167,194,.10); border-color:var(--line);}
    [data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
      background:linear-gradient(100deg,rgba(48,177,200,.20),rgba(42,105,138,.13));
      border-color:rgba(79,207,225,.27);
    }
    .sidebar-brand {display:flex; align-items:center; gap:.65rem; margin-bottom:.1rem;}
    .sidebar-mark {
      width:35px; height:35px; display:flex; align-items:center; justify-content:center; border-radius:11px;
      background:linear-gradient(145deg,#1e8aa4,#164b6c); border:1px solid rgba(113,225,239,.35);
      box-shadow:0 6px 18px rgba(14,135,164,.20); font-size:1.05rem;
    }
    .sidebar-overline {font-size:.58rem; color:#67d3e1; font-weight:800; letter-spacing:.16em; text-transform:uppercase;}
    .sidebar-name {font-size:1.02rem; font-weight:760; color:#eef9ff; letter-spacing:-.03em;}
    .sidebar-label {font-size:.62rem; text-transform:uppercase; letter-spacing:.15em; font-weight:800; color:#68cddd; margin:.5rem 0 .35rem;}
    .footer-note {color:#7893aa; font-size:.72rem; letter-spacing:.02em; padding:.2rem 0;}
    .education-card {
      min-height: 205px; height: 100%; padding: 1rem 1.05rem; border-radius: 15px;
      background: linear-gradient(145deg,rgba(19,42,62,.94),rgba(11,27,43,.94));
      border: 1px solid var(--line); box-shadow: 0 8px 22px rgba(0,0,0,.13);
      color: var(--text-soft); line-height: 1.55;
    }
    .education-card-caution {border-color:rgba(242,195,108,.35); background:linear-gradient(145deg,rgba(65,49,28,.42),rgba(23,30,39,.9));}
    .education-card-kicker {font-size:.63rem; color:#69d5e2; letter-spacing:.13em; font-weight:800; text-transform:uppercase; margin-bottom:.6rem;}
    .brain-map-wrap {
      padding:1rem; border:1px solid var(--line); border-radius:17px;
      background:radial-gradient(ellipse at 50% 32%,rgba(35,99,128,.26),transparent 60%),rgba(9,24,39,.85);
      overflow:hidden;
    }
    .brain-map-label {font-size:.65rem; color:#70dce8; font-weight:800; letter-spacing:.14em; margin:.15rem .2rem .5rem;}
    .brain-map-wrap svg {display:block; width:100%; max-height:340px;}
    .brain-map-foot {font-size:.68rem; color:#8ba9bd; border-top:1px solid var(--line); padding:.65rem .2rem .1rem;}
    @media (max-width: 760px) {
      .block-container {padding:4rem 1rem 2.5rem !important;}
      .hero-grid {grid-template-columns:1fr;}
      .hero-emblem {display:none;}
      .hero-shell h1 {font-size:1.85rem !important;}
      .hero-bottom {gap:.7rem;}
    }
    </style>
    <div class="hero-shell">
      <div class="hero-topline"><span class="pulse-dot"></span> Neuro-imaging research environment <span style="opacity:.45">/</span> 2D MRI pipeline</div>
      <div class="hero-grid">
        <div>
          <h1>Brain MRI <span>Research Lab</span></h1>
          <p>Integrated experimental workspace for four-class MRI classification, tumor-region segmentation, and explainable model attribution.</p>
        </div>
        <div class="hero-emblem">
          <div class="emblem-core">◎</div>
          <div class="emblem-caption">NEURO · AI SYSTEM</div>
        </div>
      </div>
      <div class="hero-bottom">
        <div class="hero-chip"><b>01</b> EfficientNet-B0 classification</div>
        <div class="hero-chip"><b>02</b> U-Net segmentation</div>
        <div class="hero-chip"><b>03</b> Grad-CAM attribution</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="research-warning"><span class="warning-icon">!</span><span><b>Research use only — not a medical device.</b> '
    'These models are not clinically validated. Outputs can be wrong and must not guide diagnosis, treatment, triage, or medical decisions. '
    'Use de-identified 2D images only; clinical MRI studies require review by qualified medical professionals.</span></div>',
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
        "Upload a 2D brain MRI image",
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



def show_mri_pattern_explorer(result: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-kicker">Plain-language imaging guide</div>', unsafe_allow_html=True)
    st.header("MRI Pattern Explorer")
    st.write(
        "Explore what common MRI terms describe, why radiologists consider them, and what they cannot prove. "
        "This is a learning guide, not an automated reading of your scan."
    )
    st.warning(
        "A bright or dark area is not a diagnosis. The same appearance can have different causes, "
        "and interpretation requires the complete MRI study, sequence labels, clinical history, and a radiologist."
    )

    left, right = st.columns([1.05, 1])
    with left:
        selected_pattern = st.selectbox(
            "Choose an imaging pattern",
            list(MRI_PATTERN_GUIDE),
            key="education_mri_pattern",
        )
    with right:
        selected_sequence = st.selectbox(
            "Choose the sequence context",
            list(MRI_SEQUENCE_GUIDE),
            key="education_mri_sequence",
        )

    pattern = MRI_PATTERN_GUIDE[selected_pattern]
    sequence = MRI_SEQUENCE_GUIDE[selected_sequence]
    st.markdown(f"### {selected_pattern}")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("01 · WHAT YOU SEE")
            st.write(pattern["what"])
    with c2:
        with st.container(border=True):
            st.caption("02 · WHY IT MATTERS")
            st.write(pattern["why"])
    with c3:
        with st.container(border=True):
            st.caption("03 · WHAT IT DOES NOT PROVE")
            st.write(pattern["not_proof"])

    with st.container(border=True):
        st.markdown(f"**Sequence context: {selected_sequence}**")
        st.write(sequence["plain"])
        st.caption(sequence["look_for"])
        st.info(pattern["context"])

    st.subheader("A guide to the main MRI sequences")
    sequence_columns = st.columns(3)
    for index, (name, details) in enumerate(MRI_SEQUENCE_GUIDE.items()):
        with sequence_columns[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.write(details["plain"])
                st.caption(details["look_for"])

    st.subheader("Your latest uploaded image")
    if result is not None and result.get("image") is not None:
        image_col, note_col = st.columns([1, 1.25])
        with image_col:
            st.image(result["image"], caption="Latest uploaded 2D image", use_container_width=True)
        with note_col:
            st.info(
                "The image shown here is not automatically assessed for the pattern selected above. "
                "The current classifier predicts one of four dataset classes and the U-Net proposes a foreground mask; "
                "neither system identifies or confirms these radiology signs."
            )
            st.caption("A single exported image may omit sequence type, slice position, orientation, scale, and other DICOM metadata.")
    else:
        st.info("Run an MRI analysis first to see the latest uploaded image here. The pattern guide is available without an upload.")

    st.markdown(
        "For help understanding the wording of a real report, see "
        "[How to read your brain MRI report — RadiologyInfo](https://www.radiologyinfo.org/en/info/article-brain-mri-report)."
    )


def show_brain_landmarks_page() -> None:
    st.markdown('<div class="section-kicker">Anatomy orientation guide</div>', unsafe_allow_html=True)
    st.header("Brain Anatomy & Landmarks")
    st.write(
        "Learn where major brain structures sit and what they generally do. "
        "The diagram is a simplified side-view teaching schematic, not a patient-specific MRI locator."
    )
    orientation = st.selectbox(
        "MRI viewing plane",
        ["Axial", "Coronal", "Sagittal"],
        key="education_mri_plane",
    )
    orientation_help = {
        "Axial": "Horizontal slices divide the head into upper and lower portions. Scroll through the series to move from the top of the head toward the skull base.",
        "Coronal": "Front-facing slices divide the head into front and back portions. A series moves from the face toward the back of the head.",
        "Sagittal": "Side-facing slices divide the head into left and right portions. A midline slice can show structures such as the corpus callosum, ventricles, brainstem, and pituitary region.",
    }
    st.info(orientation_help[orientation])
    st.caption("Always use the R/L markers and orientation information shown by the radiology viewer. Display conventions can vary.")

    diagram = """
    <div class="brain-map-wrap">
      <div class="brain-map-label">LATERAL VIEW · CONCEPTUAL ONLY · NOT TO SCALE</div>
      <svg viewBox="0 0 720 330" role="img" aria-label="Simplified side-view brain schematic with major anatomical regions">
        <path d="M112 174 C84 143 99 99 139 83 C153 45 201 31 239 52 C266 21 320 28 343 58 C390 42 438 74 436 116 C473 140 467 185 438 204 C444 239 417 267 381 265 C350 291 311 280 291 260 C255 282 213 266 199 246 C164 257 126 233 132 204 C115 200 105 188 112 174Z"
          fill="#14344b" stroke="#6bd8e5" stroke-width="3"/>
        <path d="M121 171 C104 140 123 105 157 92 C182 78 207 86 222 108 C222 143 208 175 187 204 C164 217 136 204 121 171Z"
          fill="#2a83a0" fill-opacity=".85" stroke="#85e9f0" stroke-width="1.5"/>
        <path d="M191 88 C221 54 273 55 302 75 C315 101 302 132 280 151 C248 155 222 141 207 119Z"
          fill="#397fae" fill-opacity=".9" stroke="#91d8fa" stroke-width="1.5"/>
        <path d="M203 151 C235 139 273 146 296 170 C300 199 283 224 255 235 C222 231 198 210 187 185Z"
          fill="#218e91" fill-opacity=".92" stroke="#8de9dc" stroke-width="1.5"/>
        <path d="M302 77 C340 61 392 75 423 112 C441 143 426 176 397 190 C367 187 343 165 324 143Z"
          fill="#5b78bb" fill-opacity=".88" stroke="#b1c4ff" stroke-width="1.5"/>
        <path d="M334 204 C356 184 393 184 415 205 C433 226 413 253 388 258 C359 263 338 243 334 223Z"
          fill="#9874c7" fill-opacity=".9" stroke="#d8c2ff" stroke-width="1.5"/>
        <path d="M300 227 C315 226 326 236 329 250 L324 285 L301 285 L294 253Z"
          fill="#c18a51" fill-opacity=".95" stroke="#f1c88f" stroke-width="1.5"/>
        <g fill="#eaf7ff" font-family="Arial, sans-serif" font-size="13" font-weight="600">
          <text x="25" y="52">FRONT</text><text x="410" y="52">BACK</text>
          <text x="32" y="280" font-size="11" fill="#a9c2d5">Simplified regions only</text>
        </g>
        <g stroke="#b6dce8" stroke-width="1.4" fill="none" opacity=".8">
          <path d="M155 119 L91 85 L33 85"/><path d="M253 80 L253 35 L305 35"/>
          <path d="M238 202 L183 276 L110 276"/><path d="M390 139 L480 98 L548 98"/>
          <path d="M385 232 L470 276 L545 276"/><path d="M314 266 L314 310 L365 310"/>
        </g>
        <g fill="#eaf7ff" font-family="Arial, sans-serif" font-size="12">
          <text x="34" y="76">Frontal</text><text x="309" y="39">Parietal</text>
          <text x="38" y="270">Temporal</text><text x="552" y="102">Occipital</text>
          <text x="550" y="280">Cerebellum</text><text x="371" y="314">Brainstem</text>
        </g>
      </svg>
      <div class="brain-map-foot">Educational orientation aid only · Not for identifying a lesion's location on an uploaded image</div>
    </div>
    """
    st.markdown(diagram, unsafe_allow_html=True)

    selected_landmark = st.selectbox(
        "Explore an anatomical landmark",
        list(BRAIN_LANDMARKS),
        key="education_brain_landmark",
    )
    landmark = BRAIN_LANDMARKS[selected_landmark]
    a, b, c = st.columns(3)
    with a:
        with st.container(border=True):
            st.markdown("**Where it is**")
            st.write(landmark["location"])
    with b:
        with st.container(border=True):
            st.markdown("**What it generally does**")
            st.write(landmark["role"])
    with c:
        with st.container(border=True):
            st.markdown("**If affected**")
            st.write(landmark["possible_effects"])

    st.subheader("Landmark quick reference")
    quick_columns = st.columns(3)
    for index, (name, details) in enumerate(BRAIN_LANDMARKS.items()):
        with quick_columns[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(details["location"])
    st.warning(
        "Symptoms cannot reliably pinpoint a lesion, and a single 2D image cannot establish its 3D location. "
        "Reliable localization needs the complete image series and spatial/orientation metadata, typically from DICOM."
    )


def show_patient_support_page() -> None:
    st.markdown('<div class="section-kicker">Patient and caregiver education</div>', unsafe_allow_html=True)
    st.header("Patient Support & Care Guidance")
    st.write(
        "Practical information for preparing questions, supporting day-to-day wellbeing, and understanding what specialist care may involve."
    )
    st.error(
        "There is no proven home remedy, food, herb, or supplement that cures a brain tumour. "
        "Do not delay medical assessment or replace a clinician's treatment plan with home remedies."
    )

    selected_topic = st.selectbox(
        "Choose a topic to learn about",
        list(TUMOUR_OVERVIEWS),
        key="education_care_topic",
    )
    with st.container(border=True):
        st.markdown(f"**{selected_topic}**")
        st.write(TUMOUR_OVERVIEWS[selected_topic])
        st.caption(
            "This topic selector is for general education only. Do not choose a condition based on this app's AI output."
        )

    st.subheader("Supportive steps that may help at home")
    st.caption("These steps support comfort and coordination of care; they do not treat or shrink a tumour.")
    care_columns = st.columns(2)
    for index, (title, description) in enumerate(PATIENT_SUPPORT_TIPS):
        with care_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(description)

    st.info(
        "Active surveillance is a specialist-led plan involving scheduled follow-up and imaging for selected cases. "
        "It is not the same as ignoring a finding or attempting to treat it at home."
    )

    st.subheader("When to seek urgent medical help")
    for sign in URGENT_WARNING_SIGNS:
        st.markdown(f"- {sign}")
    st.warning(
        "For a first seizure, a seizure lasting over 5 minutes, sudden extremely severe headache, "
        "loss of consciousness, or breathing difficulty, contact local emergency services or go to an emergency department immediately."
    )

    st.subheader("Prepare for a specialist appointment")
    selected_questions = st.multiselect(
        "Choose the questions you want to discuss",
        CARE_QUESTIONS,
        default=CARE_QUESTIONS[:4],
        key="education_care_questions",
    )
    question_text = "Questions for my specialist\n\n" + "\n".join(
        f"- {question}" for question in selected_questions
    )
    st.download_button(
        "Download appointment question list",
        data=question_text,
        file_name="brain-tumour-specialist-questions.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not selected_questions,
    )
    st.caption("This creates a simple discussion aid, not a medical record. No symptom history or personal medical details are requested.")

    st.subheader("Evidence-based reading")
    for title, url in EDUCATION_SOURCES.items():
        st.markdown(f"- [{title}]({url})")
    st.caption(
        "Educational content is not a substitute for an individual assessment. Treatment decisions depend on the specific diagnosis, "
        "pathology and molecular results when relevant, imaging, symptoms, and the treating team's assessment."
    )


NAV_ITEMS = ["Overview", "MRI Analysis", "Classification", "Segmentation", "Explainable AI", "MRI Pattern Explorer", "Brain Anatomy & Landmarks", "Patient Support & Care Guidance", "Research & Metrics"]
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><div class="sidebar-mark">◎</div><div><div class="sidebar-overline">NEURO · AI</div><div class="sidebar-name">Brain MRI Lab</div></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Research workstation  /  2D imaging")
    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
    active_page = st.radio("Workspace", NAV_ITEMS, index=1, label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sidebar-label">Active model stack</div>', unsafe_allow_html=True)
    st.caption("EfficientNet-B0  ·  U-Net  ·  Grad-CAM")
    st.markdown('<div class="sidebar-label">Baseline protocol</div>', unsafe_allow_html=True)
    st.caption("BCE U-Net is the default. BCE + soft Dice is an optional comparison.")
    st.divider()
    st.markdown('<div class="footer-note">RESEARCH BUILD · NOT CLINICALLY VALIDATED</div>', unsafe_allow_html=True)

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
elif active_page == "MRI Pattern Explorer":
    show_mri_pattern_explorer(latest)
elif active_page == "Brain Anatomy & Landmarks":
    show_brain_landmarks_page()
elif active_page == "Patient Support & Care Guidance":
    show_patient_support_page()
elif active_page == "Research & Metrics":
    show_research_page()

st.divider()
st.markdown('<div class="footer-note">BRAIN MRI RESEARCH LAB  ·  RESEARCH & EDUCATION ONLY  ·  NOT CLINICALLY VALIDATED  ·  DO NOT USE FOR MEDICAL DECISIONS</div>', unsafe_allow_html=True)
