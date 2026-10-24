from __future__ import annotations

import hashlib
import json
import os
import streamlit as st
import torch

from src.brain_tumor.inference import (
    DEFAULT_MASK_THRESHOLD,
    CheckpointError,
    InputValidationError,
    build_result_report,
    checkpoint_source_status,
    configured_checkpoint_path,
    input_quality_flags,
    load_classifier_checkpoint,
    load_segmenter_checkpoint,
    predict_classification,
    predict_segmentation,
    preliminary_consistency_flags,
    validate_uploaded_image,
)

st.set_page_config(
    page_title="Brain Tumor MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem;}
    .hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#102a43 0%,#176b87 100%); color: #fff; margin-bottom: 1rem;}
    .hero h1 {color:#fff; font-size:2.25rem; line-height:1.16; margin:0 0 .65rem 0;}
    .hero p {color:#e3f2f7; max-width:780px; margin:0; font-size:1.03rem;}
    .eyebrow {letter-spacing:.12em; text-transform:uppercase; font-size:.76rem; font-weight:700; opacity:.88; margin-bottom:.65rem;}
    .subtle {color:#536578; font-size:.94rem;}
    .warning-note {border:1px solid #dfb35b; border-left:5px solid #d39b2a; padding: .9rem 1rem; border-radius:9px; background:rgba(211,155,42,.08); margin:.8rem 0 1.2rem 0;}
    .section-title {font-size:1.18rem; font-weight:700; margin:.3rem 0 .25rem 0;}
    div[data-testid="stMetric"] {border:1px solid rgba(120,140,160,.24); border-radius:12px; padding:.8rem 1rem;}
    </style>
    <div class="hero">
      <div class="eyebrow">Research demonstration · 2D image workflow</div>
      <h1>Brain Tumor MRI Analysis</h1>
      <p>Explore image-level model scores, a model-generated segmentation mask, and classifier Grad-CAM attribution using the project's saved research checkpoints.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="warning-note"><b>Research use only — not a medical device.</b> '
    'These models have not been clinically validated. Their outputs can be wrong and must not be used for diagnosis, treatment, or other medical decisions. '
    'Upload only de-identified images. Real MRI studies must be reviewed by qualified medical professionals.</div>',
    unsafe_allow_html=True,
)

with st.expander("How it works and supported inputs", expanded=False):
    st.markdown(
        """
        1. Upload a de-identified 2D image and inspect it before inference.
        2. EfficientNet-B0 returns a four-class model prediction and uncalibrated softmax scores.
        3. The BCE-trained U-Net returns a binary foreground mask and an aligned overlay.
        4. Preliminary consistency rules can flag disagreement; they are not a validated reliability gate.

        **Supported:** PNG, JPG/JPEG, BMP, and TIFF/TIF images up to 12 MB. Images must be decodable and at least 32 × 32 pixels. For multi-frame TIFFs, only the first frame is used.

        **Not supported in this version:** DICOM, NIfTI, 3D MRI volumes, or automatic interpretation across MRI sequences. The mask's pixel count and area fraction are image-grid statistics, not physical tumor size.
        """
    )

ROOT = Path(__file__).resolve().parent

# Copy only expected model settings from Streamlit's server-side secrets store.
_SECRET_KEYS = (
    "BRAIN_MRI_CLASSIFIER_CHECKPOINT", "BRAIN_MRI_CLASSIFIER_URL",
    "BRAIN_MRI_CLASSIFIER_TOKEN", "BRAIN_MRI_CLASSIFIER_SHA256",
    "BRAIN_MRI_SEGMENTER_CHECKPOINT", "BRAIN_MRI_SEGMENTER_URL",
    "BRAIN_MRI_SEGMENTER_TOKEN", "BRAIN_MRI_SEGMENTER_SHA256",
    "BRAIN_MRI_MODEL_CACHE_DIR",
)
for _secret_key in _SECRET_KEYS:
    if not os.environ.get(_secret_key):
        try:
            _secret_value = st.secrets.get(_secret_key)
        except Exception:
            _secret_value = None
        if _secret_value:
            os.environ[_secret_key] = str(_secret_value)

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
# Report configuration without downloading potentially large private files on page load.
checkpoint_status_labels = {
    "local_available": "Server-side checkpoint available",
    "local_missing": "Configured local checkpoint unavailable",
    "private_configured": "Private artifact configured; retrieval starts when analysis runs",
    "private_incomplete": "Private artifact settings incomplete or invalid",
    "unavailable": "Checkpoint unavailable",
}
classifier_source_status = checkpoint_source_status("classifier")
segmenter_source_status = checkpoint_source_status("segmenter")


@st.cache_resource(show_spinner=False)
def cached_classifier(path: str, mtime_ns: int, size_bytes: int, device_label: str):
    # mtime and size are cache-key components so artifact replacement triggers a reload.
    return load_classifier_checkpoint(path, torch.device(device_label))


@st.cache_resource(show_spinner=False)
def cached_segmenter(path: str, mtime_ns: int, size_bytes: int, device_label: str):
    return load_segmenter_checkpoint(path, torch.device(device_label))


with st.expander("Model availability", expanded=False):
    a, b = st.columns(2)
    with a:
        st.markdown("**EfficientNet-B0 classifier**")
        st.write(checkpoint_status_labels[classifier_source_status])
        st.caption("Expected output: glioma, meningioma, pituitary tumor, or no tumor.")
    with b:
        st.markdown("**Segmentation model (BCE U-Net default)**")
        st.write(checkpoint_status_labels[segmenter_source_status])
        st.caption("Default threshold: 0.50. The BCE + Dice experiment is not the default.")
    st.caption(f"Inference device: {'CUDA GPU' if device_name == 'cuda' else 'CPU'}.")

uploaded = st.file_uploader(
    "Upload a 2D brain MRI image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    help="Use a de-identified 2D image only. Maximum file size: 12 MB.",
)

if uploaded is None:
    # Clearing the uploader also clears retained in-session visualization/report data.
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("upload_key", None)
    st.info("Upload an image to preview it and check its dimensions before running inference.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**01 · Inspect**")
        st.caption("Validate the image and review the original pixels.")
    with col2:
        st.markdown("**02 · Analyze**")
        st.caption("Run the saved classifier and binary segmenter.")
    with col3:
        st.markdown("**03 · Review**")
        st.caption("Compare outputs, inspect warnings, and export JSON.")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    original, input_metadata = validate_uploaded_image(file_bytes, uploaded.name)
except InputValidationError as exc:
    st.error(str(exc))
    st.stop()

upload_key = hashlib.sha256(file_bytes).hexdigest()
if st.session_state.get("upload_key") != upload_key:
    st.session_state["upload_key"] = upload_key
    st.session_state.pop("analysis_result", None)

preview_col, details_col = st.columns([1.25, 1])
with preview_col:
    st.markdown('<div class="section-title">Original image preview</div>', unsafe_allow_html=True)
    st.image(original, use_container_width=True)
with details_col:
    st.markdown('<div class="section-title">Input validation</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Width", f'{input_metadata["width_px"]:,} px')
    with m2:
        st.metric("Height", f'{input_metadata["height_px"]:,} px')
    st.write(f'**Format:** {input_metadata["format"]}')
    st.write(f'**File size:** {input_metadata["file_bytes"] / (1024 * 1024):.2f} MB')
    st.write(f'**Intensity variation:** {input_metadata["intensity_std_0_to_255"]:.2f} / 255')
    st.caption("Uploads are processed in memory for the active session; this app does not write uploaded images to disk or maintain an image history. Clear the uploader or close the browser tab to discard the upload.")

threshold = st.slider(
    "Segmentation pixel threshold",
    min_value=0.05,
    max_value=0.95,
    value=float(DEFAULT_MASK_THRESHOLD),
    step=0.05,
    help="Experimental display threshold. The current baseline is 0.50; this control is not calibrated and does not imply clinical reliability.",
)
if st.session_state.get("analysis_threshold") != float(threshold):
    st.session_state.pop("analysis_result", None)
    st.session_state["analysis_threshold"] = float(threshold)

if st.button("Run MRI analysis", type="primary", use_container_width=True):
    classification = None
    segmentation = None
    errors: list[str] = []
    with st.spinner("Loading available checkpoints and running inference…"):
        try:
            classifier_path = configured_checkpoint_path("classifier")
            if not classifier_path.is_file():
                errors.append("Classifier checkpoint is unavailable. Configure BRAIN_MRI_CLASSIFIER_CHECKPOINT or the private classifier artifact settings on the server.")
            else:
                stat = classifier_path.stat()
                classifier, classifier_meta = cached_classifier(
                    str(classifier_path), stat.st_mtime_ns, stat.st_size, device_name
                )
                classification = predict_classification(classifier, original, classifier_meta, device)
        except CheckpointError:
            errors.append("The private classifier checkpoint could not be retrieved or loaded. Check the server-side artifact configuration.")
        except Exception:
            errors.append("Classifier inference failed. No classification result is available for this image.")

        try:
            segmenter_path = configured_checkpoint_path("segmenter")
            if not segmenter_path.is_file():
                errors.append("Segmentation checkpoint is unavailable. Configure BRAIN_MRI_SEGMENTER_CHECKPOINT or the private segmentation artifact settings on the server.")
            else:
                stat = segmenter_path.stat()
                segmenter, segmenter_meta = cached_segmenter(
                    str(segmenter_path), stat.st_mtime_ns, stat.st_size, device_name
                )
                segmentation = predict_segmentation(
                    segmenter, original, segmenter_meta, device, threshold=float(threshold)
                )
        except CheckpointError:
            errors.append("The private segmentation checkpoint could not be retrieved or loaded. Check the server-side artifact configuration.")
        except Exception:
            errors.append("Segmentation inference failed. No mask is available for this image.")

    flags = input_quality_flags(original)
    predicted_label = classification.get("predicted_class") if classification else None
    foreground_fraction = segmentation.get("foreground_pixel_fraction") if segmentation else None
    flags.extend(preliminary_consistency_flags(predicted_label, foreground_fraction))
    st.session_state["analysis_result"] = {
        "classification": classification,
        "segmentation": segmentation,
        "errors": errors,
        "flags": flags,
        "report": build_result_report(
            input_metadata=input_metadata,
            classification=classification,
            segmentation=segmentation,
            flags=flags,
            errors=errors,
            threshold=float(threshold),
        ),
    }

result = st.session_state.get("analysis_result")
if result is not None:
    st.divider()
    st.header("Analysis results")

    if result["errors"]:
        for message in result["errors"]:
            st.error(message)
    if not result["classification"] and not result["segmentation"]:
        st.warning("No model output was generated. Check the checkpoint configuration and try again after the server artifacts are available.")
    else:
        class_result = result["classification"]
        seg_result = result["segmentation"]
        if class_result:
            st.subheader("Classification")
            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.metric("Model prediction", class_result["predicted_class"].replace("_", " ").title())
                st.caption(
                    f'Softmax model score: {class_result["score_not_calibrated_probability"]:.4f}. '
                    "This is not a calibrated probability or a measure of clinical certainty."
                )
                st.caption(
                    f'Checkpoint: {class_result["checkpoint_name"]} · SHA-256: '
                    f'{class_result["checkpoint_sha256"][:12]}… · input tensor: '
                    f'{class_result["input_tensor_shape"]}'
                )
            with c2:
                st.markdown("**Scores for all classes**")
                st.bar_chart(class_result["scores_not_calibrated_probabilities"])
                st.caption("The bar chart shows uncalibrated model scores normalized by softmax, not validated probabilities.")
            if class_result.get("gradcam_overlay") is not None:
                st.image(
                    class_result["gradcam_overlay"],
                    caption="Grad-CAM classifier attribution overlay — not a segmentation mask or proof of correctness.",
                    use_container_width=True,
                )
            else:
                st.info("Grad-CAM attribution was unavailable for this run; the classifier result remains separately labelled.")

        if seg_result:
            st.subheader("Segmentation")
            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Foreground pixels", f'{seg_result["predicted_foreground_pixels"]:,}')
            with q2:
                st.metric("Foreground fraction", f'{seg_result["foreground_pixel_fraction"] * 100:.2f}%')
            with q3:
                st.metric("Pixel threshold", f'{seg_result["threshold"]:.2f}')
            st.caption(
                f'Checkpoint: {seg_result["checkpoint_name"]} · training loss: {seg_result["loss_mode"]} · '
                f'SHA-256: {seg_result["checkpoint_sha256"][:12]}… · output grid: '
                f'{seg_result["output_grid_size_px"][0]} × {seg_result["output_grid_size_px"][1]}. '
                "Foreground pixel count and fraction are pixel-grid statistics, not physical tumor size."
            )
            mask_col, overlay_col = st.columns(2)
            with mask_col:
                st.image(seg_result["mask_display"], caption="U-Net binary foreground mask", use_container_width=True)
            with overlay_col:
                st.image(seg_result["overlay"], caption="Predicted mask overlay aligned to original image", use_container_width=True)
            st.caption("This model-generated mask is not a clinically confirmed tumor boundary. It is distinct from Grad-CAM.")

    st.subheader("Consistency review")
    st.caption("Preliminary rules only. These flags have not been validated and cannot identify every unreliable output. A disagreement does not suppress either model result.")
    for flag in result["flags"]:
        if flag["level"] == "review":
            st.warning(flag["message"])
        else:
            st.info(flag["message"])

    st.download_button(
        "Download JSON result report",
        data=json.dumps(result["report"], indent=2, ensure_ascii=False),
        file_name="brain_mri_research_result.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Brain Tumor MRI Analysis · Research and education only · Not clinically validated · "
    "Do not use for medical decisions. MRI studies require review by qualified medical professionals."
)
