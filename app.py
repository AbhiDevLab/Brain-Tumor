from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.brain_tumor.data import classify_image_tensor, segmentation_image_tensor
from src.brain_tumor.explain import generate_gradcam, overlay_gradcam
from src.brain_tumor.models import UNet, build_classifier

ROOT = Path(__file__).resolve().parent
CLASSIFIER_PATH = ROOT / "artifacts" / "classifier.pt"
SEGMENTER_PATH = ROOT / "artifacts" / "segmenter.pt"

st.set_page_config(page_title="Brain MRI Research Prototype", page_icon="🧠", layout="wide")
st.title("Brain MRI Research Prototype")
st.caption("Research-only prototype • 2D MRI image workflow • Not a medical diagnosis")
st.warning("This tool is for research and educational experimentation only. It is not clinically validated and must not be used to make treatment decisions. Have medical images reviewed by a qualified radiologist.")

uploaded = st.file_uploader("Upload a 2D MRI slice", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("Upload a de-identified PNG/JPG/TIFF slice to inspect the preprocessing workflow. No file is saved to a history database.")
    st.markdown("**Model status**")
    st.write(f"Classification checkpoint: {'available' if CLASSIFIER_PATH.exists() else 'not trained / not installed'}")
    st.write(f"Segmentation checkpoint: {'available' if SEGMENTER_PATH.exists() else 'not trained / not installed'}")
    st.stop()

try:
    original = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

arr = np.asarray(original)
left, right = st.columns([1, 1])
with left:
    st.subheader("Uploaded slice")
    st.image(original, use_container_width=True)
with right:
    st.subheader("Input checks")
    st.write({"width_px": original.width, "height_px": original.height, "channels": 3,
              "mean_intensity_0_to_255": round(float(arr.mean()), 2),
              "standard_deviation": round(float(arr.std()), 2),
              "file_bytes": len(uploaded.getvalue())})
    st.caption("The current EfficientNet-B0 baseline resizes images to the model input dimensions. It is not yet a DICOM/NIfTI volumetric workflow.")

report = {"created_utc": datetime.now(timezone.utc).isoformat(), "file_name": uploaded.name,
          "input_size_px": [original.width, original.height], "research_only": True,
          "classification": None, "segmentation": None,
          "notice": "Not a medical diagnosis; require expert review."}

if not CLASSIFIER_PATH.exists() and not SEGMENTER_PATH.exists():
    st.info("Inference is disabled because no trained model checkpoints were found. The prototype will not fabricate a label, confidence score, or tumor mask. Train the baselines using the instructions in README.md.")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if CLASSIFIER_PATH.exists():
        with st.spinner("Running classification baseline…"):
            ckpt = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
            classes = ckpt["classes"]
            model = build_classifier(len(classes), pretrained=False).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            x = classify_image_tensor(original, ckpt.get("image_size", 224)).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            order = np.argsort(probs)[::-1]
            cam = generate_gradcam(model, x, int(order[0]), output_size=(224, 224))
            gradcam_overlay = overlay_gradcam(original.resize((224, 224), Image.Resampling.BILINEAR), cam)
            predicted = classes[int(order[0])]
            report["classification"] = {"predicted_class": predicted,
                                         "model_score_not_calibrated_probability": float(probs[order[0]]),
                                         "scores": {classes[i]: float(probs[i]) for i in range(len(classes))}}
        st.subheader("Classification baseline output")
        st.write(f"Predicted label: **{predicted}**")
        st.caption(f"Model score: {probs[order[0]]:.3f} — not a calibrated probability and not a diagnosis.")
        st.bar_chart({classes[i]: float(probs[i]) for i in range(len(classes))})
        st.image(gradcam_overlay, caption="Grad-CAM attribution overlay (not a tumor segmentation mask)", use_container_width=True)
        report["classification"]["explainability"] = "Grad-CAM generated; visualization is an attribution aid, not proof of model reasoning."
    if SEGMENTER_PATH.exists():
        with st.spinner("Running segmentation baseline…"):
            ckpt = torch.load(SEGMENTER_PATH, map_location=device, weights_only=False)
            model = UNet().to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            size = ckpt.get("image_size", 256)
            x = segmentation_image_tensor(original, size).to(device)
            with torch.no_grad():
                mask = (torch.sigmoid(model(x))[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
            resized_original = original.resize((size, size), Image.Resampling.BILINEAR)
            rgb = np.asarray(resized_original).copy()
            overlay = rgb.copy()
            overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([255, 40, 40])).astype(np.uint8)
            report["segmentation"] = {"predicted_tumor_pixels": int(mask.sum()),
                                      "predicted_tumor_pixel_fraction": float(mask.mean()),
                                      "mask_threshold": 0.5, "output_size_px": [size, size]}
        st.subheader("Binary segmentation baseline output")
        c1, c2 = st.columns(2)
        with c1:
            st.image(mask * 255, caption="Predicted binary mask", clamp=True, use_container_width=True)
        with c2:
            st.image(overlay, caption="Predicted mask overlay", use_container_width=True)
        st.caption("This is a binary segmentation mask, not a validated tumor boundary or tumor-subregion delineation.")

st.divider()
st.download_button("Download research result JSON", data=json.dumps(report, indent=2),
                   file_name="brain_mri_research_result.json", mime="application/json")
