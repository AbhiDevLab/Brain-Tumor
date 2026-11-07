from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_landing_page_runs_without_model_artifacts(monkeypatch):
    for key in (
        "BRAIN_MRI_CLASSIFIER_CHECKPOINT",
        "BRAIN_MRI_CLASSIFIER_URL",
        "BRAIN_MRI_CLASSIFIER_TOKEN",
        "BRAIN_MRI_CLASSIFIER_SHA256",
        "BRAIN_MRI_SEGMENTER_CHECKPOINT",
        "BRAIN_MRI_SEGMENTER_URL",
        "BRAIN_MRI_SEGMENTER_TOKEN",
        "BRAIN_MRI_SEGMENTER_SHA256",
    ):
        monkeypatch.delenv(key, raising=False)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45).run()

    assert len(app.exception) == 0
    assert any("Research use only" in element.value for element in app.markdown)
    assert any("Upload a 2D brain MRI image" in element.label for element in app.file_uploader)

def test_education_pages_render_without_model_artifacts(monkeypatch):
    for key in (
        "BRAIN_MRI_CLASSIFIER_CHECKPOINT",
        "BRAIN_MRI_CLASSIFIER_URL",
        "BRAIN_MRI_CLASSIFIER_TOKEN",
        "BRAIN_MRI_CLASSIFIER_SHA256",
        "BRAIN_MRI_SEGMENTER_CHECKPOINT",
        "BRAIN_MRI_SEGMENTER_URL",
        "BRAIN_MRI_SEGMENTER_TOKEN",
        "BRAIN_MRI_SEGMENTER_SHA256",
    ):
        monkeypatch.delenv(key, raising=False)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=45).run()
    assert len(app.exception) == 0

    for page in (
        "MRI Pattern Explorer",
        "Brain Anatomy & Landmarks",
        "Patient Support & Care Guidance",
    ):
        app.radio[0].set_value(page).run()
        assert len(app.exception) == 0, f"Streamlit page failed to render: {page}"
