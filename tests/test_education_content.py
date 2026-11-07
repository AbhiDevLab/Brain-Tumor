from __future__ import annotations

from src.brain_tumor.education import (
    BRAIN_LANDMARKS,
    CARE_QUESTIONS,
    EDUCATION_SOURCES,
    MRI_PATTERN_GUIDE,
    MRI_SEQUENCE_GUIDE,
    PATIENT_SUPPORT_TIPS,
    TUMOUR_OVERVIEWS,
    URGENT_WARNING_SIGNS,
)


def test_mri_education_covers_core_sequences_and_patterns():
    assert {"T1-weighted", "T2-weighted", "FLAIR", "DWI and ADC"} <= set(MRI_SEQUENCE_GUIDE)
    assert {"T2 / FLAIR bright signal", "Contrast enhancement", "Edema / surrounding signal change"} <= set(MRI_PATTERN_GUIDE)
    assert all({"what", "why", "not_proof", "context"} <= set(item) for item in MRI_PATTERN_GUIDE.values())


def test_anatomy_education_covers_major_landmarks():
    assert {"Frontal lobe", "Parietal lobe", "Temporal lobe", "Occipital lobe", "Cerebellum", "Brainstem"} <= set(BRAIN_LANDMARKS)
    assert all({"location", "role", "possible_effects"} <= set(item) for item in BRAIN_LANDMARKS.values())


def test_patient_guidance_is_supportive_not_a_home_cure():
    assert any("not" in text.lower() and "cure" in text.lower() for text in TUMOUR_OVERVIEWS.values())
    assert any("supplement" in description.lower() for _, description in PATIENT_SUPPORT_TIPS)
    assert any("seizure" in item.lower() for item in URGENT_WARNING_SIGNS)
    assert CARE_QUESTIONS
    assert EDUCATION_SOURCES
