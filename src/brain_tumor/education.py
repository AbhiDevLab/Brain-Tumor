"""Plain-language, non-diagnostic education content for the Brain MRI Research Lab.

This module contains educational reference text only. It does not interpret scans,
generate diagnoses, or recommend an individual treatment plan.
"""

MRI_SEQUENCE_GUIDE = {
    "T1-weighted": {
        "plain": "A sequence often used to show brain anatomy clearly. Cerebrospinal fluid (CSF) is usually dark.",
        "look_for": "Anatomical boundaries and tissue structure; interpretation depends on the full protocol.",
    },
    "T2-weighted": {
        "plain": "Water-containing structures and many areas of tissue change appear relatively bright. CSF is usually bright.",
        "look_for": "Fluid-sensitive changes, but brightness is not specific to a tumor.",
    },
    "FLAIR": {
        "plain": "Similar to T2, but the signal from free CSF is suppressed so some nearby tissue changes are easier to see.",
        "look_for": "Areas of increased signal around ventricles, sulci, or within brain tissue.",
    },
    "T1 post-contrast": {
        "plain": "T1 images acquired after a gadolinium-based contrast agent may show areas that enhance.",
        "look_for": "Enhancement patterns interpreted alongside pre-contrast images and the rest of the MRI study.",
    },
    "DWI and ADC": {
        "plain": "Paired sequences that help assess the movement of water in tissue. They must be interpreted together.",
        "look_for": "True restricted diffusion generally has high DWI signal with corresponding low ADC signal.",
    },
    "SWI or GRE": {
        "plain": "Sequences sensitive to magnetic susceptibility, including some blood products and mineral deposits.",
        "look_for": "Susceptibility effects that need correlation with other sequences and clinical context.",
    },
}

MRI_PATTERN_GUIDE = {
    "T2 / FLAIR bright signal": {
        "what": "An area that looks brighter than nearby tissue on a T2-weighted or FLAIR image.",
        "why": "It can reflect increased tissue water or other changes. Possible causes include swelling, inflammation, infection, vascular injury, scarring, and tumor-related change.",
        "not_proof": "Brightness alone does not identify the cause, tumor type, grade, or whether a finding is cancer.",
        "context": "Check which sequence is shown, whether the finding is present on more than one sequence, its location, shape, and the radiologist's report.",
    },
    "Contrast enhancement": {
        "what": "An area becomes brighter on T1 images after intravenous contrast is given.",
        "why": "Enhancement can occur when the blood-brain barrier is altered or tissue has increased vascularity. Tumors are one possible cause, but infection, inflammation, and treatment effects can also enhance.",
        "not_proof": "Enhancement is not synonymous with cancer, and some tumors may show little or no enhancement.",
        "context": "Compare pre- and post-contrast images and review the complete study with a radiologist.",
    },
    "Edema / surrounding signal change": {
        "what": "A region of increased fluid signal in tissue near or around an abnormality.",
        "why": "It may occur around tumors, but can also accompany infection, inflammation, vascular injury, and other conditions.",
        "not_proof": "It does not by itself establish a tumor diagnosis, tumor grade, or severity.",
        "context": "The extent, location, symptoms, and any associated compression are assessed by the clinical team.",
    },
    "Mass effect / midline shift": {
        "what": "An abnormality or swelling may compress nearby structures or displace structures from their usual position.",
        "why": "Large lesions, swelling, bleeding, or other processes can produce pressure effects.",
        "not_proof": "A screenshot or single slice cannot safely measure pressure or determine the cause.",
        "context": "This requires prompt professional review in the context of the full MRI and the person's condition.",
    },
    "Ring-like enhancement": {
        "what": "Enhancement appears around part of a central area, creating a ring-like pattern.",
        "why": "Several different processes can produce this appearance, including some tumors, abscesses, and inflammatory or treatment-related changes.",
        "not_proof": "A ring pattern is not a tumor subtype and cannot distinguish these causes by itself.",
        "context": "Radiologists assess wall thickness, shape, diffusion, surrounding signal, other sequences, and clinical history.",
    },
    "Cystic or central non-enhancing area": {
        "what": "Part of an abnormality may look fluid-like or may not enhance in the same way as surrounding tissue.",
        "why": "This can have several explanations, including fluid-filled spaces, tissue breakdown, or necrosis in some lesions.",
        "not_proof": "A cystic appearance does not tell whether a lesion is benign or malignant.",
        "context": "The relationship to surrounding tissue and the full sequence set matter.",
    },
    "Restricted diffusion": {
        "what": "Water movement appears restricted on diffusion imaging, usually assessed using both DWI and ADC maps.",
        "why": "It can occur in acute ischemia, abscesses, and some highly cellular lesions, among other causes.",
        "not_proof": "DWI brightness alone is not enough; ADC correlation and timing are important.",
        "context": "This pattern needs radiologist interpretation and clinical correlation.",
    },
    "Susceptibility / possible blood products": {
        "what": "A dark or blooming signal on SWI or GRE can reflect magnetic susceptibility.",
        "why": "Blood products, veins, mineralization, and imaging artifacts can produce susceptibility effects.",
        "not_proof": "A dark spot on one sequence does not automatically mean active bleeding.",
        "context": "The sequence, appearance on other images, and clinical history must be reviewed.",
    },
}

BRAIN_LANDMARKS = {
    "Frontal lobe": {
        "location": "Toward the front of each cerebral hemisphere, behind the forehead.",
        "role": "Supports planning, attention, decision-making, aspects of personality, and voluntary movement. Speech production commonly involves the dominant frontal language network.",
        "possible_effects": "Depending on the exact area, changes can involve movement, planning, behavior, attention, or speech. These symptoms are not specific to a tumor.",
    },
    "Parietal lobe": {
        "location": "Toward the upper rear portion of each cerebral hemisphere.",
        "role": "Helps process touch and body position and supports spatial awareness and integration of sensory information.",
        "possible_effects": "Some conditions in this region can affect sensation, left-right awareness, or spatial processing.",
    },
    "Temporal lobe": {
        "location": "Along the sides of the brain, roughly behind the temples.",
        "role": "Involved in hearing, memory, emotion, and language comprehension. The hippocampus lies deep within the medial temporal region.",
        "possible_effects": "Depending on the side and exact structures involved, effects may include memory, hearing, language comprehension, or seizures.",
    },
    "Occipital lobe": {
        "location": "At the back of the brain.",
        "role": "Mainly processes visual information.",
        "possible_effects": "Some disorders in this region can affect parts of the visual field.",
    },
    "Cerebellum": {
        "location": "At the back and bottom of the skull, behind the brainstem.",
        "role": "Coordinates balance, posture, and the precision and timing of movement.",
        "possible_effects": "Problems can include unsteadiness, poor coordination, or difficulty with precise movements.",
    },
    "Brainstem": {
        "location": "At the base of the brain, connecting the brain with the spinal cord.",
        "role": "Contains pathways and centres involved in alertness, swallowing, eye movements, breathing, and cardiovascular functions.",
        "possible_effects": "Effects depend on the pathway or nerve involved and may include weakness, swallowing, eye movement, balance, or alertness changes.",
    },
    "Pituitary / sellar region": {
        "location": "Near the centre of the skull base, below the hypothalamus and close to the optic chiasm.",
        "role": "The pituitary gland helps regulate several hormone systems. Nearby structures include the optic pathways.",
        "possible_effects": "Some lesions can affect hormone levels or vision, but symptoms and significance vary widely.",
    },
    "Corpus callosum": {
        "location": "A curved band of nerve fibres deep between the two cerebral hemispheres.",
        "role": "Allows information to travel between the left and right sides of the brain.",
        "possible_effects": "Changes here may affect networks across both hemispheres; symptoms depend on the underlying condition.",
    },
    "Ventricles and CSF spaces": {
        "location": "Fluid-filled spaces within the brain that connect with the surrounding CSF pathways.",
        "role": "Contain and circulate cerebrospinal fluid, which cushions the brain and spinal cord.",
        "possible_effects": "Some conditions can obstruct CSF flow or enlarge these spaces. Ventricular size must be judged by a radiologist.",
    },
}

TUMOUR_OVERVIEWS = {
    "General information": "A scan label or AI class is not a confirmed diagnosis. The next steps depend on the full MRI, symptoms, examination, medical history, and specialist review.",
    "Glioma (educational overview)": "Gliomas arise from glial-lineage cells in the central nervous system. Their behaviour and treatment vary considerably by integrated pathology and molecular findings, location, and grade. An AI image class cannot establish any of these details.",
    "Meningioma (educational overview)": "Meningiomas arise from the meninges, the membranes around the brain and spinal cord. Some grow slowly and may be monitored by specialists; others need treatment depending on growth, symptoms, location, and other factors. Monitoring is still planned medical care.",
    "Pituitary-region lesion (educational overview)": "Pituitary-region lesions have varied causes. Assessment may involve hormone testing, visual assessment, and specialist review of dedicated imaging. Do not infer hormone function or the need for treatment from a standard 2D MRI image.",
    "No-tumor model output": "A model output of no tumor is not a guarantee that an MRI is normal and does not rule out disease. A radiologist's report and clinical assessment remain important when symptoms or concerns persist.",
}

PATIENT_SUPPORT_TIPS = [
    ("Keep a symptom record", "Note new or changing symptoms, when they happen, medicines taken as prescribed, and questions for the care team."),
    ("Support nutrition", "Aim for a varied, balanced diet that suits the person's needs. A doctor or oncology dietitian can help if appetite, swallowing, weight, or treatment side effects are a problem."),
    ("Plan rest and activity", "Use a manageable rest routine. Ask the clinical team what level of activity is safe, especially if balance, seizures, or weakness are concerns."),
    ("Use prescribed medicines as directed", "Do not start, stop, reduce, or change anti-seizure medicines, steroids, cancer medicines, or other prescriptions without the treating clinician."),
    ("Review supplements first", "Tell the care team about vitamins, herbs, teas, and supplements before using them. Some products can interact with cancer medicines or other treatment."),
    ("Arrange practical support", "Plan transport, help with appointments, and support for daily activities when needed. Emotional and caregiver support are part of care."),
]

URGENT_WARNING_SIGNS = [
    "A first seizure, a seizure lasting more than 5 minutes, repeated seizures without recovery, or a seizure causing serious injury.",
    "A sudden, extremely severe headache.",
    "Loss of consciousness, difficulty waking, or difficulty breathing.",
    "Sudden new weakness, difficulty speaking, or sudden major vision change.",
]

CARE_QUESTIONS = [
    "What does the radiology report say, and which findings are uncertain?",
    "Do I need a neurologist, neurosurgeon, neuro-oncologist, or endocrinologist?",
    "Are additional MRI sequences, comparison scans, tests, or a biopsy needed?",
    "What symptoms should trigger an urgent call or emergency assessment?",
    "Are my medicines, vitamins, herbs, or supplements safe with the proposed care?",
    "What supportive services or rehabilitation are available?",
]

EDUCATION_SOURCES = {
    "How to read a brain MRI report — RadiologyInfo (ACR/RSNA)": "https://www.radiologyinfo.org/en/info/article-brain-mri-report",
    "Brain MRI — RadiologyInfo (ACR/RSNA)": "https://www.radiologyinfo.org/info/mri-brain",
    "Adult CNS tumor treatment — National Cancer Institute": "https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq",
    "Complementary and alternative medicine — National Cancer Institute": "https://www.cancer.gov/about-cancer/treatment/cam/patient",
    "Diet, supplements, and cancer — National Cancer Institute": "https://www.cancer.gov/about-cancer/treatment/cam/diets-supplements",
    "Brain tumour symptoms and emergency advice — NHS": "https://www.nhs.uk/conditions/malignant-brain-tumour/symptoms/",
}
