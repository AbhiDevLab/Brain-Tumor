# Brain Tumor MRI: Existing Research vs. Proposed Work

**Status: research plan — planned improvements are hypotheses, not demonstrated results.**

The flowchart distinguishes established approaches in prior literature from this project's baseline and the controlled experiments planned after the data audit. It does not claim that the research question or the underlying techniques have never been studied.

```mermaid
flowchart TD
    A["Prior research: established methods"] --> B["Transfer-learning CNNs<br/>image-level classification"]
    A --> C["U-Net / CNNs<br/>pixel-level segmentation"]
    A --> D["YOLO-family methods<br/>detection / localization"]
    A --> E["Attention, transformers<br/>and multi-task models"]

    B --> F["Research gap to test<br/>Which changes help on our verified data<br/>under a leakage-aware held-out evaluation?"]
    C --> F
    D --> F
    E --> F

    F --> G["Step 0: Audit dataset<br/>labels, annotated masks, pairing,<br/>duplicates and split integrity"]
    G --> H["Baseline classifier<br/>EfficientNet-B0<br/>glioma / meningioma / pituitary / no tumor"]
    G --> I["Baseline segmenter<br/>U-Net binary tumor/background mask<br/>only for valid annotated cases"]

    H --> J["Evaluate independently<br/>macro-F1 and per-class recall"]
    I --> K["Evaluate segmentation<br/>Dice (primary), IoU and tumor recall"]
    J --> L["Controlled experiments and error analysis"]
    K --> L

    L --> M["A. Baseline U-Net<br/>binary cross-entropy"]
    M --> N["B. Loss test<br/>BCE + Dice; keep split and setup fixed"]
    N --> O["C. Attention test<br/>attention-gated skip connections"]
    O --> P["D. Difficult-case analysis<br/>small, irregular, low-contrast tumors;<br/>false positives and missed regions"]

    P --> Q{"Held-out improvement repeatable,<br/>recall acceptable and cost justified?"}
    Q -->|Yes| R["Retain candidate and report evidence"]
    Q -->|No| S["Do not claim improvement;<br/>keep baseline or revise hypothesis"]

    R --> T["Potential contribution:<br/>reproducible, same-split comparison<br/>and sensitivity-aware model selection"]
    S --> T

    classDef prior fill:#E8EEF7,stroke:#46648D,color:#17263D;
    classDef baseline fill:#E5F3E8,stroke:#3B7650,color:#15351F;
    classDef experiment fill:#FFF1D8,stroke:#A56B18,color:#4B3210;
    classDef caution fill:#FBE5E5,stroke:#A64242,color:#501A1A;
    class A,B,C,D,E prior;
    class G,H,I,J,K baseline;
    class L,M,N,O,P,Q experiment;
    class S,T caution;
```

## How to interpret it

### Already established in prior research
- CNN-based transfer learning for image-level classification.
- U-Net-style encoder-decoder segmentation.
- YOLO-family object detection/localization and related mask-prediction approaches.
- Attention, transformer and multi-task modeling, including the reviewed Swin-HAFNet direction.

These are existing research directions, not claimed inventions of this project.

### What this project will implement first
- A dataset audit and reproducible split protocol.
- An EfficientNet-B0 four-class image classifier.
- A separate U-Net binary segmenter, trained only when valid image-mask annotations are available.
- Independent evaluation before combining models in the local demo.

### What this project will test
A controlled comparison of BCE versus BCE + Dice loss, then attention-gated skip connections if warranted by error analysis, with special review of small, irregular and low-contrast tumors.

**Primary segmentation target:** Dice Similarity Coefficient (DSC). **Safety guardrail:** tumor sensitivity/recall. Also report IoU and inspect false-positive masks, including predictions on no-tumor images. For classification, report macro-F1 and per-class recall.

A change is retained only when held-out evidence supports it without unacceptable sensitivity loss or unjustified complexity. Training on real data and model metrics are pending; no improvement in accuracy or clinical utility is claimed. This is an academic research prototype, not a clinically validated diagnostic system.
