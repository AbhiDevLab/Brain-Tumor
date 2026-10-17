# Brain tumor research tracker

**Last reviewed:** 17 October 2026. This is a curated list, not an exhaustive survey. Verify versions, data releases, licenses, and experimental details before reproduction.

| Year | Work | Status | Relevance to this project |
|---|---|---|---|
| 2024 | [Almufareh et al., IEEE Access](https://doi.org/10.1109/ACCESS.2024.3359418) | Peer-reviewed | YOLOv5-seg vs YOLOv7-seg; useful for a later localization/mask comparison. Check label alignment and patient-level splitting. |
| 2025 | [Dudhe et al., IEEE SCEECS](https://doi.org/10.1109/SCEECS64059.2025.10940343) | Conference review | Overview of ML, CNN/U-Net, transfer learning, attention, transformers, and evaluation metrics. |
| 2025 | [Fateh et al., BRISC / Swin-HAFUNet](https://arxiv.org/abs/2506.14318) | arXiv preprint and dataset manuscript | Candidate four-class, 2D T1ce dataset. Verify mask availability, license, release, and official split before use. |
| 2026-03-22 | [Peng et al., adaptive feature fusion](https://doi.org/10.1038/s41598-026-44887-6) | Scientific Reports | Highlights why slice-level splitting can overstate performance if slices from one patient cross splits. |
| 2026-03-26 | [Systematic review on current trends](https://doi.org/10.1007/s44196-026-01283-2) | Peer-reviewed review | Supports reporting generalization, calibration, explainability, and domain shift. |
| 2026-04-11 | [Pajany et al., multimodal feature fusion](https://doi.org/10.1038/s41598-026-44957-9) | Scientific Reports | More complex classification and segmentation approach; possible later benchmark if data and compute match. |
| 2026 | [Jin et al., transformer MRI segmentation review](https://doi.org/10.1016/j.compmedimag.2026.102729) | Journal review | Background for a later transformer comparison. |
| Ongoing | [BraTS repository](https://github.com/brats) | Challenge ecosystem | Potential source for a separate volumetric/multimodal research track. |

When adding a paper, record the DOI or arXiv link, dataset version, patient count, modalities, split protocol, task, metrics, compute cost, code availability, limitations, and whether the result has been independently reproduced. Headline accuracy alone is not a valid basis for ranking models.
