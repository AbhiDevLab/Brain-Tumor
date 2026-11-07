from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


def build_pdf_report(
    *,
    report: dict[str, Any],
    original_image: Image.Image,
    gradcam_image: Image.Image | None = None,
    segmentation_views: dict[str, dict[str, Any]] | None = None,
) -> bytes:
    """Create a downloadable, non-diagnostic PDF summary using inference outputs only.

    No per-upload accuracy, Dice, IoU, or clinical claim is computed because the
    uploaded image has no supplied ground-truth annotation.
    """
    output = BytesIO()
    classification = report.get("classification") or {}
    input_info = report.get("input") or {}
    processing = report.get("processing") or {}
    segmentation_models = report.get("segmentation_models") or {}
    flags = report.get("review_flags") or []
    segmentation_views = segmentation_views or {}

    with PdfPages(output) as pdf:
        pdf_metadata = pdf.infodict()
        pdf_metadata["Title"] = "Brain MRI Research Inference Report"
        pdf_metadata["Subject"] = "Research-only output summary; not a medical diagnosis"
        pdf_metadata["Keywords"] = "brain MRI, research prototype, EfficientNet-B0, U-Net, Grad-CAM"

        fig = plt.figure(figsize=(11.7, 8.3), constrained_layout=True)
        fig.suptitle("Brain MRI Research Report", fontsize=22, fontweight="bold", x=0.06, ha="left")
        fig.text(0.06, 0.91, "RESEARCH USE ONLY  |  NOT CLINICALLY VALIDATED", fontsize=10, fontweight="bold")
        fig.text(
            0.06, 0.875,
            f"Generated (UTC): {report.get('created_utc', 'Not recorded')}",
            fontsize=9,
        )

        lines = [
            "INPUT IMAGE",
            f"Dimensions: {input_info.get('width_px', '—')} × {input_info.get('height_px', '—')} pixels",
            f"Format: {input_info.get('format', '—')}  |  File size: {input_info.get('file_bytes', '—')} bytes",
            "",
            "CLASSIFICATION — EFFICIENTNET-B0",
        ]
        if classification:
            lines.append(f"Predicted category: {str(classification.get('predicted_class', 'Unavailable')).replace('_', ' ').title()}")
            lines.append(
                "Top uncalibrated softmax score: "
                f"{float(classification.get('score_not_calibrated_probability', 0.0)):.5f}"
            )
            lines.append("Class scores (uncalibrated):")
            for label, score in (classification.get("scores_not_calibrated_probabilities") or {}).items():
                lines.append(f"  • {str(label).replace('_', ' ').title()}: {float(score):.5f}")
            lines.append(f"Checkpoint: {classification.get('checkpoint_name', '—')}")
            lines.append(f"Checkpoint SHA-256: {classification.get('checkpoint_sha256', '—')}")
        else:
            lines.append("No classification output was generated.")
        lines.extend(["", "SEGMENTATION — PIXEL GRID SUMMARY"])
        if segmentation_models:
            for name, details in segmentation_models.items():
                lines.extend([
                    str(name),
                    f"  Loss: {details.get('loss_mode', '—')}  |  Threshold: {float(details.get('threshold', 0.5)):.2f}",
                    f"  Foreground fraction: {float(details.get('foreground_pixel_fraction', 0.0)) * 100:.3f}%",
                    f"  Foreground pixels: {int(details.get('predicted_foreground_pixels', 0)):,}",
                    f"  Checkpoint: {details.get('checkpoint_name', '—')}",
                ])
        else:
            lines.append("No segmentation output was generated.")
        lines.extend([
            "",
            "PROCESSING",
            f"Device: {processing.get('device', '—')}",
            f"Total measured inference path: {float(processing.get('total_seconds', 0.0)):.4f} seconds",
        ])
        for name, seconds in (processing.get("seconds") or {}).items():
            lines.append(f"  {name.replace('_', ' ')}: {float(seconds):.4f} seconds")
        lines.extend(["", "REVIEW FLAGS"])
        if flags:
            for flag in flags:
                lines.append(f"  [{str(flag.get('level', 'info')).upper()}] {flag.get('message', '')}")
        else:
            lines.append("No review flags recorded.")

        # Two-column text layout on the first page.
        left_text = "\n".join(lines[: max(1, len(lines) // 2)])
        right_text = "\n".join(lines[max(1, len(lines) // 2):])
        fig.text(0.06, 0.83, left_text, va="top", fontsize=9.2, linespacing=1.35, family="DejaVu Sans")
        fig.text(0.53, 0.83, right_text, va="top", fontsize=9.2, linespacing=1.35, family="DejaVu Sans")
        fig.text(
            0.06, 0.055,
            "Important: uploaded-image predictions have no ground-truth mask. This report does not calculate "
            "per-image accuracy, Dice, IoU, tumor size, or diagnostic certainty.",
            fontsize=8.2, wrap=True,
        )
        pdf.savefig(fig)
        plt.close(fig)

        visual_items: list[tuple[str, Image.Image]] = [("Original MRI", original_image)]
        if gradcam_image is not None:
            visual_items.append(("Grad-CAM classifier attribution (not a segmentation mask)", gradcam_image))
        if visual_items:
            fig, axes = plt.subplots(1, len(visual_items), figsize=(11.7, 6.3), squeeze=False)
            fig.suptitle("Input and classifier attribution", fontsize=17, fontweight="bold")
            for axis, (label, image) in zip(axes[0], visual_items):
                axis.imshow(np.asarray(image.convert("RGB")))
                axis.set_title(label, fontsize=10)
                axis.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if segmentation_views:
            names = list(segmentation_views)
            for start in range(0, len(names), 2):
                batch = names[start:start + 2]
                fig, axes = plt.subplots(len(batch), 3, figsize=(11.7, 4.5 * len(batch)), squeeze=False)
                fig.suptitle("Segmentation outputs — model-generated foreground proposals", fontsize=16, fontweight="bold")
                for row, name in enumerate(batch):
                    view = segmentation_views[name]
                    axes[row, 0].imshow(np.asarray(original_image.convert("RGB")))
                    axes[row, 0].set_title(f"{name}: input", fontsize=9)
                    axes[row, 1].imshow(view["mask_display"], cmap="gray", vmin=0, vmax=255)
                    axes[row, 1].set_title("Binary mask", fontsize=9)
                    axes[row, 2].imshow(np.asarray(view["overlay"].convert("RGB")))
                    axes[row, 2].set_title("Mask overlay", fontsize=9)
                    for axis in axes[row]:
                        axis.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    return output.getvalue()
