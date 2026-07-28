"""reporting/generate_report.py - Generate complete case reports from a
PredictionResult (Phase 8 schema).

Produces a Markdown report with the following sections:
    1. Header             (patient info, date, model version)
    2. Findings summary   (prediction + confidence)
    3. Tumor characterization (type, location, size)         [if tumor]
    4. XAI findings       (what the model attended to)       [if available]
    5. Clinical interpretation (rule-based, confidence-tier) [if tumor]
    6. Recommendations    (rule-based, confidence-tier)
    7. Per-class probabilities table
    8. Disclaimer

Implementation uses plain string templates (no Jinja2 dependency).

Public API:
    generate_markdown_report(result, patient_info=None, model_version=None,
                             include_disclaimer=True) -> str
    generate_summary_block(result) -> str
    generate_clinical_interpretation(result) -> str
    generate_recommendations(result) -> str
    write_report_files(result, out_dir, base_name=...,
                       patient_info=None) -> list[Path]

CLI:
    python -m reporting.generate_report --input result.json
    python -m reporting.generate_report --input result.json --out-dir reports/
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .schema import PredictionResult


# ---------------------------------------------------------------------------
# Tumor-type descriptions (factual, neutral; for LLM downstream context)
# ---------------------------------------------------------------------------
_TUMOR_DESCRIPTIONS = {
    "glioma": (
        "Gliomas are a heterogeneous group of tumors that arise from the glial "
        "cells of the brain or spinal cord. They include astrocytomas, "
        "oligodendrogliomas, and ependymomas, and can range from low-grade "
        "(WHO Grade I-II) to high-grade (WHO Grade III-IV, e.g. glioblastoma). "
        "Definitive grading and molecular subtyping (IDH, 1p/19q, MGMT) "
        "require histopathology and molecular profiling."
    ),
    "meningioma": (
        "Meningiomas arise from the meninges (the membranes surrounding the brain "
        "and spinal cord). The large majority are benign (WHO Grade I); a smaller "
        "fraction are atypical (Grade II) or anaplastic (Grade III). They are "
        "typically extra-axial and well-circumscribed on imaging."
    ),
    "pituitary": (
        "Pituitary tumors arise from the pituitary gland at the base of the brain. "
        "Most are benign adenomas. Clinically they are categorized by size "
        "(micro- vs macro-adenoma) and by hormone-secretion status (functioning vs "
        "non-functioning). Workup includes endocrinology evaluation and "
        "dedicated sellar MRI."
    ),
}

# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------
def _confidence_tier(conf: float) -> str:
    if conf >= 0.90:
        return "high"
    if conf >= 0.75:
        return "moderate"
    if conf >= 0.60:
        return "low"
    return "very_low"


def _confidence_tier_human(tier: str) -> str:
    return {
        "high": "High confidence",
        "moderate": "Moderate confidence",
        "low": "Low confidence",
        "very_low": "Very low confidence",
    }[tier]


# ---------------------------------------------------------------------------
# Section: Clinical interpretation (rule-based)
# ---------------------------------------------------------------------------
def generate_clinical_interpretation(result: PredictionResult) -> str:
    """Plain-language interpretation of the prediction. Intentionally cautious."""
    tier = _confidence_tier(result.confidence)

    if result.prediction == "notumor":
        if tier == "high":
            return (
                f"The model is highly confident ({result.confidence*100:.1f}%) "
                "that no tumor is visible in this image. This is consistent "
                "with a normal brain MRI."
            )
        return (
            f"The model predicts no tumor with {result.confidence*100:.1f}% "
            "confidence. Confidence is below the high-confidence threshold; "
            "the result should be reviewed alongside the original imaging and "
            "any clinical correlates."
        )

    # tumor case
    desc = _TUMOR_DESCRIPTIONS.get(result.tumor_type, "")
    tier_label = _confidence_tier_human(tier).lower()
    sz_phrase = ""
    if result.tumor_size is not None:
        sz_phrase = (
            f" The attention region occupies "
            f"approximately {result.tumor_size.brain_coverage_pct:.1f}% of the "
            f"estimated brain area, which falls in the "
            f"'{result.tumor_size.category.replace('_', ' ')}' bucket."
        )

    return (
        f"The model predicts {result.tumor_type} with {tier_label} "
        f"({result.confidence*100:.1f}%). {desc}{sz_phrase}"
    )


# ---------------------------------------------------------------------------
# Section: Recommendations (rule-based)
# ---------------------------------------------------------------------------
def generate_recommendations(result: PredictionResult) -> str:
    """Generic next-step prompts, written as continuous prose."""
    tier = _confidence_tier(result.confidence)

    if result.prediction == "notumor":
        if tier in ("high", "moderate"):
            return (
                "No abnormality detected. Routine clinical follow-up as "
                "indicated by the referring clinician; no urgent imaging "
                "action recommended on the basis of this analysis alone."
            )
        return (
            "The model favors a non-tumor finding but with limited confidence. "
            "Consider re-reading by a radiologist, repeat or higher-quality "
            "imaging if the clinical picture warrants, and correlation with "
            "patient history and symptoms."
        )

    # tumor case
    base = (
        "Specialist review by a neuroradiologist is recommended. "
        "This automated finding should not be used as a sole basis for "
        "diagnosis or treatment planning."
    )
    if result.tumor_type == "glioma":
        more = (
            " Advanced sequences (DWI, perfusion, MR spectroscopy) and, where "
            "indicated, biopsy with molecular profiling (IDH, 1p/19q, MGMT) "
            "are typically required for definitive grading and management."
        )
    elif result.tumor_type == "meningioma":
        more = (
            " Contrast-enhanced MRI is the standard for further characterization. "
            "Management depends on size, location, growth rate, and symptoms; "
            "options range from observation to surgical resection and/or "
            "radiotherapy."
        )
    elif result.tumor_type == "pituitary":
        more = (
            " Dedicated sellar/parasellar MRI and endocrinology evaluation "
            "(pituitary hormone panel, visual fields if macroadenoma) are "
            "recommended for further workup."
        )
    else:
        more = ""

    if tier in ("low", "very_low"):
        more += (
            " Note that model confidence is limited; the prediction should be "
            "weighted accordingly and confirmed with expert review."
        )
    return base + more


# ---------------------------------------------------------------------------
# Section: Summary block (3-line clinical-style summary)
# ---------------------------------------------------------------------------
def generate_summary_block(result: PredictionResult) -> str:
    """A compact 3-line summary for tabular dashboards."""
    tier = _confidence_tier(result.confidence)
    line1 = (
        f"Prediction: {result.prediction.upper()} | "
        f"Confidence: {result.confidence*100:.2f}% ({tier})"
    )
    if result.is_tumor and result.tumor_size is not None and result.tumor_location is not None:
        line2 = (
            f"Region: {result.tumor_size.category.replace('_', ' ')} | "
            f"{result.tumor_size.brain_coverage_pct:.1f}% of brain | "
            f"center=({result.tumor_location.center_x}, "
            f"{result.tumor_location.center_y})"
        )
    elif result.is_tumor:
        line2 = "Region: not determined."
    else:
        line2 = "No tumor region reported."
    if result.xai_summary is not None:
        line3 = (
            f"XAI: {result.xai_summary.method} | "
            f"focus spread {result.xai_summary.focus_spread_pct:.1f}% | "
            f"peak {result.xai_summary.peak_activation:.3f}"
        )
    else:
        line3 = "XAI: not available."
    return "\n".join([line1, line2, line3])


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
_DISCLAIMER = (
    "This report is generated by an AI-assisted classification system intended "
    "for research and educational use. The output is NOT a clinical diagnosis. "
    "Predictions reflect statistical patterns learned from training data and "
    "may be wrong. The attention region indicates where the model focused; it "
    "is not a clinically validated tumor segmentation. All findings must be "
    "reviewed and confirmed by a qualified medical professional before any "
    "clinical decision is made."
)


def _format_per_class_table(probs: Dict[str, float], predicted: str) -> str:
    if not probs:
        return "*(no per-class probabilities provided)*"
    lines = ["| Class | Probability | Predicted |", "|---|---:|:---:|"]
    for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        marker = ">>" if cls == predicted else ""
        lines.append(f"| {cls} | {p*100:.2f}% | {marker} |")
    return "\n".join(lines)


def _format_patient_block(patient_info: Optional[Dict]) -> str:
    if not patient_info:
        return "*(patient information not supplied)*"
    rows = []
    for k, v in patient_info.items():
        rows.append(f"- **{k.replace('_', ' ').title()}**: {v}")
    return "\n".join(rows)


def generate_markdown_report(
    result: PredictionResult,
    patient_info: Optional[Dict] = None,
    model_version: Optional[str] = None,
    include_disclaimer: bool = True,
    *,
    verified: bool = False,
    corrected: bool = False,
    verified_by: Optional[str] = None,
) -> str:
    """Render a full case report as a single Markdown string.

    When ``verified`` is True (a clinician reviewed/edited the result via the GUI
    Clinical Verification Panel) the title switches from "MindScan Auto-Generated"
    to "Clinician Verified" (and "and Corrected" when ``corrected`` is True), and a
    verification banner is added.
    """
    tier = _confidence_tier(result.confidence)
    tier_human = _confidence_tier_human(tier)

    # ---- 1. Header
    if verified:
        title = ("# Brain MRI - Clinician Verified and Corrected Report"
                 if corrected else "# Brain MRI - Clinician Verified Report")
    else:
        title = "# Brain MRI - MindScan Auto-Generated Tumor Classification Report"
    header = [
        title,
        "",
        f"**Report generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"**Analysis timestamp:** {result.timestamp or 'unknown'}",
    ]
    if verified:
        who = f" by {verified_by}" if verified_by else ""
        header.append(
            f"**Verification:** Reviewed and "
            f"{'corrected' if corrected else 'verified'}{who} via the Clinical "
            f"Verification Panel.")
    if result.image_path:
        header.append(f"**Source image:** `{result.image_path}`")
    if result.model_info is not None:
        header.append(f"**Model:** `{Path(result.model_info.model_path).name}` "
                      f"(input {result.model_info.img_size}px)")
    if model_version:
        header.append(f"**Model version:** {model_version}")

    # ---- 2. Patient block
    sections = ["", "## Patient information", _format_patient_block(patient_info)]

    # ---- 3. Findings summary
    sections += [
        "",
        "## Findings",
        "",
        f"**Prediction:** {result.prediction.upper()}  ",
        f"**Confidence:** {result.confidence*100:.2f}%  ({tier_human})",
    ]
    if result.tumor_type:
        sections.append(f"**Tumor type (model):** {result.tumor_type}")

    # ---- 4. Tumor characterization (when applicable)
    if result.is_tumor:
        sections += ["", "## Tumor characterization", ""]
        if result.tumor_location is not None:
            loc = result.tumor_location
            sections += [
                f"- **Bounding box (resized image space):** "
                f"x={loc.bbox_x}, y={loc.bbox_y}, w={loc.bbox_w}, h={loc.bbox_h}",
                f"- **Centroid:** ({loc.center_x}, {loc.center_y})",
                f"- **Region area:** {loc.region_area_px} pixels  "
                f"({loc.image_coverage_pct:.2f}% of image, "
                f"{loc.brain_coverage_pct:.2f}% of estimated brain area)",
                f"- **Estimated brain area:** {loc.brain_area_px} pixels",
            ]
        else:
            sections.append(
                "- *Localization not available (no attention region above threshold).*"
            )
        if result.tumor_size is not None:
            sz = result.tumor_size
            sections.append(
                f"- **Size category:** {sz.category.replace('_', ' ')} "
                f"({sz.brain_coverage_pct:.2f}% of estimated brain)"
            )

    # ---- 5. XAI findings
    if result.xai_summary is not None:
        x = result.xai_summary
        sections += [
            "", "## Explainability (XAI)",
            "",
            f"- **Method:** {x.method}",
            f"- **Focus spread:** {x.focus_spread_pct:.2f}%  "
            "(lower = more focal attention)",
            f"- **Peak activation:** {x.peak_activation:.4f}",
            f"- **Mean activation:** {x.mean_activation:.4f}",
            f"- **Center of mass:** ({x.center_of_mass_x:.1f}, {x.center_of_mass_y:.1f})",
        ]
        if x.target_layer:
            sections.append(f"- **Target convolutional layer:** `{x.target_layer}`")
        if x.better_method:
            sections.append(
                f"- **Comparison (Grad-CAM vs Grad-CAM++):** {x.better_method} "
                "produced the tighter, more localized heatmap."
            )

    # ---- 6. Segmentation summary (if a real seg model produced one)
    if result.segmentation_summary is not None:
        s = result.segmentation_summary
        sections += [
            "", "## Segmentation summary",
            "",
            f"- **Method:** {s.method}",
            f"- **Mask area:** {s.mask_area_px} pixels "
            f"({s.mask_brain_coverage_pct:.2f}% of brain)",
        ]
        if s.dice_against_attention is not None:
            sections.append(
                f"- **Dice overlap with XAI attention:** {s.dice_against_attention:.3f}"
            )
        if s.mask_path:
            sections.append(f"- **Mask file:** `{s.mask_path}`")

    # ---- 7. Per-class probabilities
    sections += [
        "", "## Per-class probabilities",
        "",
        _format_per_class_table(result.per_class_probabilities, result.prediction),
    ]

    # ---- 8. Clinical interpretation + Recommendations
    sections += [
        "", "## Clinical interpretation",
        "",
        generate_clinical_interpretation(result),
        "", "## Recommendations",
        "",
        generate_recommendations(result),
    ]

    # ---- 9. Warnings (if any)
    if result.warnings:
        sections += ["", "## Warnings", ""]
        for w in result.warnings:
            sections.append(f"- {w}")

    # ---- 10. Disclaimer
    if include_disclaimer:
        sections += ["", "---", "", "## Disclaimer", "", _DISCLAIMER]

    return "\n".join(header + sections) + "\n"


# ---------------------------------------------------------------------------
# Convenience: write a full report bundle to disk
# ---------------------------------------------------------------------------
def write_report_files(
    result: PredictionResult,
    out_dir: str | Path,
    base_name: str = "case_report",
    patient_info: Optional[Dict] = None,
    model_version: Optional[str] = None,
) -> List[Path]:
    """Write three companion files: <base>.md, <base>.json, <base>.minimal.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{base_name}.md"
    json_path = out_dir / f"{base_name}.json"
    min_path = out_dir / f"{base_name}.minimal.json"

    md_text = generate_markdown_report(
        result, patient_info=patient_info, model_version=model_version,
    )
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(result.to_json(indent=2), encoding="utf-8")
    min_path.write_text(json.dumps(result.to_minimal_dict(), indent=2), encoding="utf-8")

    return [md_path, json_path, min_path]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Path to a serialized PredictionResult JSON.")
    ap.add_argument("--out-dir", default=".",
                    help="Directory to write the case report files into.")
    ap.add_argument("--base-name", default="case_report")
    ap.add_argument("--patient-info", default=None,
                    help="Optional path to a JSON file with patient metadata.")
    ap.add_argument("--model-version", default=None)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Accept either the full result dict or a Predictor's wrapped output
    if "json" in data and isinstance(data["json"], dict):
        data = data["json"]

    # Try the schema first; fall back to predictor-output adapter
    try:
        result = PredictionResult.from_dict(data)
    except (TypeError, KeyError, ValueError):
        result = PredictionResult.from_predictor_output(data)

    patient_info = None
    if args.patient_info:
        with open(args.patient_info, "r", encoding="utf-8") as fh:
            patient_info = json.load(fh)

    paths = write_report_files(
        result, out_dir=args.out_dir, base_name=args.base_name,
        patient_info=patient_info, model_version=args.model_version,
    )
    for p in paths:
        print(f"wrote: {p}")


if __name__ == "__main__":
    main()
