"""braintumor/llm_report.py - Prepare structured outputs for a local LLM.

The user has a downloaded Hugging Face model. This module is the integration
seam: it turns the pipeline's structured results (classification, confidence,
tumor analysis, Grad-CAM findings, segmentation findings) into

    1. a single JSON payload with explicit guardrails, and
    2. a ready-to-send chat prompt (system + user),

and provides an OPTIONAL ``generate_report_with_llm`` that runs a local
``transformers`` model if one is available. The payload/prompt builders have no
heavy dependencies, so they always work even without transformers installed;
generation degrades gracefully (returns the deterministic template report from
``generate_report`` if the LLM can't be loaded).

Nothing here sends data to any external service - it is built for a *local*
model, consistent with the project's offline / privacy posture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .schema import PredictionResult


GUARDRAILS = [
    "You are assisting with a research/educational brain-MRI analysis tool.",
    "Do NOT provide a definitive diagnosis, tumor grade, or prognosis.",
    "Describe the model's findings as statistical predictions and image-analysis "
    "measurements, not ground truth.",
    "The attention region and weak segmentation show where the model looked; they "
    "are NOT clinically validated tumor boundaries.",
    "Always preserve the disclaimer and recommend qualified specialist review.",
    "Be precise and concise; use the numbers provided, do not invent values.",
]

SYSTEM_PROMPT = (
    "You are a careful medical-imaging report assistant. You write clear, "
    "structured, non-alarmist radiology-style narrative reports from STRUCTURED "
    "model outputs. You never state a definitive diagnosis; you frame everything "
    "as AI-assisted findings requiring clinician confirmation. You keep the "
    "provided disclaimer."
)


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------
def build_llm_payload(
    result: PredictionResult,
    *,
    tumor_analysis: Optional[Dict] = None,
    segmentation: Optional[Dict] = None,
    patient_info: Optional[Dict] = None,
) -> Dict[str, object]:
    """Combine all structured outputs into one LLM-ready payload.

    Parameters
    ----------
    result : the typed PredictionResult (classification + XAI + size/location).
    tumor_analysis : optional dict from tumor_analysis.to_llm_payload()['findings']
                     or TumorReport.to_dict().
    segmentation : optional dict from SegmentationResult.summary_dict().
    """
    findings = {
        "classification": {
            "prediction": result.prediction,
            "is_tumor": result.is_tumor,
            "tumor_type": result.tumor_type,
            "confidence": result.confidence,
            "per_class_probabilities": result.per_class_probabilities,
        },
        "explainability": (
            None if result.xai_summary is None else {
                "method": result.xai_summary.method,
                "focus_spread_pct": result.xai_summary.focus_spread_pct,
                "peak_activation": result.xai_summary.peak_activation,
                "center_of_mass": [result.xai_summary.center_of_mass_x,
                                   result.xai_summary.center_of_mass_y],
                "better_method": result.xai_summary.better_method,
                "interpretation": "Lower focus spread = more localized attention.",
            }
        ),
        "tumor_location": (None if result.tumor_location is None
                           else result.tumor_location.__dict__),
        "tumor_size": (None if result.tumor_size is None
                       else result.tumor_size.__dict__),
        "tumor_analysis": tumor_analysis,
        "segmentation": segmentation,
    }
    return {
        "task": "Write a concise, structured AI-assisted brain-MRI findings report "
                "from the structured data below. Sections: Summary, Findings, "
                "Explainability, Tumor characteristics, Recommendations, Disclaimer.",
        "guardrails": GUARDRAILS,
        "patient_info": patient_info or {},
        "model_info": (None if result.model_info is None else {
            "model": Path(result.model_info.model_path).name if result.model_info.model_path else "",
            "architecture": result.model_info.architecture,
            "img_size": result.model_info.img_size,
        }),
        "findings": findings,
        "disclaimer": (
            "AI-assisted research tool. Not a medical diagnosis. All findings must "
            "be confirmed by a qualified clinician."
        ),
    }


def build_prompt(payload: Dict[str, object]) -> Dict[str, str]:
    """Return {'system': ..., 'user': ...} chat messages for the LLM."""
    user = (
        "Generate the report from this structured data. Follow every guardrail. "
        "Use only the values given.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    return {"system": SYSTEM_PROMPT, "user": user}


def save_payload(payload: Dict[str, object], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Optional local generation (transformers). Degrades gracefully.
# ---------------------------------------------------------------------------
def generate_report_with_llm(
    payload: Dict[str, object],
    *,
    model_dir: Optional[str | Path] = None,
    max_new_tokens: int = 600,
    temperature: float = 0.3,
    fallback_result: Optional[PredictionResult] = None,
) -> str:
    """Generate a narrative report with a LOCAL Hugging Face model.

    ``model_dir`` should point at a locally downloaded HF model directory (or a
    hub id already cached offline). If transformers/the model is unavailable,
    this returns the deterministic template report (so the pipeline never breaks).
    """
    prompt = build_prompt(payload)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if model_dir is None:
            raise RuntimeError("No model_dir supplied for local LLM generation.")

        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype="auto", device_map="auto"
        )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        try:
            inputs = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
        except Exception:
            text = prompt["system"] + "\n\n" + prompt["user"]
            inputs = tok(text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
            )
        gen = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        return gen.strip()
    except Exception as exc:
        print(f"[llm_report] local LLM unavailable ({exc}); using template report.")
        if fallback_result is not None:
            from .generate_report import generate_markdown_report
            return generate_markdown_report(fallback_result)
        return ("LLM generation unavailable and no fallback PredictionResult given.\n\n"
                + json.dumps(payload, indent=2, default=str))
