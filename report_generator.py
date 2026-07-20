"""
report_generator.py
--------------------
Builds the clinical-report LLM prompt, parses the structured JSON response,
renders it into a professional PDF, and ties prediction + retrieval + LLM +
PDF together in generate_medical_report().
"""

import datetime
import json as _json
import os
import re as _re

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from config import REPORTS_DIR
from llm_client import call_openrouter
from model_pipeline import predict_tumor
from retrieval import retrieve_knowledge

# --------------------------------------------------------------------------
# Prompt building + response parsing
# --------------------------------------------------------------------------

def build_clinical_report_prompt(prediction: dict, retrieved_chunks_df, patient_context: str = "") -> str:
    """
    The model's prediction is the PRIMARY fact this report is built around.
    The retrieved literature is SECONDARY -- used only to explain/contextualize
    the prediction, never to override, second-guess, or replace it.

    Asks the LLM to return STRICT JSON matching a standard radiology-report
    shape, so it can be rendered into a clean, professional PDF rather than
    dumped as raw text.
    """
    evidence_lines = []
    for i, row in retrieved_chunks_df.reset_index(drop=True).iterrows():
        evidence_lines.append(
            f"[Source {i + 1} | {row['category']} | {row['source_file']} p.{row['page_number']}]\n{row['text']}"
        )
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No supporting literature retrieved."

    confidence_pct = prediction["confidence"] * 100

    prompt = f"""You are a radiology decision-support assistant helping a physician interpret the
output of an MRI-based tumor classification model. You are NOT a substitute for a licensed
radiologist or oncologist. Your output is a draft note that must be reviewed by a qualified
clinician before any clinical decision is made.

THE MODEL'S PREDICTION IS THE PRIMARY FINDING OF THIS REPORT. Everything else supports it.

MODEL PREDICTION (PRIMARY FINDING):
- Predicted tumor type: {prediction['predicted_class']}
- Model confidence: {confidence_pct:.1f}%
- Full probability distribution: {prediction['all_probabilities']}

PATIENT CONTEXT (if provided):
{patient_context if patient_context else "No additional patient context provided."}

SUPPORTING LITERATURE (SECONDARY -- background context only, retrieved from the knowledge base):
{evidence_text}

TASK:
Return your answer as STRICT JSON ONLY (no markdown code fences, no preamble, no commentary
outside the JSON object) matching exactly this schema:

{{
  "clinical_history": "1-2 sentences restating the patient context in clinical language, or a note that none was provided.",
  "technique": "1-2 sentences describing this was an AI/ML classification of an MRI image against a trained tumor-type model, plus RAG-based literature retrieval.",
  "findings": "The model output stated as the primary finding, in plain clinical language, followed by literature-grounded imaging features/WHO classification/differentials that explain (not question) this finding. Cite sources inline using [Source N].",
  "impression": "1-3 sentences: the predicted tumor type and confidence as the bottom-line impression. If confidence is below 70%, state the result is uncertain and name differentials worth ruling out. If confidence is high, state that plainly.",
  "recommendations": "General next steps a clinician might consider, grounded in the prediction first, informed by literature second. Written as a short list of 2-4 items separated by semicolons."
}}

RULES:
- The model's prediction and confidence are the primary finding and must not be contradicted,
  downplayed, or reinterpreted based on the literature. The literature explains the prediction;
  it never overrides it.
- Use the literature only to add supporting detail. Do not invent citations, statistics, or facts
  not present in the evidence.
- If the literature does not cover something, say so rather than guessing -- this does not change
  the primary finding.
- Never state a definitive diagnosis -- describe the model's output as a prediction requiring
  clinical confirmation.
- Do not include a disclaimer field -- the disclaimer is added separately by the system.
- Return ONLY the JSON object, nothing else.
"""
    return prompt


def parse_report_json(raw_text: str) -> dict:
    """
    Robustly parses the LLM's JSON response, in case it wraps the JSON in
    code fences or adds stray text before/after (common even when explicitly
    told not to). Falls back to putting the raw text into 'findings' if
    parsing fails entirely, so a malformed response still produces a PDF
    instead of crashing the pipeline.
    """
    text = raw_text.strip()
    text = _re.sub(r"^```(json)?", "", text.strip(), flags=_re.IGNORECASE).strip()
    text = _re.sub(r"```$", "", text.strip()).strip()

    match = _re.search(r"\{.*\}", text, flags=_re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = _json.loads(text)
    except Exception as e:
        print("Warning: could not parse model output as JSON, falling back to raw text.", e)
        data = {
            "clinical_history": "Not available (parsing error).",
            "technique": "AI/ML-based MRI tumor classification with literature retrieval.",
            "findings": raw_text,
            "impression": "See findings -- structured impression unavailable due to a parsing error.",
            "recommendations": "Physician review required.",
        }

    for key in ["clinical_history", "technique", "findings", "impression", "recommendations"]:
        data.setdefault(key, "Not provided.")

    return data


# --------------------------------------------------------------------------
# PDF rendering
# --------------------------------------------------------------------------

_styles = getSampleStyleSheet()

_report_title_style = ParagraphStyle(
    "ReportTitle", parent=_styles["Title"], fontSize=18, spaceAfter=2,
    textColor=colors.HexColor("#1a2b4c"),
)
_subtitle_style = ParagraphStyle(
    "Subtitle", parent=_styles["Normal"], fontSize=9,
    textColor=colors.HexColor("#555555"), spaceAfter=10,
)
_section_heading_style = ParagraphStyle(
    "SectionHeading", parent=_styles["Heading2"], fontSize=12, spaceBefore=14,
    spaceAfter=6, textColor=colors.HexColor("#1a2b4c"), borderPadding=0,
)
_body_style = ParagraphStyle(
    "Body", parent=_styles["Normal"], fontSize=10, leading=14, spaceAfter=4,
)
_impression_style = ParagraphStyle(
    "Impression", parent=_styles["Normal"], fontSize=11, leading=15,
    spaceAfter=4, fontName="Helvetica-Bold",
)
_source_style = ParagraphStyle(
    "Source", parent=_styles["Normal"], fontSize=8.5, leading=12,
    spaceAfter=6, textColor=colors.HexColor("#333333"),
)
_disclaimer_style = ParagraphStyle(
    "Disclaimer", parent=_styles["Normal"], fontSize=8.5, leading=11,
    textColor=colors.HexColor("#7a1f1f"),
)


def _confidence_band(confidence_pct: float):
    if confidence_pct >= 85:
        return "High", colors.HexColor("#1e7e34")
    elif confidence_pct >= 70:
        return "Moderate", colors.HexColor("#8a6d00")
    else:
        return "Low / Uncertain", colors.HexColor("#a12626")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    footer_text = (
        "AI-generated decision support only -- NOT a diagnosis. Requires physician review. "
        f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    canvas.drawCentredString(letter[0] / 2, 0.4 * inch, footer_text)
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()


def generate_report_pdf(prediction: dict, retrieved_chunks_df, patient_context: str,
                         report_data: dict, output_path: str = None,
                         report_id: str = None) -> str:
    """
    Renders the structured report (dict with clinical_history, technique,
    findings, impression, recommendations) plus the model prediction and
    cited sources into a professionally formatted PDF.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_id = report_id or datetime.datetime.now().strftime("MS-%Y%m%d-%H%M%S")
    output_path = output_path or os.path.join(str(REPORTS_DIR), f"{report_id}.pdf")

    confidence_pct = prediction["confidence"] * 100
    band_label, band_color = _confidence_band(confidence_pct)

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    story = []

    story.append(Paragraph("MindScan AI-Assisted Radiology Decision Support", _report_title_style))
    story.append(Paragraph(
        f"Report ID: {report_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}",
        _subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#1a2b4c")))
    story.append(Spacer(1, 10))

    story.append(Paragraph("PATIENT INFORMATION", _section_heading_style))
    patient_table_data = [
        ["Clinical Context:", patient_context if patient_context else "Not provided"],
        ["Exam Type:", "MRI -- AI-Assisted Tumor Classification"],
    ]
    patient_table = Table(patient_table_data, colWidths=[1.5 * inch, 5.3 * inch])
    patient_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(patient_table)

    story.append(Paragraph("MODEL PREDICTION (PRIMARY FINDING)", _section_heading_style))
    prob_rows = [["Tumor Type", "Probability"]]
    for cls, prob in sorted(prediction["all_probabilities"].items(), key=lambda x: -x[1]):
        prob_rows.append([cls, f"{prob * 100:.1f}%"])

    pred_summary = Table(
        [[Paragraph(f"<b>Predicted class:</b> {prediction['predicted_class']}", _body_style),
          Paragraph(f"<b>Confidence:</b> {confidence_pct:.1f}% ({band_label})", _body_style)]],
        colWidths=[3.4 * inch, 3.4 * inch],
    )
    pred_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2f9")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#1a2b4c")),
        ("TEXTCOLOR", (1, 0), (1, 0), band_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(pred_summary)
    story.append(Spacer(1, 8))

    prob_table = Table(prob_rows, colWidths=[3.4 * inch, 3.4 * inch])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(prob_table)

    story.append(Paragraph("CLINICAL HISTORY", _section_heading_style))
    story.append(Paragraph(report_data["clinical_history"], _body_style))

    story.append(Paragraph("TECHNIQUE", _section_heading_style))
    story.append(Paragraph(report_data["technique"], _body_style))

    story.append(Paragraph("FINDINGS", _section_heading_style))
    story.append(Paragraph(report_data["findings"], _body_style))

    story.append(Paragraph("IMPRESSION", _section_heading_style))
    story.append(Paragraph(report_data["impression"], _impression_style))

    story.append(Paragraph("RECOMMENDATIONS", _section_heading_style))
    for item in [r.strip() for r in report_data["recommendations"].split(";") if r.strip()]:
        story.append(Paragraph(f"&bull; {item}", _body_style))

    story.append(Paragraph("SUPPORTING LITERATURE", _section_heading_style))
    if len(retrieved_chunks_df) == 0:
        story.append(Paragraph("No supporting literature retrieved.", _body_style))
    else:
        for i, row in retrieved_chunks_df.reset_index(drop=True).iterrows():
            story.append(Paragraph(
                f"[Source {i + 1}] {row['category']} -- {row['source_file']}, p.{row['page_number']}",
                _source_style,
            ))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#a12626")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report was generated by an AI decision-support system and is "
        "NOT a medical diagnosis. It is a draft intended solely to assist a licensed physician's "
        "review and must be independently verified before any clinical decision is made.",
        _disclaimer_style,
    ))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path


# --------------------------------------------------------------------------
# End-to-end orchestration
# --------------------------------------------------------------------------

def generate_medical_report(collection, api_key: str, image_path: str, patient_context: str = "",
                             top_k: int = 5, openrouter_model: str = None,
                             save_pdf: bool = True, output_path: str = None) -> dict:
    """
    Full pipeline: classify the MRI image -> retrieve supporting literature
    -> ask the LLM to draft a structured report -> render it to PDF.
    """
    prediction = predict_tumor(image_path)

    query = (
        f"{prediction['predicted_class']} tumor MRI diagnosis imaging features "
        f"WHO classification differential diagnosis"
    )

    retrieved = retrieve_knowledge(
        collection=collection,
        query=query,
        predicted_class=prediction["predicted_class"],
        k=top_k,
    )

    prompt = build_clinical_report_prompt(prediction, retrieved, patient_context)
    raw_response = call_openrouter(prompt, api_key=api_key, model=openrouter_model)
    report_data = parse_report_json(raw_response)

    pdf_path = None
    if save_pdf:
        pdf_path = generate_report_pdf(
            prediction=prediction,
            retrieved_chunks_df=retrieved,
            patient_context=patient_context,
            report_data=report_data,
            output_path=output_path,
        )
        print(f"Report PDF saved to: {pdf_path}")

    return {
        "prediction": prediction,
        "retrieved_chunks": retrieved,
        "prompt": prompt,
        "report_data": report_data,
        "report_raw": raw_response,
        "pdf_path": pdf_path,
    }
