"""
qa.py
-----
Lets a doctor ask a free-text follow-up question about a patient case.
The model's prediction/confidence/probabilities (if provided) are the
PRIMARY basis for the answer. Retrieved PDF literature is SECONDARY --
used only to add supporting context, never to override or re-evaluate
the model's output.
"""

from llm_client import call_openrouter
from retrieval import retrieve_knowledge


def ask_question(collection, api_key: str, question: str, prediction: dict = None,
                  top_k: int = 5, openrouter_model: str = None,
                  include_general: bool = True) -> dict:
    """
    - question: the doctor's question
    - prediction: pass result["prediction"] from generate_medical_report() to
      include this patient's predicted_class, confidence, and full
      probability distribution as the primary answer basis. Pass None if the
      question is general (not about a specific patient).
    """
    predicted_class = prediction["predicted_class"] if prediction else None

    retrieved = retrieve_knowledge(
        collection=collection,
        query=question,
        predicted_class=predicted_class,
        k=top_k,
        include_general=include_general,
    )

    evidence_lines = []
    for i, row in retrieved.reset_index(drop=True).iterrows():
        evidence_lines.append(
            f"[Source {i + 1} | {row['category']} | {row['source_file']} p.{row['page_number']}]\n{row['text']}"
        )
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No supporting literature retrieved for this question."

    if prediction:
        prediction_text = (
            f"- Predicted tumor type: {prediction['predicted_class']}\n"
            f"- Model confidence: {prediction['confidence'] * 100:.1f}%\n"
            f"- Full probability distribution: {prediction['all_probabilities']}"
        )
    else:
        prediction_text = "No patient-specific model output provided for this question."

    prompt = f"""You are a radiology decision-support assistant. A physician has asked a follow-up
question about a patient case. The model's prediction below is the PRIMARY basis for your answer.
The retrieved literature is SECONDARY -- use it only to add supporting context, never to override,
question, or re-evaluate the model's output.

QUESTION:
{question}

MODEL OUTPUT FOR THIS PATIENT (PRIMARY):
{prediction_text}

SUPPORTING LITERATURE (SECONDARY -- background context only):
{evidence_text}

RULES:
- Treat the model's prediction and confidence as the primary, authoritative finding. Do not use the
  literature to contradict, downplay, or second-guess it.
- If the question is about the model's numbers (confidence, probabilities), answer primarily by
  referencing those numbers directly.
- Use the literature only to add supporting explanation. Cite every literature-based claim with [Source N].
- Do not invent facts, statistics, or citations not present in the model output or literature above.
- If the literature does not fully cover the question, say so -- this does not change the primary finding.
- Keep the answer concise and clinically appropriate. This is decision support, not a diagnosis.
"""

    answer = call_openrouter(prompt, api_key=api_key, model=openrouter_model)

    return {
        "question": question,
        "prediction": prediction,
        "retrieved_chunks": retrieved,
        "answer": answer,
    }
