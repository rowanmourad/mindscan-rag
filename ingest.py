"""
ingest.py
---------
Loads PDF papers from data/papers/<category>/*.pdf, cleans the extracted
text, filters out bibliography-heavy pages, and splits the remaining text
into overlapping chunks ready for embedding.
"""

import glob
import os
import re

import pandas as pd
from pypdf import PdfReader


def clean_pdf_text(text: str) -> str:
    """Collapses whitespace and strips common running headers/footers."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(Journal of [\w\s\-]+\(\d{4}\)\s*\d+[:\-–]\d+)", "", text)
    text = re.sub(
        r"(e-Prime - Advances in Electrical Engineering, Electronics and Energy \d+ \(\d{4}\) \S+)",
        "", text,
    )
    text = re.sub(r"(P\. Priyadarshini et al\.)", "", text)
    text = re.sub(r"(Springer|Frontiers in Oncology \| www\.frontiersin\.org)", "", text)
    return text.strip()


def looks_like_reference_page(text: str, digit_ratio_threshold: float = 0.12,
                               bracket_ref_threshold: int = 6) -> bool:
    """
    Heuristic: pages that are mostly a numbered bibliography add little
    retrieval value and dilute chunk quality (lots of names/years/DOIs,
    few actual claims).
    """
    if not text:
        return False
    digits = sum(c.isdigit() for c in text)
    digit_ratio = digits / max(len(text), 1)
    bracket_refs = len(re.findall(r"\[\d+\]", text))
    return digit_ratio > digit_ratio_threshold or bracket_refs > bracket_ref_threshold


def load_pdfs_from_folder(pdf_dir, min_chars: int = 30) -> list[dict]:
    """
    Walks pdf_dir/<category>/*.pdf and returns a list of per-page records:
        {doc_id, source_file, category, page_number, text}

    Expected folder layout:
        papers/glioma/*.pdf
        papers/meningioma/*.pdf
        papers/pituitary/*.pdf
        papers/mri_diagnosis/*.pdf
        papers/general_overview/*.pdf
    """
    records = []
    doc_id = 0

    categories = [
        d for d in sorted(os.listdir(pdf_dir))
        if os.path.isdir(os.path.join(pdf_dir, d))
    ]

    if not categories:
        print("No category subfolders found under", pdf_dir)
        print("Expected: papers/glioma, papers/meningioma, papers/pituitary,")
        print("          papers/mri_diagnosis, papers/general_overview")
        return records

    for category in categories:
        category_path = os.path.join(pdf_dir, category)
        pdf_files = sorted(glob.glob(os.path.join(category_path, "*.pdf")))

        for pdf_path in pdf_files:
            try:
                reader = PdfReader(pdf_path)
            except Exception as e:
                print(f"Failed to read {pdf_path}: {e}")
                continue

            for page_num, page in enumerate(reader.pages):
                try:
                    raw_text = page.extract_text() or ""
                except Exception as e:
                    print(f"  Skipped {pdf_path} page {page_num + 1}: {e}")
                    continue

                if looks_like_reference_page(raw_text):
                    continue  # skip bibliography-heavy pages

                text = clean_pdf_text(raw_text)

                if len(text) < min_chars:
                    continue  # skip near-empty pages

                records.append({
                    "doc_id": doc_id,
                    "source_file": os.path.basename(pdf_path),
                    "category": category,
                    "page_number": page_num + 1,
                    "text": text,
                })
                doc_id += 1

    return records


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 60) -> list[str]:
    """Splits text into overlapping word-count chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks_dataframe(pdf_records: list[dict]) -> pd.DataFrame:
    """Turns page-level records into a DataFrame of chunk rows ready to embed."""
    chunk_rows = []
    chunk_counter = 0

    for rec in pdf_records:
        for piece in chunk_text(rec["text"]):
            chunk_rows.append({
                "chunk_id": f"{rec['category']}_{rec['doc_id']}_{chunk_counter}",
                "category": rec["category"],
                "source_file": rec["source_file"],
                "page_number": rec["page_number"],
                "chunk_text": piece,
            })
            chunk_counter += 1

    return pd.DataFrame(chunk_rows)
