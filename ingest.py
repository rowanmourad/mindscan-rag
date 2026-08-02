"""
ingest.py
---------
Loads PDF papers from data/papers/<category>/*.pdf, cleans the extracted
text, filters out bibliography-heavy pages, and splits the remaining text
into overlapping chunks ready for embedding.
"""

import glob                            # Finds all PDF files inside folders automatically.
import os                              # Handles file paths and folder navigation.
import re                              # Used for cleaning text with Regular Expressions.


import pandas as pd                    # Stores processed chunks in a structured DataFrame.
from pypdf import PdfReader            # Reads and extracts text from PDF papers


# ==========================================================
# STEP 1 — TEXT CLEANING
# ==========================================================
# Purpose:
# PDF extraction usually contains extra spaces, headers,
# footers, journal names, page numbers, and author names.
#
# These repeated elements appear on almost every page but
# contain no useful medical knowledge.
#
# Removing them produces cleaner text, which later generates
# better embeddings and improves retrieval accuracy.

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


# ==========================================================
# STEP 2 — REMOVE LOW-VALUE PAGES
# ==========================================================
# Purpose:
# Not every page inside a research paper contains useful
# information.
#
# Bibliography pages mostly contain:
#   • author names
#   • years
#   • DOIs
#   • citation numbers
#
# These pages add noise to the knowledge base.
#
# This function detects reference-heavy pages using two
# heuristics:
#
# 1. High percentage of digits.
# 2. Many citation patterns like [1], [2], [3]...
#
# If either condition is true, the page is skipped.

def looks_like_reference_page(text: str, digit_ratio_threshold: float = 0.12,            #12% of all characters on the page are digits, remove
                               bracket_ref_threshold: int = 6) -> bool:                  #so 6 citation are kept more than that removed
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



# ==========================================================
# STEP 3 — LOAD PDF PAPERS
# ==========================================================
# Purpose:
# This function scans every category folder, opens every PDF,
# extracts every page, cleans the text, removes unwanted pages,
# and stores the remaining pages in a structured format.
#
# Example folder structure:
#
# papers/
#     glioma/
#     meningioma/
#     pituitary/
#     mri_diagnosis/
#     general_overview/
#
# Each extracted page becomes one record containing:
#
# • document id
# • source paper
# • category
# • page number
# • cleaned text
#
# These records are later converted into chunks.

def load_pdfs_from_folder(pdf_dir, min_chars: int = 30) -> list[dict]:          #If the extracted text has less than 30 characters, skip that page,(not useful)
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



# ==========================================================
# STEP 4 — TEXT CHUNKING
# ==========================================================
# Purpose:
# Large Language Models do not work efficiently with very
# long documents.
#
# Therefore, every page is divided into smaller pieces
# called chunks.
#
# Chunk size = 250 words
# Overlap = 60 words
#
# Why overlap?
#
# Important medical information may be split between two
# chunks.
#
# Keeping 60 shared words preserves context and prevents
# losing information at chunk boundaries.
#
# Example:
#
# Chunk 1:
# words 1 → 250
#
# Chunk 2:
# words 191 → 440
#
# Shared words = 60


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


# ==========================================================
# STEP 5 — BUILD FINAL DATASET
# ==========================================================
# Purpose:
# Converts all page records into a structured DataFrame where
# every row represents one chunk.
#
# Each chunk stores:
#
# • Unique Chunk ID
# • Medical Category
# • Source Paper
# • Page Number
# • Chunk Text
#
# This DataFrame is the final output of preprocessing.
#
# The next stage (Embedding) will convert each chunk into
# a numerical vector before storing it inside ChromaDB.

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
