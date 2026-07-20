"""
config.py
---------
Central configuration for the MindScan Brain-Tumor RAG project.

This REPLACES the original notebook's Google-Colab cell that did:

    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/brain_tumor_rag"
    REPO_DIR = os.path.join(BASE_DIR, "mindscan")

Instead, every path is resolved relative to this project folder (the folder
the doctor unzips on their own machine), and every path can be overridden
with an environment variable -- no Google Drive, no Colab, no hardcoded
personal paths.

The doctor / user does NOT need to edit this file. If they want to store
data somewhere else (e.g. an external drive with the PDF library and the
trained models), they can set environment variables before running, e.g.:

    export MINDSCAN_BASE_DIR="/path/to/my/data"
    export MINDSCAN_REPO_DIR="/path/to/mindscan"

On Windows (PowerShell):
    $env:MINDSCAN_BASE_DIR = "C:/path/to/my/data"
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Project root = the folder this file lives in (i.e. the unzipped project).
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# BASE_DIR: where the PDF library and the vector database live.
# Defaults to <project_root>/data, but can be overridden.
# --------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("MINDSCAN_BASE_DIR", PROJECT_ROOT / "data"))
PDF_DIR = Path(os.environ.get("MINDSCAN_PDF_DIR", BASE_DIR / "papers"))
CHROMA_DIR = Path(os.environ.get("MINDSCAN_CHROMA_DIR", BASE_DIR / "vector_db"))
SAMPLE_IMAGES_DIR = Path(
    os.environ.get("MINDSCAN_SAMPLE_IMAGES_DIR", BASE_DIR / "sample_images")
)

# --------------------------------------------------------------------------
# REPO_DIR: the folder that contains the `braintumor/` package and the
# `models/` folder (trained .keras models + class_indices.json).
# Defaults to the project root itself, since the doctor installs everything
# from one zip file. Override this only if the braintumor package lives
# somewhere else on their machine.
# --------------------------------------------------------------------------
REPO_DIR = Path(os.environ.get("MINDSCAN_REPO_DIR", PROJECT_ROOT))
MODELS_DIR = Path(os.environ.get("MINDSCAN_MODELS_DIR", REPO_DIR / "models"))

# --------------------------------------------------------------------------
# Where generated PDF reports are written.
# --------------------------------------------------------------------------
REPORTS_DIR = Path(os.environ.get("MINDSCAN_REPORTS_DIR", PROJECT_ROOT / "reports"))

# --------------------------------------------------------------------------
# OpenRouter (LLM) configuration.
# --------------------------------------------------------------------------
# Free-tier model availability on OpenRouter changes often (models get pulled
# to paid-only with little notice), so we try a list in order and fall back
# automatically if one is unavailable.
OPENROUTER_MODEL_CANDIDATES = [
    "openai/gpt-oss-20b:free",                  # primary
    "meta-llama/llama-3.3-70b-instruct:free",   # backup #1
    "qwen/qwen3-next-80b-a3b-instruct:free",    # backup #2
]
OPENROUTER_MODEL = OPENROUTER_MODEL_CANDIDATES[0]

# Biomedical sentence-transformer used for embeddings.
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# Speeds up HuggingFace downloads if the `hf_transfer` package is installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def ensure_directories() -> None:
    """Create all working directories if they don't exist yet."""
    for d in (PDF_DIR, CHROMA_DIR, SAMPLE_IMAGES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_openrouter_api_key() -> str:
    """
    Reads the OpenRouter API key from the OPENROUTER_API_KEY environment
    variable. Falls back to an interactive prompt (hidden input) if it's
    not set, so the script still works if the user forgot to export it.

    Get a free key at https://openrouter.ai/keys (no credit card required).
    IMPORTANT: also visit https://openrouter.ai/settings/privacy and enable
    "Free model publication" -- this is required to use ANY ':free' model;
    skipping it causes a confusing 404 error instead of a clear permissions
    error.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        from getpass import getpass
        api_key = getpass("Enter your OpenRouter API key: ")
    return api_key
