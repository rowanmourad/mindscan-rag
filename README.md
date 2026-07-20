# MindScan Brain-Tumor RAG Assistant

An AI decision-support tool for radiologists: it classifies a brain MRI
against a trained tumor model, retrieves supporting evidence from a
library of medical PDFs, and uses an LLM (via [OpenRouter](https://openrouter.ai))
to draft a structured clinical report (PDF) and answer free-text follow-up
questions.

This is a script version of the original Colab notebook
(`Brain_Tumor_RAG_MindScan_OpenRouter_Final.ipynb`), reorganized into
importable Python modules so it can run outside of Google Colab, with no
Google Drive dependency.

> **Disclaimer:** This tool produces AI-generated decision support, **not**
> a medical diagnosis. All output must be reviewed by a licensed physician
> before any clinical decision is made.

## Project layout

```
mindscan_rag/
├── main.py               # CLI entry point (build-kb / report / ask / test-llm)
├── config.py              # All paths + settings (env-var overridable)
├── ingest.py               # PDF loading, cleaning, chunking
├── vector_store.py         # Chroma vector DB setup + embedding
├── retrieval.py            # Semantic search over the knowledge base
├── model_pipeline.py       # Wraps the `braintumor` tumor-classification pipeline
├── llm_client.py           # OpenRouter API calls
├── report_generator.py     # Report prompt, JSON parsing, PDF rendering
├── qa.py                   # Free-text follow-up Q&A
├── requirements.txt
├── .env.example
├── data/
│   ├── papers/              # Put your PDF library here (see below)
│   ├── vector_db/            # Auto-created: persistent Chroma DB
│   └── sample_images/        # Optional sample MRI images for testing
├── models/                  # Put the trained .keras models + class_indices.json here
├── braintumor/               # The tumor-classification package (provided separately)
└── reports/                  # Auto-created: generated PDF reports land here
```

## 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`chromadb` can optionally use `zstd` for compression. If you hit an error
related to it, install the system package:

```bash
# Debian/Ubuntu
sudo apt-get install -y zstd
```

## 2. Set up your data (no Google Drive needed)

Everything is read from local folders relative to this project. When you
unzip the MindScan package on your own machine, make sure the folder looks
like this (or point `MINDSCAN_*` environment variables at wherever you keep
these instead -- see `.env.example`):

- `braintumor/` -- the tumor-classification package (must be importable)
- `models/` -- trained `.keras` models + `class_indices.json`
- `data/papers/<category>/*.pdf` -- your PDF library, one subfolder per
  category, e.g.:
  ```
  data/papers/glioma/*.pdf
  data/papers/meningioma/*.pdf
  data/papers/pituitary/*.pdf
  data/papers/mri_diagnosis/*.pdf
  data/papers/general_overview/*.pdf
  ```
- `data/sample_images/` -- (optional) MRI images to test with

No code changes are required to point at these folders -- just place the
files there. If you'd rather keep your data somewhere else entirely (e.g.
an external drive), set environment variables instead of editing any file:

```bash
export MINDSCAN_BASE_DIR="/path/to/your/data"
export MINDSCAN_REPO_DIR="/path/to/mindscan"
```

## 3. Set your OpenRouter API key

Get a free key at https://openrouter.ai/keys (no credit card required).
**Important:** also visit https://openrouter.ai/settings/privacy and enable
"Free model publication" -- this is required to use any `:free` model.

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

If you skip this step, the script will prompt you for the key interactively.

## 4. Usage

**Build/update the knowledge base** (run once, and again whenever you add PDFs):

```bash
python main.py build-kb
```

**Test your OpenRouter connection:**

```bash
python main.py test-llm
```

**Generate a full AI-assisted report** for an MRI image:

```bash
python main.py report \
  --image data/sample_images/glioma_sample1.jpg \
  --context "45-year-old patient, headache and visual disturbance for 3 weeks."
```

This prints the prediction, retrieved sources, and structured report to the
console, and saves a formatted PDF under `reports/`.

**Ask a free-text follow-up question:**

```bash
python main.py ask \
  --question "Why is the model confidence only this high, and is this probability spread typical?" \
  --image data/sample_images/glioma_sample1.jpg
```

Drop `--image` to ask a general question not tied to a specific patient.

## Notes on what changed from the notebook

- Removed the Google Colab / Google Drive mount cell and duplicated cells.
- Removed debug-only cells (e.g. listing `.keras` files).
- Split one long notebook into focused modules that import from each other.
- All Colab-specific paths were replaced with local, configurable paths in
  `config.py`.
- Added a proper CLI (`main.py`) so the pipeline can be run as scripts
  rather than executed cell-by-cell.
