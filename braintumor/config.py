"""Single source of truth for the braintumor package.

Centralizes paths, class definitions, and the EfficientNet training recipe that
were previously scattered across notebook cells and three different packages
(``improvements/``, ``all/``, ``GUI2/``). Edit here, not in individual modules.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Repo root = the parent of this ``braintumor`` package.
ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "Datasets" / "archive (2)"
TRAIN_DIR = DATASET_DIR / "Training"        # 1400 imgs/class
TEST_DIR = DATASET_DIR / "Testing"          # 400 imgs/class

# Canonical output locations (project root, not buried in a sub-package).
OUT_DIR = ROOT / "artifacts"
# Trained model weights live in ONE canonical store: ``models/``. Every trainer
# (train_4class / train_binary / train_subtype / fusion) writes here, and the
# registry discovers them here first. ``artifacts/`` is for generated reports,
# figures, audits and per-image case bundles only.
MODELS_DIR = ROOT / "models"
REPORTS_DIR = OUT_DIR / "reports"
AUDIT_DIR = OUT_DIR / "audit"
CASES_DIR = OUT_DIR / "cases"               # generated per-image case reports
for _d in (OUT_DIR, MODELS_DIR, REPORTS_DIR, AUDIT_DIR, CASES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Model search path (where registry.py looks for trained weights) -------
# Ordered by preference within each role. The registry picks the highest-priority
# model that exists; non-existent dirs are skipped silently. ``models/`` is the
# canonical store (all trained .keras + the alternative-track .pth live there);
# ``artifacts/models/`` holds models produced by new training runs.
MODEL_SEARCH_DIRS = [
    MODELS_DIR,                     # models/ — canonical store + new training runs
    OUT_DIR / "models",             # artifacts/models — legacy/extra runs
    # Legacy fallbacks (kept so the registry still works if models were not moved)
    ROOT / "improvements" / "artifacts" / "models",
    ROOT / "stadge2",
    ROOT / "_legacy" / "stadge2",
    ROOT / "_legacy" / "improvements" / "artifacts" / "models",
]

# --- Reproducibility -------------------------------------------------------
SEED = 42

# --- Classes ---------------------------------------------------------------
# Folder order is alphabetical, which is what Keras generators use too.
# Canonical no-tumor label is "notumor" (matches dataset folders + schema).
CLASSES_4 = ["glioma", "meningioma", "notumor", "pituitary"]
CLASSES_3 = ["glioma", "meningioma", "pituitary"]
CLASSES_BINARY = ["healthy", "tumor"]

# --- EfficientNet recipe ---------------------------------------------------
IMG_SIZE = {"b0": 224, "b2": 260, "b3": 300}
DEFAULT_VARIANT = "b2"

BATCH_SIZE = 32
PHASE1_EPOCHS = 12          # frozen-backbone head warmup
PHASE2_EPOCHS = 25          # gradual fine-tune
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
LABEL_SMOOTHING = 0.05      # mild, robust to label noise
UNFREEZE_FROM_BLOCK = 5     # block-aware fine-tune; BN kept frozen

# --- Audit thresholds ------------------------------------------------------
AUDIT_FLAG_CONF = 0.90      # glioma->meningioma confidence to flag as noise
HASH_LEAK_THRESHOLD = 5     # Hamming distance (of 64) for near-duplicate

# --- TTA -------------------------------------------------------------------
TTA_ENABLED = True
