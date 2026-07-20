"""improvements/two_stage.py - Hierarchical Binary -> Subtype inference.

Pipeline:
    image -> Binary classifier
           -> If healthy (P_tumor < threshold): return notumor, stop.
           -> Otherwise: run Subtype classifier, return final subtype.

Produces a unified 4-class probability vector aligned to CLASSES_4
(glioma, meningioma, notumor, pituitary) so this can be evaluated head-to-head
against the single-stage 4-class model.

Joint probability mapping for tumor branch:
    P(glioma)     = P(tumor) * P(glioma | tumor)
    P(meningioma) = P(tumor) * P(meningioma | tumor)
    P(pituitary)  = P(tumor) * P(pituitary | tumor)
    P(notumor)    = P(healthy)

Usage:
    from improvements.two_stage import TwoStageClassifier
    clf = TwoStageClassifier(
        binary_model_path="artifacts/models/binary_b2_clean.keras",
        subtype_model_path="artifacts/models/subtype_b2_clean.keras",
        binary_img_size=260, subtype_img_size=260,
    )
    result = clf.predict_image("path/to/mri.jpg")
    metrics = clf.evaluate_dataframe(test_df)

CLI:
    python -m improvements.two_stage \
        --binary artifacts/models/binary_b2_clean.keras \
        --subtype artifacts/models/subtype_b2_clean.keras \
        --binary-size 260 --subtype-size 260
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config


# Index order used everywhere in this module (alphabetical, matches Keras default)
FULL_CLASSES = list(config.CLASSES_4)          # ["glioma","meningioma","notumor","pituitary"]
SUBTYPE_CLASSES = list(config.CLASSES_3)        # ["glioma","meningioma","pituitary"]
BINARY_CLASSES = ["healthy", "tumor"]

# Subtype index in CLASSES_3 -> position in CLASSES_4
_SUBTYPE_TO_FULL = {
    SUBTYPE_CLASSES.index(c): FULL_CLASSES.index(c) for c in SUBTYPE_CLASSES
}
_NOTUMOR_IDX = FULL_CLASSES.index("notumor")


# ---------------------------------------------------------------------------
# Image loading (raw [0,255], no /255 — EfficientNet handles it; for the
# Xception branch we apply preprocess_input as an in-graph Lambda in train_*.py
# so the same raw input is correct for either backbone family).
# ---------------------------------------------------------------------------
def _load_image_batch(paths: List[str], size: int) -> np.ndarray:
    from PIL import Image

    arr = np.zeros((len(paths), size, size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((size, size))
            arr[i] = np.asarray(im, dtype=np.float32)
    return arr


class TwoStageClassifier:
    """Hierarchical binary -> subtype pipeline producing 4-class probabilities."""

    def __init__(
        self,
        binary_model_path: str | Path,
        subtype_model_path: str | Path,
        binary_img_size: int = 260,
        subtype_img_size: int = 260,
        tumor_threshold: float = 0.5,
    ):
        import tensorflow as tf

        self.binary_path = Path(binary_model_path)
        self.subtype_path = Path(subtype_model_path)
        self.binary_img_size = binary_img_size
        self.subtype_img_size = subtype_img_size
        self.tumor_threshold = float(tumor_threshold)

        print(f"[two-stage] loading binary  : {self.binary_path}")
        self.binary_model = tf.keras.models.load_model(self.binary_path)
        print(f"[two-stage] loading subtype : {self.subtype_path}")
        self.subtype_model = tf.keras.models.load_model(self.subtype_path)

        # Sanity-check output widths
        n_bin = int(self.binary_model.output_shape[-1])
        n_sub = int(self.subtype_model.output_shape[-1])
        if n_bin != 2:
            raise ValueError(f"Binary model has {n_bin} outputs, expected 2.")
        if n_sub != 3:
            raise ValueError(f"Subtype model has {n_sub} outputs, expected 3.")

    # ----------------------------------------------------------------
    # Single image
    # ----------------------------------------------------------------
    def predict_image(self, img_path: str) -> Dict:
        """Full hierarchical prediction for one image. Returns a dict ready for
        downstream JSON reporting (Phase 8).
        """
        # Binary stage
        x_bin = _load_image_batch([img_path], self.binary_img_size)
        p_bin = self.binary_model.predict(x_bin, verbose=0)[0]   # [p_healthy, p_tumor]
        p_healthy = float(p_bin[0])
        p_tumor = float(p_bin[1])

        full_probs = np.zeros(4, dtype=np.float32)
        full_probs[_NOTUMOR_IDX] = p_healthy

        if p_tumor < self.tumor_threshold:
            return {
                "stage_reached": "binary",
                "prediction": "notumor",
                "is_tumor": False,
                "confidence": p_healthy,
                "binary_probs": {"healthy": p_healthy, "tumor": p_tumor},
                "subtype_probs": None,
                "full_probs": {c: float(full_probs[i]) for i, c in enumerate(FULL_CLASSES)},
                "tumor_threshold": self.tumor_threshold,
            }

        # Subtype stage
        x_sub = _load_image_batch([img_path], self.subtype_img_size)
        p_sub = self.subtype_model.predict(x_sub, verbose=0)[0]  # [p_g, p_m, p_p]
        for sub_i, p in enumerate(p_sub):
            full_probs[_SUBTYPE_TO_FULL[sub_i]] = p_tumor * float(p)

        # Final prediction is argmax over the 4-class joint distribution
        pred_idx = int(np.argmax(full_probs))
        pred_class = FULL_CLASSES[pred_idx]
        conf = float(full_probs[pred_idx])

        return {
            "stage_reached": "subtype",
            "prediction": pred_class,
            "is_tumor": pred_class != "notumor",
            "confidence": conf,
            "binary_probs": {"healthy": p_healthy, "tumor": p_tumor},
            "subtype_probs": {SUBTYPE_CLASSES[i]: float(p_sub[i]) for i in range(3)},
            "full_probs": {c: float(full_probs[i]) for i, c in enumerate(FULL_CLASSES)},
            "tumor_threshold": self.tumor_threshold,
        }

    # ----------------------------------------------------------------
    # Batch over a dataframe (returns probability matrix aligned to CLASSES_4)
    # ----------------------------------------------------------------
    def predict_dataframe(
        self, df: pd.DataFrame, batch_size: int = 32,
    ) -> np.ndarray:
        """Return (N, 4) probability matrix in CLASSES_4 order."""
        N = len(df)
        out = np.zeros((N, 4), dtype=np.float32)

        # 1) Binary pass over everything
        p_bin_all = np.zeros((N, 2), dtype=np.float32)
        for s in range(0, N, batch_size):
            paths = df["Class Path"].iloc[s:s + batch_size].tolist()
            x = _load_image_batch(paths, self.binary_img_size)
            p_bin_all[s:s + batch_size] = self.binary_model.predict(x, verbose=0)

        p_healthy = p_bin_all[:, 0]
        p_tumor = p_bin_all[:, 1]
        out[:, _NOTUMOR_IDX] = p_healthy

        # 2) Subtype pass only on rows where the binary stage said TUMOR
        tumor_mask = p_tumor >= self.tumor_threshold
        tumor_idx = np.where(tumor_mask)[0]
        if len(tumor_idx) > 0:
            for s in range(0, len(tumor_idx), batch_size):
                sub_idx = tumor_idx[s:s + batch_size]
                paths = df["Class Path"].iloc[sub_idx].tolist()
                x = _load_image_batch(paths, self.subtype_img_size)
                p_sub = self.subtype_model.predict(x, verbose=0)
                for k, row_i in enumerate(sub_idx):
                    for sub_j in range(3):
                        out[row_i, _SUBTYPE_TO_FULL[sub_j]] = (
                            p_tumor[row_i] * p_sub[k, sub_j]
                        )

        # For rows below threshold (healthy), distribute tiny residual tumor mass
        # uniformly across the 3 tumor classes so the row sums to 1 cleanly.
        for i in np.where(~tumor_mask)[0]:
            residual = max(0.0, 1.0 - float(out[i].sum()))
            for sub_j in range(3):
                out[i, _SUBTYPE_TO_FULL[sub_j]] = residual / 3.0

        return out

    # ----------------------------------------------------------------
    # Evaluation against ground-truth labels in df["Class"]
    # ----------------------------------------------------------------
    def evaluate_dataframe(
        self, df: pd.DataFrame, *, save_dir: Optional[Path] = None,
        title: str = "two_stage",
    ) -> Dict:
        from .evaluate import compute_metrics

        probs = self.predict_dataframe(df)
        y_true = df["Class"].map(lambda c: FULL_CLASSES.index(c)).to_numpy()
        y_pred = probs.argmax(axis=1)
        metrics = compute_metrics(y_true, y_pred, probs, FULL_CLASSES)

        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(
            y_true, y_pred, target_names=FULL_CLASSES, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

        print("=" * 60)
        print(f" EVALUATION: {title}")
        print("=" * 60)
        print(f"  accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"  macro F1 : {metrics['f1']*100:.2f}%")
        print(f"  macro AUC: {metrics['roc_auc']:.4f}")
        print(report)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            _save_confusion(cm, FULL_CLASSES, save_dir / f"{title}_confusion.png", title)
            with open(save_dir / f"{title}_metrics.json", "w", encoding="utf-8") as fh:
                json.dump({
                    "metrics": metrics,
                    "binary_model": str(self.binary_path),
                    "subtype_model": str(self.subtype_path),
                    "tumor_threshold": self.tumor_threshold,
                    "binary_img_size": self.binary_img_size,
                    "subtype_img_size": self.subtype_img_size,
                }, fh, indent=2)
        return {"metrics": metrics, "report": report,
                "confusion_matrix": cm, "probs": probs}


def _save_confusion(cm, class_names, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([c.upper() for c in class_names], rotation=45, ha="right")
    ax.set_yticklabels([c.upper() for c in class_names])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    th = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > th else "black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from .data import build_dataframes

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", required=True,
                    help="Path to trained binary .keras model.")
    ap.add_argument("--subtype", required=True,
                    help="Path to trained subtype .keras model.")
    ap.add_argument("--binary-size", type=int, default=260)
    ap.add_argument("--subtype-size", type=int, default=260)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--image", default=None,
                    help="Single image path to predict (otherwise evaluates the test set).")
    args = ap.parse_args()

    clf = TwoStageClassifier(
        binary_model_path=args.binary,
        subtype_model_path=args.subtype,
        binary_img_size=args.binary_size,
        subtype_img_size=args.subtype_size,
        tumor_threshold=args.threshold,
    )

    if args.image:
        result = clf.predict_image(args.image)
        print(json.dumps(result, indent=2, default=str))
        return

    _, _, test_df = build_dataframes(config.CLASSES_4)
    clf.evaluate_dataframe(test_df, save_dir=config.REPORTS_DIR, title="two_stage")


if __name__ == "__main__":
    main()
