"""Honest evaluation harness + Test-Time Augmentation (H1, H4).

Provides a single ``evaluate_model`` that reports the full metric suite
(accuracy, macro precision/recall/F1, per-class report, confusion matrix,
macro ROC-AUC) on a held-out generator, with optional TTA.

Design goals:
  * Honest: evaluates on the untouched val/test split from ``data.py``.
    The split is never filtered by the audit quarantine, so a recall gain
    here reflects real generalization, not removal of hard test cases.
  * TTA (H4): averages softmax over a small set of label-preserving
    augmentations (horizontal flip + a couple of small zoom/shift crops).
    Free accuracy at inference; cannot leak because it touches only the
    input image, never the label or the train set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    avg = "binary" if len(class_names) == 2 else "macro"
    m: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    try:
        if len(class_names) == 2:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            m["roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


# --------------------------------------------------------------------------
# Plain prediction over a (shuffle=False) generator
# --------------------------------------------------------------------------
def _predict_generator(model, gen) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probs, y_true) preserving generator order (shuffle must be off)."""
    probs = model.predict(gen, verbose=0)
    return probs, np.asarray(gen.classes)


# --------------------------------------------------------------------------
# Test-Time Augmentation (H4)
# --------------------------------------------------------------------------
def _tta_predict(model, gen) -> np.ndarray:
    """Average softmax over label-preserving augmentations.

    We pull the raw batches from the generator (which yields EfficientNet-ready
    [0,255] floats) and apply numpy-level horizontal flip + small shifts. All
    are geometry-preserving w.r.t. the label.
    """
    import numpy as np

    n = gen.samples
    n_classes = len(gen.class_indices)
    acc = np.zeros((n, n_classes), dtype=np.float64)

    # View 1: identity. Views 2..: hflip and small +/- pixel shifts.
    shifts = [(0, 0), (0, 0), (0, 6), (0, -6), (6, 0), (-6, 0)]
    flips = [False, True, True, False, False, True]

    gen.reset()
    # Reconstruct each image batch once, then derive all TTA views from it.
    batches = []
    seen = 0
    while seen < n:
        x, _ = next(gen)
        batches.append(x)
        seen += x.shape[0]

    for (dy, dx), do_flip in zip(shifts, flips):
        row = 0
        for x in batches:
            xv = x
            if do_flip:
                xv = xv[:, :, ::-1, :]
            if dy or dx:
                xv = np.roll(xv, shift=(dy, dx), axis=(1, 2))
            p = model.predict(xv, verbose=0)
            acc[row:row + p.shape[0]] += p
            row += p.shape[0]
    acc /= len(shifts)
    return acc.astype(np.float32)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def evaluate_model(
    model,
    gen,
    class_names: List[str],
    *,
    use_tta: bool = config.TTA_ENABLED,
    title: str = "model",
    save_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Evaluate ``model`` on ``gen`` and (optionally) print + save artifacts.

    Returns a dict with metrics, the sklearn classification_report string,
    and the confusion matrix. ``gen`` must have ``shuffle=False``.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    if use_tta:
        probs = _tta_predict(model, gen)
        y_true = np.asarray(gen.classes)
    else:
        probs, y_true = _predict_generator(model, gen)

    y_pred = probs.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred, probs, class_names)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print("=" * 60)
    print(f" EVALUATION: {title}  ({'TTA' if use_tta else 'plain'})")
    print("=" * 60)
    print(f"  accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  macro F1 : {metrics['f1']*100:.2f}%")
    print(f"  macro AUC: {metrics['roc_auc']:.4f}")
    print(report)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_confusion_matrix(cm, class_names, save_dir / f"{title}_confusion.png",
                                  title=title)
        except Exception as exc:  # plotting is optional, never fatal
            print(f"  [warn] confusion-matrix plot skipped: {exc}")

    return {"metrics": metrics, "report": report, "confusion_matrix": cm}


# --------------------------------------------------------------------------
# Self-contained confusion-matrix plot (no external package dependency)
# --------------------------------------------------------------------------
def save_confusion_matrix(cm, class_names: List[str], out_path: Path,
                          title: str = "confusion matrix") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([c.upper() for c in class_names], rotation=45, ha="right")
    ax.set_yticklabels([c.upper() for c in class_names])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    th = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > th else "black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
