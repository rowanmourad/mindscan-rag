"""Non-destructive SARTAJ glioma label-noise audit (H2).

The dominant error across ALL your models is glioma -> meningioma
(glioma recall ~0.68 on a perfectly class-balanced dataset). On balanced
data that is the signature of *label noise*, not imbalance - the well-known
SARTAJ glioma contamination in the Figshare+SARTAJ+Br35H union.

This module produces a QUARANTINE LIST of suspect glioma training images so
you can (a) eyeball them and (b) exclude them from training via
``data.build_dataframes(quarantine=...)``. It does the following and NOTHING
else:

  * loads an already-trained EfficientNet model (your saved .keras),
  * scores every glioma-folder TRAINING image,
  * flags those predicted as meningioma with confidence > threshold,
  * writes ``artifacts/audit/glioma_quarantine.csv`` + a visual review grid.

It never deletes, moves, or relabels any file on disk. Removing flagged
images from TRAIN only (never val/test) keeps the evaluation honest.

Usage:
    python -m improvements.audit
    python -m improvements.audit --model "stadge2/brain_tumor_efficientnet.keras"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

from . import config
from .reproducibility import set_global_seed


def _default_model_path() -> Path:
    """Prefer the 4-class EfficientNet (it can see meningioma vs glioma)."""
    candidates = [
        config.ROOT / "stadge2" / "brain_tumor_efficientnet.keras",
        config.ROOT / "stadge2" / "brain_tumor_efficientnet_3class.keras",
        config.ROOT / "stadge2" / "brain_tumor_model.keras",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "No trained .keras model found in stadge2/. Train one first or pass "
        "--model."
    )


def load_quarantine(path: Path | None = None) -> Set[str]:
    """Load a previously-saved quarantine CSV into a set of file paths.

    Returns an empty set if the file does not exist, so callers can do
    ``build_dataframes(quarantine=load_quarantine())`` unconditionally.
    """
    path = path or (config.AUDIT_DIR / "glioma_quarantine.csv")
    if not Path(path).exists():
        return set()
    df = pd.read_csv(path)
    return {str(Path(p)) for p in df["Class Path"].tolist()}


def run_audit(
    model_path: Path,
    flag_conf: float = config.AUDIT_FLAG_CONF,
    img_size: int = config.IMG_SIZE["b0"],
    save_grid: bool = True,
) -> pd.DataFrame:
    """Score glioma training images and write a quarantine list.

    Returns the flagged dataframe (also written to CSV).
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    set_global_seed(config.SEED)

    print(f"[audit] loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Determine the model's class order from its output width.
    n_out = int(model.output_shape[-1])
    classes = config.CLASSES_4 if n_out == 4 else config.CLASSES_3
    if "meningioma" not in classes:
        raise ValueError("Model has no meningioma output; cannot audit glioma noise.")
    glioma_idx = classes.index("glioma")
    mening_idx = classes.index("meningioma")

    # Build a glioma-only TRAINING dataframe (shuffle off to align rows).
    from .data import _scan
    train_df = _scan(config.TRAIN_DIR, classes)
    glioma_df = train_df[train_df["Class"] == "glioma"].reset_index(drop=True)
    print(f"[audit] glioma training images: {len(glioma_df)}")

    eval_datagen = ImageDataGenerator()  # raw [0,255], no aug
    gen = eval_datagen.flow_from_dataframe(
        glioma_df, x_col="Class Path", y_col="Class",
        target_size=(img_size, img_size), class_mode=None,
        batch_size=32, shuffle=False,
    )

    probs = model.predict(gen, verbose=1)
    pred_idx = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    mening_conf = probs[:, mening_idx]

    glioma_df = glioma_df.iloc[: len(probs)].copy()
    glioma_df["pred_idx"] = pred_idx
    glioma_df["pred_class"] = [classes[i] for i in pred_idx]
    glioma_df["pred_conf"] = pred_conf
    glioma_df["meningioma_conf"] = mening_conf

    # Self-confusion summary (the model on its own training data).
    pct_as_mening = 100.0 * (glioma_df["pred_class"] == "meningioma").mean()
    print(f"[audit] glioma-folder images predicted MENINGIOMA: {pct_as_mening:.1f}%")
    print("        interpretation:  <2% clean | 2-10% some noise | >10% SARTAJ contamination")

    flagged = glioma_df[
        (glioma_df["pred_class"] == "meningioma")
        & (glioma_df["meningioma_conf"] >= flag_conf)
    ].sort_values("meningioma_conf", ascending=False).reset_index(drop=True)

    out_csv = config.AUDIT_DIR / "glioma_quarantine.csv"
    flagged[["Class Path", "Class", "pred_class", "meningioma_conf"]].to_csv(
        out_csv, index=False
    )
    print(f"[audit] flagged {len(flagged)} suspect gliomas "
          f"(meningioma conf >= {flag_conf}) -> {out_csv}")
    print("        NOTE: nothing deleted. Review the grid, then train with "
          "data.build_dataframes(quarantine=audit.load_quarantine()).")

    # Also save the full per-image scores for transparency / thesis appendix.
    glioma_df.to_csv(config.AUDIT_DIR / "glioma_scores_full.csv", index=False)

    if save_grid and len(flagged) > 0:
        _save_review_grid(flagged.head(16), config.AUDIT_DIR / "suspect_gliomas.png")

    return flagged


def _save_review_grid(df: pd.DataFrame, out_path: Path, cols: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(df)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, (_, r) in zip(axes, df.iterrows()):
        try:
            ax.imshow(Image.open(r["Class Path"]).convert("RGB"))
        except Exception as exc:
            ax.text(0.5, 0.5, f"load error\n{exc}", ha="center", va="center")
        ax.set_title(f"men_conf={r['meningioma_conf']:.2f}", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "Suspect gliomas (model says MENINGIOMA, high conf)\n"
        "Round/peripheral/dural-attached => likely mislabeled",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[audit] review grid -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="Path to a trained .keras model.")
    ap.add_argument("--flag-conf", type=float, default=config.AUDIT_FLAG_CONF)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE["b0"])
    args = ap.parse_args()

    model_path = Path(args.model) if args.model else _default_model_path()
    run_audit(model_path, flag_conf=args.flag_conf, img_size=args.img_size)


if __name__ == "__main__":
    main()
