"""improvements/train_subtype.py - Stage-2 tumor subtype classifier (3-class).

Trains a 3-class classifier on tumor-only images:
    0 = glioma
    1 = meningioma
    2 = pituitary

Notumor is excluded entirely - this model is invoked AFTER the binary classifier
has determined the image contains a tumor.

Supports EfficientNet-B0/B2/B3 and Xception. Same recipe as train_binary:
EfficientNet feeds raw [0,255]; Xception preprocessing is an in-graph Lambda.

Outputs (per variant):
    artifacts/models/subtype_{variant}{_clean}.keras
    artifacts/reports/subtype_{variant}{_clean}_plain_confusion.png
    artifacts/reports/subtype_{variant}{_clean}_tta_confusion.png
    artifacts/reports/subtype_{variant}{_clean}_summary.json
    artifacts/reports/subtype_comparison.csv  (when --compare)

Run:
    python -m improvements.train_subtype --variant b2
    python -m improvements.train_subtype --compare
    python -m improvements.train_subtype --variant b2 --no-quarantine
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .reproducibility import set_global_seed


SUBTYPE_CLASSES = list(config.CLASSES_3)  # ["glioma", "meningioma", "pituitary"]

_BACKBONES: Dict[str, Tuple[str, int, str]] = {
    "b0":       ("EfficientNetB0", 224, "efficientnet"),
    "b2":       ("EfficientNetB2", 260, "efficientnet"),
    "b3":       ("EfficientNetB3", 300, "efficientnet"),
    "xception": ("Xception",       299, "xception"),
}


def _filter_tumor_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only glioma/meningioma/pituitary rows (drop notumor)."""
    return df[df["Class"].isin(SUBTYPE_CLASSES)].reset_index(drop=True)


def _build_model(variant: str, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import applications, layers
    from tensorflow.keras.models import Model

    name, _, family = _BACKBONES[variant]
    inp = layers.Input((img_size, img_size, 3), name="input")

    if family == "efficientnet":
        backbone_cls = getattr(applications, name)
        base = backbone_cls(include_top=False, weights="imagenet",
                            input_shape=(img_size, img_size, 3))
        base.trainable = False
        x = base(inp, training=False)
    else:
        from tensorflow.keras.applications.xception import (
            preprocess_input as xcep_pre,
        )
        base = applications.Xception(include_top=False, weights="imagenet",
                                     input_shape=(img_size, img_size, 3))
        base.trainable = False
        x = layers.Lambda(xcep_pre, name="xcep_pre")(inp)
        x = base(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.3, name="head_drop1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.25, name="head_drop2")(x)
    out = layers.Dense(3, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name=f"{name}_Subtype")
    return model, base


def _unfreeze_from_block(base, from_block: int, family: str) -> int:
    import tensorflow as tf

    base.trainable = True
    n = 0
    for layer in base.layers:
        m = re.match(r"block(\d+)", layer.name)
        block_no = int(m.group(1)) if m else 0
        train_it = block_no >= from_block
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            train_it = False
        layer.trainable = train_it
        n += int(train_it)
    return n


def _class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    counts = train_df["Class"].value_counts().reindex(SUBTYPE_CLASSES).fillna(0).values
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (3 * counts)
    return {i: float(w[i]) for i in range(3)}


def train(variant: str = "b2", use_quarantine: bool = True,
          phase1_epochs: int = config.PHASE1_EPOCHS,
          phase2_epochs: int = config.PHASE2_EPOCHS,
          verbose: int = 1) -> Tuple[Path, Dict]:
    """Train one subtype backbone end-to-end. Returns (model_path, metrics_dict)."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import Precision, Recall
    from tensorflow.keras.optimizers import Adam

    from .audit import load_quarantine
    from .data import build_dataframes, make_generators
    from .evaluate import evaluate_model

    if variant not in _BACKBONES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_BACKBONES)}.")
    name, img_size, family = _BACKBONES[variant]

    set_global_seed(config.SEED)

    quarantine = load_quarantine() if use_quarantine else set()
    if use_quarantine and quarantine:
        print(f"[subtype] audit quarantine: {len(quarantine)} gliomas excluded from TRAIN.")
    elif use_quarantine:
        print("[subtype] no quarantine list found; training on full data.")

    # Use the 3-class dataframe directly (notumor excluded by build_dataframes).
    train_df, valid_df, test_df = build_dataframes(
        SUBTYPE_CLASSES, quarantine=quarantine
    )
    # Defensive: ensure notumor is not present
    train_df = _filter_tumor_only(train_df)
    valid_df = _filter_tumor_only(valid_df)
    test_df = _filter_tumor_only(test_df)
    print(f"[subtype] sizes  train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    tr_gen, va_gen, ts_gen = make_generators(
        train_df, valid_df, test_df, img_size=img_size, augment=True
    )

    cw = _class_weights(train_df)
    print(f"[subtype] class weights: {cw}")

    model, base = _build_model(variant, img_size)
    loss = CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)

    # ---- Phase 1 -------------------------------------------------------
    model.compile(
        optimizer=Adam(config.PHASE1_LR),
        loss=loss,
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    print(f"\n[subtype] Phase 1 ({name}, {img_size}px) - head warmup")
    model.fit(
        tr_gen, validation_data=va_gen, epochs=phase1_epochs,
        class_weight=cw,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4,
                          restore_best_weights=True, verbose=verbose),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                              min_lr=1e-6, verbose=verbose),
        ],
        verbose=verbose,
    )

    # ---- Phase 2 -------------------------------------------------------
    from_block = config.UNFREEZE_FROM_BLOCK if family == "efficientnet" else 13
    n_unf = _unfreeze_from_block(base, from_block, family)
    print(f"[subtype] Phase 2 - unfroze {n_unf} layers from block{from_block} (BN frozen)")
    model.compile(
        optimizer=Adam(config.PHASE2_LR),
        loss=loss,
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    model.fit(
        tr_gen, validation_data=va_gen, epochs=phase2_epochs,
        class_weight=cw,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, verbose=verbose),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                              min_lr=1e-7, verbose=verbose),
        ],
        verbose=verbose,
    )

    # ---- save + evaluate ----------------------------------------------
    tag = f"subtype_{variant}{'_clean' if quarantine else ''}"
    out_path = config.MODELS_DIR / f"{tag}.keras"
    model.save(out_path)
    print(f"[subtype] saved -> {out_path}")

    res_plain = evaluate_model(model, ts_gen, SUBTYPE_CLASSES, use_tta=False,
                               title=f"{tag}_plain", save_dir=config.REPORTS_DIR)
    res_tta = evaluate_model(model, ts_gen, SUBTYPE_CLASSES, use_tta=True,
                             title=f"{tag}_tta", save_dir=config.REPORTS_DIR)

    summary = {
        "variant": variant, "backbone": name, "img_size": img_size,
        "classes": SUBTYPE_CLASSES, "class_weights": cw,
        "quarantined_gliomas": len(quarantine),
        "test_plain": res_plain["metrics"], "test_tta": res_tta["metrics"],
    }
    with open(config.REPORTS_DIR / f"{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return out_path, summary


def compare_all(use_quarantine: bool = True,
                variants: List[str] = None) -> pd.DataFrame:
    variants = variants or ["b0", "b2", "b3", "xception"]
    rows = []
    for v in variants:
        try:
            path, summary = train(v, use_quarantine=use_quarantine)
            tta = summary["test_tta"]
            rows.append({
                "variant": v, "backbone": summary["backbone"],
                "img_size": summary["img_size"], "model_path": str(path),
                "accuracy_tta": tta["accuracy"], "precision_tta": tta["precision"],
                "recall_tta": tta["recall"], "f1_tta": tta["f1"],
                "roc_auc_tta": tta["roc_auc"],
                "accuracy_plain": summary["test_plain"]["accuracy"],
            })
        except Exception as exc:
            print(f"[subtype] FAILED for {v}: {exc}")
            rows.append({"variant": v, "error": str(exc)})

    df = pd.DataFrame(rows)
    if "accuracy_tta" in df.columns:
        df = df.sort_values("accuracy_tta", ascending=False, na_position="last")
    out = config.REPORTS_DIR / "subtype_comparison.csv"
    df.to_csv(out, index=False)
    print("\n" + "=" * 70)
    print("SUBTYPE CLASSIFIER COMPARISON (test TTA)")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n[subtype] comparison -> {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=list(_BACKBONES), default="b2")
    ap.add_argument("--no-quarantine", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare_all(use_quarantine=not args.no_quarantine)
    else:
        train(variant=args.variant, use_quarantine=not args.no_quarantine)


if __name__ == "__main__":
    main()
