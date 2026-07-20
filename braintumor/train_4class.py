"""Upgraded EfficientNet training recipe (H3).

Improvements over the notebook's EfficientNetB0 recipe, each tied to the
advisory report:

  H3  bigger backbone: EfficientNet-B2 at native 260px (more capacity for the
      hard glioma/meningioma boundary) - switchable via ``--variant``.
  H9  deeper, block-aware gradual unfreeze: unfreeze from block5 onward
      instead of a flat "last 20 layers" (which on B0 was barely the head).
  H4  class weighting: even though the set is balanced, after quarantining
      noisy gliomas the glioma class shrinks; class weights keep it from being
      under-represented.
  H7  label smoothing (0.05): regularizes against residual label noise.
  H5  MRI-safe augmentation (in data.make_generators): rotation/shift/zoom/
      hflip, NO vertical flip.
  H2  optional quarantine: drop audit-flagged gliomas from TRAIN only.

BatchNorm layers in the backbone are kept frozen during fine-tuning (kept
from the original, correct best practice with small batches).

Nothing here overwrites the existing .keras files; output goes to
``improvements/artifacts/models/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .reproducibility import set_global_seed


_BACKBONES = {
    "b0": "EfficientNetB0",
    "b2": "EfficientNetB2",
    "b3": "EfficientNetB3",
}


def _build_model(variant: str, n_classes: int, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import applications, layers
    from tensorflow.keras.models import Model

    backbone_cls = getattr(applications, _BACKBONES[variant])
    base = backbone_cls(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False  # Phase 1: head only

    inputs = layers.Input((img_size, img_size, 3), name="input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.3, name="head_drop1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense1")(x)
    x = layers.Dropout(0.25, name="head_drop2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name=f"{_BACKBONES[variant]}_BrainTumor")
    return model, base


def _class_weights(train_df, classes: List[str]) -> Dict[int, float]:
    counts = train_df["Class"].value_counts().reindex(classes).fillna(0).values
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (len(classes) * counts)
    return {i: float(w[i]) for i in range(len(classes))}


def _unfreeze_from_block(base, from_block: int) -> int:
    """Unfreeze layers in blocks >= from_block; keep all BN frozen. Returns count."""
    import tensorflow as tf
    import re

    base.trainable = True
    n_trainable = 0
    for layer in base.layers:
        # EfficientNet layer names look like "block6a_..."; extract the number.
        m = re.match(r"block(\d+)", layer.name)
        block_no = int(m.group(1)) if m else 0
        train_it = block_no >= from_block
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            train_it = False  # BN stays frozen during fine-tuning
        layer.trainable = train_it
        if train_it:
            n_trainable += 1
    return n_trainable


def train(
    variant: str = config.DEFAULT_VARIANT,
    classes: List[str] | None = None,
    use_quarantine: bool = True,
) -> Path:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import Precision, Recall
    from tensorflow.keras.optimizers import Adam

    set_global_seed(config.SEED)
    classes = classes or config.CLASSES_4
    img_size = config.IMG_SIZE[variant]

    # --- data (with optional quarantine of flagged gliomas) ----------------
    from .data import build_dataframes, make_generators
    quarantine = set()
    if use_quarantine:
        from .audit import load_quarantine
        quarantine = load_quarantine()
        if quarantine:
            print(f"[train] applying audit quarantine: {len(quarantine)} gliomas excluded from TRAIN.")
        else:
            print("[train] no quarantine list found; training on full data. "
                  "Run `python -m improvements.audit` first for the data-cleaning gain.")

    train_df, valid_df, test_df = build_dataframes(classes, quarantine=quarantine)
    tr_gen, va_gen, ts_gen = make_generators(
        train_df, valid_df, test_df, img_size=img_size, augment=True
    )

    cw = _class_weights(train_df, classes)
    print(f"[train] class weights: {cw}")

    # --- model -------------------------------------------------------------
    model, base = _build_model(variant, len(classes), img_size)
    loss = CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)

    # --- Phase 1: frozen backbone, head warmup -----------------------------
    model.compile(
        optimizer=Adam(config.PHASE1_LR), loss=loss,
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    cbs = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    print(f"\n[train] Phase 1 ({_BACKBONES[variant]}, {img_size}px) - head warmup")
    model.fit(tr_gen, validation_data=va_gen, epochs=config.PHASE1_EPOCHS,
              class_weight=cw, callbacks=cbs, verbose=1)

    # --- Phase 2: deeper gradual unfreeze ---------------------------------
    n_unf = _unfreeze_from_block(base, config.UNFREEZE_FROM_BLOCK)
    print(f"[train] Phase 2 - unfroze {n_unf} layers from block{config.UNFREEZE_FROM_BLOCK} "
          f"(BN frozen)")
    model.compile(
        optimizer=Adam(config.PHASE2_LR), loss=loss,
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )
    cbs2 = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    model.fit(tr_gen, validation_data=va_gen, epochs=config.PHASE2_EPOCHS,
              class_weight=cw, callbacks=cbs2, verbose=1)

    # --- save (never overwrites the notebook's files) ----------------------
    tag = f"effnet_{variant}_{len(classes)}c{'_clean' if quarantine else ''}"
    out_path = config.MODELS_DIR / f"{tag}.keras"
    model.save(out_path)
    print(f"[train] saved -> {out_path}")

    # --- honest evaluation (plain + TTA) on the untouched test split -------
    from .evaluate import evaluate_model
    res_plain = evaluate_model(model, ts_gen, classes, use_tta=False,
                               title=f"{tag}_plain", save_dir=config.REPORTS_DIR)
    res_tta = evaluate_model(model, ts_gen, classes, use_tta=True,
                             title=f"{tag}_tta", save_dir=config.REPORTS_DIR)

    summary = {
        "variant": variant, "classes": classes, "img_size": img_size,
        "quarantined_gliomas": len(quarantine),
        "test_plain": res_plain["metrics"], "test_tta": res_tta["metrics"],
    }
    with open(config.REPORTS_DIR / f"{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[train] summary -> {config.REPORTS_DIR / (tag + '_summary.json')}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=list(_BACKBONES), default=config.DEFAULT_VARIANT)
    ap.add_argument("--classes", choices=["4", "3"], default="4")
    ap.add_argument("--no-quarantine", action="store_true",
                    help="Train on full data without excluding flagged gliomas.")
    args = ap.parse_args()
    classes = config.CLASSES_4 if args.classes == "4" else config.CLASSES_3
    train(variant=args.variant, classes=classes, use_quarantine=not args.no_quarantine)


if __name__ == "__main__":
    main()
