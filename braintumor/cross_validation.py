"""5-fold stratified cross-validation + statistical significance (M2).

Matches the rigor of the reference paper's Table 5: per-fold accuracy /
precision / recall / F1, Shapiro-Wilk normality test on the fold-wise metric,
and a paired comparison (paired t-test if normal, Wilcoxon signed-rank
otherwise) between two model variants - e.g. baseline EfficientNet vs. the
upgraded recipe, or single-backbone vs. fusion.

Honesty / leakage notes (important for the writeup):
  * CV here is IMAGE-LEVEL stratified (the dataset has no patient IDs), so
    adjacent slices of one patient may fall in different folds. This inflates
    CV scores vs. a true patient-level split. We print this caveat with every
    run; report it in the thesis.
  * The audit quarantine is applied to each fold's TRAIN portion only; the
    held-out fold is never filtered, so per-fold scores remain honest tests.

Because 5x full training is expensive, CV defaults to EfficientNet-B0 and a
reduced epoch budget. Use it to establish robustness + significance, then
train the final B2/fusion model once with the full recipe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .reproducibility import set_global_seed


def _pooled_dataframe(classes: List[str]) -> pd.DataFrame:
    """All labeled images (Training + Testing folders) for CV folding."""
    from .data import _scan
    tr = _scan(config.TRAIN_DIR, classes)
    te = _scan(config.TEST_DIR, classes)
    return pd.concat([tr, te], ignore_index=True)


def _build_b0(n_classes: int, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import applications, layers
    from tensorflow.keras.models import Model

    base = applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    inp = layers.Input((img_size, img_size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out), base


def run_cv(
    classes: List[str] | None = None,
    n_folds: int = 5,
    epochs: int = 10,
    use_quarantine: bool = True,
    label_smoothing: float = config.LABEL_SMOOTHING,
) -> pd.DataFrame:
    """Run K-fold CV; return a per-fold metrics dataframe (also saved)."""
    import tensorflow as tf
    from sklearn.model_selection import StratifiedKFold
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    set_global_seed(config.SEED)
    classes = classes or config.CLASSES_4
    img_size = config.IMG_SIZE["b0"]

    print("[cv] NOTE: image-level stratified CV (no patient IDs) - scores are "
          "optimistic vs. a patient-level split. Report this caveat.")

    df = _pooled_dataframe(classes)
    quarantine = set()
    if use_quarantine:
        from .audit import load_quarantine
        quarantine = load_quarantine()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    rows = []

    aug = ImageDataGenerator(
        brightness_range=(0.8, 1.2), rotation_range=15,
        width_shift_range=0.08, height_shift_range=0.08,
        zoom_range=0.10, horizontal_flip=True, fill_mode="nearest",
    )
    ev = ImageDataGenerator()
    loss = CategoricalCrossentropy(label_smoothing=label_smoothing)

    for fold, (tr_i, va_i) in enumerate(skf.split(df["Class Path"], df["Class"]), 1):
        tr_df = df.iloc[tr_i].reset_index(drop=True)
        va_df = df.iloc[va_i].reset_index(drop=True)
        if quarantine:                       # clean TRAIN portion only
            tr_df = tr_df[~tr_df["Class Path"].isin(quarantine)].reset_index(drop=True)

        common = dict(x_col="Class Path", y_col="Class",
                      target_size=(img_size, img_size), class_mode="categorical")
        tr_gen = aug.flow_from_dataframe(tr_df, batch_size=config.BATCH_SIZE,
                                         shuffle=True, seed=config.SEED, **common)
        va_gen = ev.flow_from_dataframe(va_df, batch_size=config.BATCH_SIZE,
                                        shuffle=False, **common)

        model, base = _build_b0(len(classes), img_size)
        model.compile(optimizer=Adam(config.PHASE1_LR), loss=loss, metrics=["accuracy"])
        model.fit(tr_gen, validation_data=va_gen, epochs=epochs,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=3,
                                           restore_best_weights=True)],
                  verbose=0)

        from .evaluate import compute_metrics
        probs = model.predict(va_gen, verbose=0)
        y_true = np.asarray(va_gen.classes)
        y_pred = probs.argmax(axis=1)
        m = compute_metrics(y_true, y_pred, probs, classes)
        m["fold"] = fold
        rows.append(m)
        print(f"[cv] fold {fold}/{n_folds}  acc={m['accuracy']*100:.2f}%  "
              f"f1={m['f1']*100:.2f}%")
        tf.keras.backend.clear_session()

    cv_df = pd.DataFrame(rows)[["fold", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    out = config.REPORTS_DIR / "cv_results.csv"
    cv_df.to_csv(out, index=False)

    summary = {c: {"mean": float(cv_df[c].mean()), "std": float(cv_df[c].std())}
               for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    print("\n[cv] summary (mean +/- std):")
    for c, s in summary.items():
        print(f"     {c:10s} {s['mean']*100:6.2f}% +/- {s['std']*100:.2f}")
    with open(config.REPORTS_DIR / "cv_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[cv] saved -> {out}")
    return cv_df


def compare_variants(scores_a: List[float], scores_b: List[float],
                     name_a: str = "A", name_b: str = "B") -> Dict[str, object]:
    """Paired significance test between two variants' fold-wise scores.

    Shapiro-Wilk normality on the paired differences; paired t-test if normal,
    Wilcoxon signed-rank otherwise. Reports effect size (Cohen's d / matched r).
    """
    from scipy import stats

    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    diff = a - b
    sw_stat, sw_p = stats.shapiro(diff) if len(diff) >= 3 else (np.nan, np.nan)
    normal = (sw_p >= 0.05) if not np.isnan(sw_p) else False

    if normal:
        t, p = stats.ttest_rel(a, b)
        d = diff.mean() / (diff.std(ddof=1) + 1e-12)   # Cohen's d (paired)
        test, stat, effect = "paired t-test", float(t), float(d)
        effect_name = "cohen_d"
    else:
        try:
            w, p = stats.wilcoxon(a, b)
        except ValueError:               # all-zero diffs
            w, p = np.nan, 1.0
        z = 0.0
        r = abs(z) / np.sqrt(len(a)) if len(a) else 0.0
        test, stat, effect = "wilcoxon", float(w), float(r)
        effect_name = "matched_rank_r"

    result = {
        "name_a": name_a, "name_b": name_b,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "shapiro_p": float(sw_p), "normal": bool(normal),
        "test": test, "statistic": stat, "p_value": float(p),
        effect_name: effect, "significant_0.05": bool(p < 0.05),
    }
    print(f"[cv] {name_a} ({a.mean()*100:.2f}%) vs {name_b} ({b.mean()*100:.2f}%): "
          f"{test} p={p:.4f} {'(significant)' if p < 0.05 else '(n.s.)'}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", choices=["4", "3"], default="4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--no-quarantine", action="store_true")
    args = ap.parse_args()
    classes = config.CLASSES_4 if args.classes == "4" else config.CLASSES_3
    run_cv(classes=classes, n_folds=args.folds, epochs=args.epochs,
           use_quarantine=not args.no_quarantine)


if __name__ == "__main__":
    main()
