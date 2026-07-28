"""Leakage-aware, reproducible data handling (H1).

Reuses the EXACT same on-disk dataset as the notebook
(``Datasets/archive (2)/{Training,Testing}``) and the same split logic
(Training -> train; Testing -> 50/50 val/test, stratified, seeded), but adds:

  * a fixed seed so the split is identical every run (H1),
  * an optional ``quarantine`` set of file paths to exclude from TRAIN ONLY
    (used by the label-noise audit, H2 - never deletes anything),
  * EfficientNet-correct preprocessing (raw [0,255], no /255).

Why a separate module instead of editing the notebook generators?
So the original 90% pipeline stays runnable and we can compare apples to
apples. Nothing here writes to the dataset folders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _scan(root: Path, classes: Iterable[str]) -> pd.DataFrame:
    """Build a (Class Path, Class) dataframe from a class-folder tree."""
    classes = set(classes)
    rows = []
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in classes:
            continue
        for img in sorted(label_dir.iterdir()):
            if img.suffix.lower() in _IMG_EXTS:
                rows.append({"Class Path": str(img), "Class": label_dir.name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root} for {classes}.")
    return df


def build_dataframes(
    classes: Optional[List[str]] = None,
    quarantine: Optional[Set[str]] = None,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, valid_df, test_df) reproducibly.

    Parameters
    ----------
    classes : list[str] | None
        Class folders to include. Defaults to the 4-class set. Pass
        ``config.CLASSES_3`` for the tumor-only variant.
    quarantine : set[str] | None
        Absolute file paths to drop **from the training set only**. This is
        how the audit (H2) removes flagged mislabeled gliomas without
        deleting them from disk. Val/test are never filtered, so the
        evaluation remains an honest test of generalization.
    seed : int
        Controls the stratified 50/50 val/test split of the Testing folder.
        Matches the notebook's intent but is now fixed for reproducibility.
    """
    classes = classes or config.CLASSES_4
    quarantine = quarantine or set()
    quarantine = {str(Path(p)) for p in quarantine}

    train_df = _scan(config.TRAIN_DIR, classes)
    test_pool = _scan(config.TEST_DIR, classes)

    if quarantine:
        before = len(train_df)
        train_df = train_df[~train_df["Class Path"].isin(quarantine)].reset_index(drop=True)
        removed = before - len(train_df)
        print(f"[data] quarantined {removed} flagged training images (kept on disk).")

    valid_df, test_df = train_test_split(
        test_pool,
        train_size=0.5,
        random_state=seed,
        stratify=test_pool["Class"],
    )
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, valid_df, test_df


def make_generators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_size: int,
    batch_size: int = config.BATCH_SIZE,
    augment: bool = True,
):
    """Build Keras generators with EfficientNet-correct preprocessing.

    Returns (tr_gen, va_gen, ts_gen). Key correctness points vs. the
    original notebook:

      * NO ``rescale=1/255`` - EfficientNet normalizes internally (kept).
      * Validation/test use an eval generator with NO augmentation (fixes the
        Xception-section bug where val used the brightness-augmenting datagen).
      * Stronger but MRI-safe augmentation on train (H5): brightness, mild
        rotation, width/height shift, zoom. NO vertical flip (anatomically
        invalid); horizontal flip is acceptable for axial brain MRI.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if augment:
        train_datagen = ImageDataGenerator(
            brightness_range=(0.8, 1.2),
            rotation_range=15,
            width_shift_range=0.08,
            height_shift_range=0.08,
            zoom_range=0.10,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator()
    eval_datagen = ImageDataGenerator()  # deterministic: no aug, no rescale

    common = dict(
        x_col="Class Path", y_col="Class",
        target_size=(img_size, img_size), class_mode="categorical",
    )
    tr_gen = train_datagen.flow_from_dataframe(
        train_df, batch_size=batch_size, shuffle=True, seed=config.SEED, **common
    )
    va_gen = eval_datagen.flow_from_dataframe(
        valid_df, batch_size=batch_size, shuffle=False, **common
    )
    ts_gen = eval_datagen.flow_from_dataframe(
        test_df, batch_size=16, shuffle=False, **common
    )
    return tr_gen, va_gen, ts_gen
