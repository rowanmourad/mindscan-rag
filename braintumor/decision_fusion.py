"""improvements/decision_fusion.py - Decision-level ensemble of multiple models.

Combines predictions from multiple trained models into a single decision via
four fusion methods, then picks the winner on the validation set and reports
the held-out test-set performance of that winner.

Inputs:
    - one or more 4-class .keras models (single-stage, e.g. EfficientNet-B2 or fusion)
    - optional two-stage classifier (binary + subtype combined)

All sources are mapped to a (N, 4) probability matrix in CLASSES_4 order, so
fusion is performed in a shared probability space.

Fusion methods:
    1. weighted_voting         -- each model casts a one-hot vote weighted by w_i
    2. confidence_weighting    -- each model's vote weighted by its max softmax
    3. probability_average     -- weighted average of the softmax probability vectors
    4. stacking                -- LogisticRegression meta-learner on concatenated probs;
                                  trained on the VALIDATION split, evaluated on TEST

The meta-learner is trained on validation predictions (not on train) to avoid
re-using the data the base models were fit on. This is an acknowledged but
honest leakage profile - validation was also used for early stopping, so its
predictions are slightly optimistic but the test set remains untouched.

CLI:
    python -m improvements.decision_fusion \
        --models  artifacts/models/effnet_b2_4c_clean.keras=260 \
                  artifacts/models/fusion_eff_xcep_4c_clean.keras=260 \
        --two-stage artifacts/models/binary_b2_clean.keras=260,artifacts/models/subtype_b2_clean.keras=260
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config


FULL_CLASSES = list(config.CLASSES_4)


# ---------------------------------------------------------------------------
# Source wrappers - each produces a (N, 4) probability matrix
# ---------------------------------------------------------------------------
class _ModelSource:
    """Wraps a 4-class Keras model and predicts (N, 4) probabilities."""

    def __init__(self, path: str | Path, img_size: int, name: Optional[str] = None):
        import tensorflow as tf

        self.path = Path(path)
        self.img_size = int(img_size)
        self.name = name or self.path.stem
        print(f"[fusion] loading 4-class model '{self.name}' (size={self.img_size}): {self.path}")
        self.model = tf.keras.models.load_model(self.path)
        n_out = int(self.model.output_shape[-1])
        if n_out != 4:
            raise ValueError(
                f"Model {self.path} has {n_out} outputs; "
                "decision_fusion expects 4-class models in this slot."
            )

    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        N = len(df)
        out = np.zeros((N, 4), dtype=np.float32)
        for s in range(0, N, batch_size):
            paths = df["Class Path"].iloc[s:s + batch_size].tolist()
            x = np.zeros((len(paths), self.img_size, self.img_size, 3), dtype=np.float32)
            for i, p in enumerate(paths):
                with Image.open(p) as im:
                    im = im.convert("RGB").resize((self.img_size, self.img_size))
                    x[i] = np.asarray(im, dtype=np.float32)
            out[s:s + batch_size] = self.model.predict(x, verbose=0)
        return out


class _TwoStageSource:
    """Wraps a two-stage classifier and predicts (N, 4) probabilities."""

    def __init__(
        self, binary_path: str, subtype_path: str,
        binary_img_size: int, subtype_img_size: int,
        threshold: float = 0.5, name: str = "two_stage",
    ):
        from .two_stage import TwoStageClassifier

        self.name = name
        self.clf = TwoStageClassifier(
            binary_model_path=binary_path,
            subtype_model_path=subtype_path,
            binary_img_size=binary_img_size,
            subtype_img_size=subtype_img_size,
            tumor_threshold=threshold,
        )

    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        return self.clf.predict_dataframe(df, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Fusion methods (all consume a list of (N, 4) probability matrices)
# ---------------------------------------------------------------------------
def fuse_weighted_voting(probs_list: List[np.ndarray],
                         weights: Optional[List[float]] = None) -> np.ndarray:
    """Each model casts a one-hot vote weighted by w_i; return argmax tally as softmax."""
    K = len(probs_list)
    N = probs_list[0].shape[0]
    if weights is None:
        weights = [1.0 / K] * K
    weights = np.asarray(weights, dtype=np.float32) / sum(weights)

    tally = np.zeros((N, 4), dtype=np.float32)
    for w, probs in zip(weights, probs_list):
        votes = np.zeros_like(probs)
        votes[np.arange(N), probs.argmax(axis=1)] = 1.0
        tally += w * votes
    return tally


def fuse_confidence_weighting(probs_list: List[np.ndarray]) -> np.ndarray:
    """Each model's vote weighted by its max softmax confidence on that sample."""
    N = probs_list[0].shape[0]
    tally = np.zeros((N, 4), dtype=np.float32)
    total_w = np.zeros(N, dtype=np.float32)
    for probs in probs_list:
        conf = probs.max(axis=1, keepdims=True)
        votes = np.zeros_like(probs)
        votes[np.arange(N), probs.argmax(axis=1)] = 1.0
        tally += conf * votes
        total_w += conf.squeeze(-1)
    return tally / np.maximum(total_w[:, None], 1e-12)


def fuse_probability_average(probs_list: List[np.ndarray],
                             weights: Optional[List[float]] = None) -> np.ndarray:
    """Weighted average of softmax probability vectors (the standard ensemble)."""
    K = len(probs_list)
    if weights is None:
        weights = [1.0 / K] * K
    weights = np.asarray(weights, dtype=np.float32) / sum(weights)
    avg = np.zeros_like(probs_list[0])
    for w, probs in zip(weights, probs_list):
        avg += w * probs
    return avg


def fuse_stacking(
    val_probs_list: List[np.ndarray], val_y: np.ndarray,
    test_probs_list: List[np.ndarray],
) -> np.ndarray:
    """Train a LogisticRegression meta-learner on validation, predict on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_val = np.concatenate(val_probs_list, axis=1)   # (N, 4 * K)
    X_test = np.concatenate(test_probs_list, axis=1)

    meta = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, multi_class="multinomial",
                                  C=1.0, random_state=config.SEED)),
    ])
    meta.fit(X_val, val_y)
    return meta.predict_proba(X_test)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class DecisionFusion:
    METHODS = ("weighted_voting", "confidence_weighting", "probability_average", "stacking")

    def __init__(self, sources: List):
        if len(sources) < 2:
            raise ValueError("Decision fusion needs at least 2 sources.")
        self.sources = sources

    def _predict_all(self, df: pd.DataFrame) -> List[np.ndarray]:
        return [s.predict(df) for s in self.sources]

    def compare_and_pick_best(
        self, valid_df: pd.DataFrame, test_df: pd.DataFrame,
        weights: Optional[List[float]] = None,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """Run all four methods. Pick the winner by VAL accuracy; report TEST."""
        from .evaluate import compute_metrics
        from sklearn.metrics import classification_report, confusion_matrix

        save_dir = Path(save_dir) if save_dir else config.REPORTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        print("[fusion] predicting on validation ...")
        val_probs_list = self._predict_all(valid_df)
        val_y = valid_df["Class"].map(lambda c: FULL_CLASSES.index(c)).to_numpy()

        print("[fusion] predicting on test ...")
        test_probs_list = self._predict_all(test_df)
        test_y = test_df["Class"].map(lambda c: FULL_CLASSES.index(c)).to_numpy()

        results: Dict[str, Dict] = {}

        # 1. Weighted voting (uniform weights = simple majority)
        val_p = fuse_weighted_voting(val_probs_list, weights)
        test_p = fuse_weighted_voting(test_probs_list, weights)
        results["weighted_voting"] = self._record(val_y, val_p, test_y, test_p)

        # 2. Confidence weighting
        val_p = fuse_confidence_weighting(val_probs_list)
        test_p = fuse_confidence_weighting(test_probs_list)
        results["confidence_weighting"] = self._record(val_y, val_p, test_y, test_p)

        # 3. Probability average (uniform)
        val_p = fuse_probability_average(val_probs_list, weights)
        test_p = fuse_probability_average(test_probs_list, weights)
        results["probability_average"] = self._record(val_y, val_p, test_y, test_p)

        # 4. Stacking
        try:
            test_p = fuse_stacking(val_probs_list, val_y, test_probs_list)
            # For consistency, also evaluate stacking "validation" by 5-fold CV on val
            # (cheap; uses the same meta-learner pipeline)
            from sklearn.model_selection import cross_val_predict
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            meta = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, multi_class="multinomial",
                                          C=1.0, random_state=config.SEED)),
            ])
            X_val = np.concatenate(val_probs_list, axis=1)
            val_p = cross_val_predict(meta, X_val, val_y, cv=5, method="predict_proba")
            results["stacking"] = self._record(val_y, val_p, test_y, test_p)
        except Exception as exc:
            print(f"[fusion] stacking failed: {exc}")
            results["stacking"] = {"error": str(exc)}

        # Per-source baselines (individual model TTA-free numbers, for context)
        for src, probs in zip(self.sources, test_probs_list):
            y_pred = probs.argmax(axis=1)
            m = compute_metrics(test_y, y_pred, probs, FULL_CLASSES)
            results[f"_source_{src.name}"] = {"test": m}

        # Pick the best fusion method by VAL accuracy
        scored = [
            (name, r["val"]["accuracy"])
            for name, r in results.items()
            if name in self.METHODS and isinstance(r, dict) and "val" in r
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_name = scored[0][0] if scored else None
        if best_name:
            print(f"\n[fusion] BEST METHOD (by validation acc): {best_name}")
            print(f"         val acc = {results[best_name]['val']['accuracy']*100:.2f}%")
            print(f"         test acc = {results[best_name]['test']['accuracy']*100:.2f}%")

            # Detailed test-set report for the winner
            cm = confusion_matrix(test_y, results[best_name]["test_y_pred"],
                                  labels=list(range(4)))
            report = classification_report(
                test_y, results[best_name]["test_y_pred"],
                target_names=FULL_CLASSES, zero_division=0,
            )
            print("\n[fusion] winner test confusion + report:")
            print(report)
            _save_confusion(cm, FULL_CLASSES,
                            save_dir / f"decision_fusion_{best_name}_confusion.png",
                            f"Decision Fusion - {best_name}")

        summary = {
            "sources": [s.name for s in self.sources],
            "winner": best_name,
            "methods": {name: {k: v for k, v in r.items() if k not in ("test_y_pred",)}
                        for name, r in results.items()},
        }
        with open(save_dir / "decision_fusion_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"[fusion] summary -> {save_dir / 'decision_fusion_summary.json'}")
        return summary

    @staticmethod
    def _record(val_y, val_p, test_y, test_p) -> Dict:
        from .evaluate import compute_metrics
        val_pred = val_p.argmax(axis=1)
        test_pred = test_p.argmax(axis=1)
        return {
            "val":  compute_metrics(val_y, val_pred, val_p, FULL_CLASSES),
            "test": compute_metrics(test_y, test_pred, test_p, FULL_CLASSES),
            "test_y_pred": test_pred,
        }


# ---------------------------------------------------------------------------
# Confusion-matrix helper (kept local to avoid coupling on plotting)
# ---------------------------------------------------------------------------
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
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    th = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > th else "black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_model_spec(s: str) -> Tuple[str, int]:
    """Parse 'path=size' specifications from the CLI."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Model spec must be 'path=imgsize' (got {s!r})."
        )
    path, size = s.rsplit("=", 1)
    return path, int(size)


def main() -> None:
    from .data import build_dataframes

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", type=_parse_model_spec, default=[],
                    help="One or more 4-class model specs: 'path=imgsize'")
    ap.add_argument("--two-stage", default=None,
                    help="Two-stage spec: 'bin_path=size,sub_path=size'")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    sources: List = []
    for path, size in args.models:
        sources.append(_ModelSource(path=path, img_size=size))

    if args.two_stage:
        bin_part, sub_part = args.two_stage.split(",")
        bin_path, bin_size = _parse_model_spec(bin_part)
        sub_path, sub_size = _parse_model_spec(sub_part)
        sources.append(_TwoStageSource(
            binary_path=bin_path, subtype_path=sub_path,
            binary_img_size=bin_size, subtype_img_size=sub_size,
            threshold=args.threshold,
        ))

    if len(sources) < 2:
        raise SystemExit("Need at least 2 sources for decision fusion "
                         "(--models + optionally --two-stage).")

    _, valid_df, test_df = build_dataframes(config.CLASSES_4)
    fusion = DecisionFusion(sources)
    fusion.compare_and_pick_best(valid_df, test_df, save_dir=config.REPORTS_DIR)


if __name__ == "__main__":
    main()
