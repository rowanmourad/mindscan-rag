"""Dual-backbone late-fusion model (M1).

Implements the paper's core design: two complementary ImageNet backbones,
each global-average-pooled, concatenated, then classified. The reference paper
(Adamu et al. 2026) fuses MobileNetV2 + EfficientNetV2B0 and puts a KNN head on
the fused vector. We use EfficientNet-B2 + Xception with a trainable softmax
head instead, because:

  * a well-trained softmax head matches KNN accuracy on fused features (KNN was
    chosen there for interpretability/simplicity, not as an accuracy lever), and
  * a softmax head plugs straight into our Grad-CAM / TTA / report pipeline.

Preprocessing subtlety handled internally: EfficientNet expects raw [0,255]
while Xception expects [-1,1]. A single raw-[0,255] input feeds both; each
branch applies its own ``preprocess_input`` as an in-graph layer. So callers
use the SAME data generators as the single-backbone path (no /255).

Two-phase training (frozen -> gradual unfreeze of both backbones), class
weighting, and label smoothing, consistent with train_efficientnet.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np

from . import config
from .reproducibility import set_global_seed


# ---------------------------------------------------------------------------
# Serializable preprocessing wrappers.
#
# The original fusion model wrapped the raw ``efficientnet.preprocess_input`` /
# ``xception.preprocess_input`` functions in bare Lambda layers. Keras 3 cannot
# deserialize those (both serialize to the ambiguous name 'preprocess_input'),
# which is why the saved file would not reload. These named, registered wrappers
# make any NEW fusion model round-trip cleanly via ``load_model``; the OLD file
# is handled by ``load_fusion_model`` (rebuild + load_weights) below.
# ---------------------------------------------------------------------------
def _register():
    import keras

    @keras.saving.register_keras_serializable(package="braintumor", name="eff_preprocess")
    def eff_preprocess(x):
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input(x)

    @keras.saving.register_keras_serializable(package="braintumor", name="xcep_preprocess")
    def xcep_preprocess(x):
        from tensorflow.keras.applications.xception import preprocess_input
        return preprocess_input(x)

    return eff_preprocess, xcep_preprocess


def _build_fusion_model(n_classes: int, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import applications, layers
    from tensorflow.keras.models import Model

    eff_pre, xcep_pre = _register()

    inp = layers.Input((img_size, img_size, 3), name="input")

    # Branch A: EfficientNet-B2 (raw [0,255] -> internal norm).
    effb = applications.EfficientNetB2(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    effb._name = "eff_backbone"
    effb.trainable = False
    a = layers.Lambda(eff_pre, name="eff_pre")(inp)
    a = effb(a, training=False)
    a = layers.GlobalAveragePooling2D(name="eff_gap")(a)

    # Branch B: Xception (expects [-1,1]).
    xcep = applications.Xception(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    xcep._name = "xcep_backbone"
    xcep.trainable = False
    b = layers.Lambda(xcep_pre, name="xcep_pre")(inp)
    b = xcep(b, training=False)
    b = layers.GlobalAveragePooling2D(name="xcep_gap")(b)

    # Late fusion: concatenate pooled feature vectors.
    x = layers.Concatenate(name="fusion_concat")([a, b])
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Dropout(0.5, name="fusion_drop")(x)        # paper uses 0.5
    x = layers.Dense(256, activation="relu", name="fusion_dense")(x)
    x = layers.Dropout(0.3, name="fusion_drop2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name="DualBackbone_Eff_Xcep")
    return model, (effb, xcep)


def load_fusion_model(path, n_classes: int = 4, img_size: int | None = None):
    """Load a fusion model robustly, even if it was saved with bare Lambda
    preprocessing layers that Keras 3 cannot deserialize.

    Strategy: try a normal ``load_model`` first (works for models saved with the
    registered wrappers above). If that raises, reconstruct the identical
    architecture in code and load only the *weights* from the ``.keras`` archive.
    The Lambda layers carry no weights, so the weight load is unaffected, and the
    rebuilt Lambdas use the correct per-branch preprocessing (EfficientNet raw,
    Xception [-1,1]) - exactly what the model was trained with - so predictions
    are faithful.
    """
    import tensorflow as tf

    path = str(path)
    img_size = img_size or config.IMG_SIZE["b2"]
    try:
        return tf.keras.models.load_model(path)
    except Exception as exc:
        print(f"[fusion] direct load failed ({type(exc).__name__}); "
              f"reconstructing architecture and loading weights from {path}")
        model, _ = _build_fusion_model(n_classes, img_size)
        try:
            model.load_weights(path)        # Keras 3 reads weights from the .keras archive
            return model
        except Exception:
            # Fallback: extract the weights file from the .keras zip and load that.
            import zipfile, tempfile, os
            with zipfile.ZipFile(path) as zf:
                wname = next((n for n in zf.namelist()
                              if n.endswith(".weights.h5") or n.endswith(".h5")), None)
                if wname is None:
                    raise
                tmp = tempfile.mkdtemp()
                zf.extract(wname, tmp)
                model.load_weights(os.path.join(tmp, wname))
            return model


def _unfreeze_top(backbone, from_block: int) -> int:
    """Unfreeze deep blocks; keep BN frozen. Returns #layers unfrozen."""
    import tensorflow as tf

    backbone.trainable = True
    n = 0
    for layer in backbone.layers:
        m = re.match(r"block(\d+)", layer.name)
        block_no = int(m.group(1)) if m else 0
        train_it = block_no >= from_block
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            train_it = False
        layer.trainable = train_it
        n += int(train_it)
    return n


def train(classes: List[str] | None = None, use_quarantine: bool = True) -> Path:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.metrics import Precision, Recall
    from tensorflow.keras.optimizers import Adam

    set_global_seed(config.SEED)
    classes = classes or config.CLASSES_4
    img_size = config.IMG_SIZE["b2"]    # 260; both backbones accept it

    from .data import build_dataframes, make_generators
    quarantine = set()
    if use_quarantine:
        from .audit import load_quarantine
        quarantine = load_quarantine()
        if quarantine:
            print(f"[fusion] quarantine: {len(quarantine)} gliomas excluded from TRAIN.")

    train_df, valid_df, test_df = build_dataframes(classes, quarantine=quarantine)
    tr_gen, va_gen, ts_gen = make_generators(
        train_df, valid_df, test_df, img_size=img_size, augment=True
    )

    # class weights
    counts = train_df["Class"].value_counts().reindex(classes).fillna(0).values
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (len(classes) * counts)
    cw = {i: float(w[i]) for i in range(len(classes))}

    model, (effb, xcep) = _build_fusion_model(len(classes), img_size)
    loss = CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)
    metrics = ["accuracy", Precision(name="precision"), Recall(name="recall")]

    # Phase 1: both backbones frozen, train fusion head.
    model.compile(optimizer=Adam(config.PHASE1_LR), loss=loss, metrics=metrics)
    cbs = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    print("\n[fusion] Phase 1 - fusion head warmup (both backbones frozen)")
    model.fit(tr_gen, validation_data=va_gen, epochs=config.PHASE1_EPOCHS,
              class_weight=cw, callbacks=cbs, verbose=1)

    # Phase 2: unfreeze deep blocks of BOTH backbones.
    n_eff = _unfreeze_top(effb, config.UNFREEZE_FROM_BLOCK)
    n_xcep = _unfreeze_top(xcep, 13)   # Xception blocks go to 14; unfreeze 13-14
    print(f"[fusion] Phase 2 - unfroze eff:{n_eff} xcep:{n_xcep} layers (BN frozen)")
    model.compile(optimizer=Adam(config.PHASE2_LR), loss=loss, metrics=metrics)
    cbs2 = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]
    model.fit(tr_gen, validation_data=va_gen, epochs=config.PHASE2_EPOCHS,
              class_weight=cw, callbacks=cbs2, verbose=1)

    tag = f"fusion_eff_xcep_{len(classes)}c{'_clean' if quarantine else ''}"
    out_path = config.MODELS_DIR / f"{tag}.keras"
    model.save(out_path)
    print(f"[fusion] saved -> {out_path}")

    from .evaluate import evaluate_model
    rp = evaluate_model(model, ts_gen, classes, use_tta=False,
                        title=f"{tag}_plain", save_dir=config.REPORTS_DIR)
    rt = evaluate_model(model, ts_gen, classes, use_tta=True,
                        title=f"{tag}_tta", save_dir=config.REPORTS_DIR)
    with open(config.REPORTS_DIR / f"{tag}_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"classes": classes, "img_size": img_size,
                   "quarantined_gliomas": len(quarantine),
                   "test_plain": rp["metrics"], "test_tta": rt["metrics"]},
                  fh, indent=2)
    return out_path


# ===========================================================================
# Multi-pathway hybrid cascade classifier (Pathway A + Pathway B + consensus)
# ===========================================================================
class HybridCascadeClassifier:
    """Dual-pathway classifier with an adaptive-consensus head.

    Pathway A - Direct fusion: the late-fusion 4-class model(s) (EfficientNet-B2
        + Xception, and the single B2), each trained with label smoothing 0.05 to
        regularize the SARTAJ label noise found in EDA. Probabilities are averaged.

    Pathway B - Hierarchical cascade: a binary "tumor vs no-tumor" stage; if it
        flags a tumor, a ternary "glioma/meningioma/pituitary" stage runs and the
        joint distribution is mapped back to the 4-class space
        (P(class) = P(tumor) * P(class|tumor); P(notumor) = P(healthy)).
        Active only when the binary + subtype models exist in the registry.

    Adaptive Consensus Head: merges the two pathways by weighted voting. With
        ``adaptive=True`` each pathway's base weight is scaled by its own max-softmax
        confidence, so the more decisive pathway dominates per-image; otherwise the
        fixed ``base_weights`` are used. If Pathway B is unavailable, Pathway A is
        returned unchanged.

    Note on label smoothing: it is a *training-time* regularizer and is already
    baked into the Pathway-A models' weights; it is not (and cannot be) re-applied
    at inference. This class is inference-only.
    """

    def __init__(self, registry=None, *, base_weights=(0.5, 0.5),
                 tumor_threshold: float = 0.5, adaptive: bool = True,
                 verbose: bool = True):
        from .registry import ModelRegistry
        self.reg = registry or ModelRegistry()
        self.class_names = self.reg.class_names(4)
        self.base_weights = tuple(base_weights)
        self.tumor_threshold = float(tumor_threshold)
        self.adaptive = bool(adaptive)

        # --- Pathway A sources (single 4-class + fusion) ---
        self._a = []                 # list of dicts: {name, model, img_size, rescale, kind}
        prim = self.reg.latest_4class()
        if prim is None:
            raise RuntimeError("HybridCascade needs at least one 4-class model.")
        self._a.append({"name": prim.path.name, "model": prim.load(),
                        "img_size": prim.img_size, "rescale": prim.rescale_to_unit})
        fus = self.reg.latest_fusion()
        if fus is not None:
            try:
                self._a.append({"name": fus.path.name,
                                "model": load_fusion_model(fus.path, 4, fus.img_size),
                                "img_size": fus.img_size, "rescale": fus.rescale_to_unit})
            except Exception as exc:
                if verbose:
                    print(f"[hybrid] fusion model unavailable for Pathway A: {exc}")

        # --- Pathway B (binary -> subtype cascade) ---
        self.two_stage = None
        if self.reg.has_two_stage():
            from .two_stage import TwoStageClassifier
            b, s = self.reg.latest_binary(), self.reg.latest_subtype()
            self.two_stage = TwoStageClassifier(
                binary_model_path=b.path, subtype_model_path=s.path,
                binary_img_size=b.img_size, subtype_img_size=s.img_size,
                tumor_threshold=self.tumor_threshold)
        if verbose:
            print(f"[hybrid] Pathway A sources: {[a['name'] for a in self._a]}")
            print(f"[hybrid] Pathway B: "
                  f"{'binary->subtype cascade' if self.two_stage else 'inactive (train binary+subtype)'}")

    # ------------------------------------------------------------------
    def _probs_for(self, src, img_path) -> np.ndarray:
        from .preprocessing import load_rgb, preprocess_array
        x = preprocess_array(load_rgb(img_path), src["img_size"],
                             rescale_to_unit=src["rescale"])
        return src["model"].predict(x, verbose=0)[0].astype(np.float32)

    def pathway_a(self, img_path) -> np.ndarray:
        vecs = [self._probs_for(s, img_path) for s in self._a]
        p = np.mean(np.stack(vecs, 0), 0)
        return p / (p.sum() + 1e-12)

    def pathway_b(self, img_path):
        if self.two_stage is None:
            return None
        res = self.two_stage.predict_image(img_path)
        fp = res["full_probs"]
        p = np.array([fp[c] for c in self.class_names], dtype=np.float32)
        return p / (p.sum() + 1e-12)

    # ------------------------------------------------------------------
    def predict(self, img_path) -> dict:
        a = self.pathway_a(img_path)
        b = self.pathway_b(img_path)

        if b is None:
            consensus = a
            weights = {"pathway_a": 1.0, "pathway_b": 0.0}
        else:
            wa, wb = self.base_weights
            if self.adaptive:                          # scale by each pathway's confidence
                wa *= float(a.max()); wb *= float(b.max())
            tot = wa + wb + 1e-12
            wa, wb = wa / tot, wb / tot
            consensus = wa * a + wb * b
            consensus = consensus / (consensus.sum() + 1e-12)
            weights = {"pathway_a": round(wa, 3), "pathway_b": round(wb, 3)}

        idx = int(np.argmax(consensus))
        return {
            "prediction": self.class_names[idx],
            "confidence": float(consensus[idx]),
            "consensus_probs": {self.class_names[i]: float(consensus[i]) for i in range(4)},
            "pathway_a_probs": {self.class_names[i]: float(a[i]) for i in range(4)},
            "pathway_b_probs": (None if b is None else
                                {self.class_names[i]: float(b[i]) for i in range(4)}),
            "consensus_weights": weights,
            "method": "adaptive_weighted_voting" if self.adaptive else "weighted_voting",
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", choices=["4", "3"], default="4")
    ap.add_argument("--no-quarantine", action="store_true")
    args = ap.parse_args()
    classes = config.CLASSES_4 if args.classes == "4" else config.CLASSES_3
    train(classes=classes, use_quarantine=not args.no_quarantine)


if __name__ == "__main__":
    main()
