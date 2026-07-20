"""braintumor/clinical_feedback.py - Interactive clinician feedback loop.

Backend for the GUI "Clinical Verification Panel". Three responsibilities:

1. ``recompute_from_user_mask`` - take a physician-painted binary mask and
   instantly recompute the structural metrics (bbox, major/minor axes, area,
   quadrant) from it, replacing the model-derived mask.

2. ``region_constrained_reclassify`` - the "spatial feature masking" idea,
   implemented HONESTLY. It multiplies the user mask into the backbone's final
   conv feature maps and re-runs the classifier head.

   *** Scientific-integrity warning (read before using in a paper) ***
   This is an OPERATOR-GUIDED OVERRIDE, not an independent model inference.
   Suppressing all features outside the painted region trivially forces both the
   class score and any subsequent Grad-CAM toward that region - the result is
   circular (the operator dictates where the "evidence" is). It is useful as an
   interaction/teaching tool and to harvest correction data, but it must NOT be
   reported as the model "self-correcting" or as a valid explainability result.
   The returned dict is explicitly tagged ``operator_guided=True``.

3. ``log_feedback`` - persist (original image, model prediction, physician mask,
   corrected label, metrics) under ``artifacts/clinical_feedback/`` for future
   fine-tuning / active-learning runs.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config


FEEDBACK_DIR = config.OUT_DIR / "clinical_feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = FEEDBACK_DIR / "cache"          # aHash-keyed clinician-verified masks
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Perceptual hash (average-hash) + clinician-verified mask cache.
# Shared by the pipeline (authoritative Tier-1 segmenter) and the Streamlit UI,
# so a corrected slice is NEVER mis-segmented again.
# ---------------------------------------------------------------------------
def _as_pil_gray(image):
    from PIL import Image
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("L")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            import cv2
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return Image.fromarray(arr.astype(np.uint8))
    return image.convert("L")               # PIL


def ahash(image, size: int = 16) -> str:
    g = np.asarray(_as_pil_gray(image).resize((size, size)), dtype=np.float32)
    bits = (g > g.mean()).astype(np.uint8).flatten()
    return "".join(f"{int(''.join(map(str, bits[i:i+4])), 2):x}"
                   for i in range(0, len(bits), 4))


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return 10 ** 9
    to_bits = lambda hx: "".join(f"{int(c, 16):04b}" for c in hx)
    return sum(c1 != c2 for c1, c2 in zip(to_bits(a), to_bits(b)))


def save_cached_mask(image, mask, label, metrics: Optional[Dict] = None,
                     image_hash: Optional[str] = None) -> Path:
    """Persist a clinician-verified mask keyed by the slice's aHash.

    Pass ``image_hash`` to key by a hash computed elsewhere (e.g. the Streamlit UI
    hashes the full-resolution upload) so the pipeline's ``lookup_cached_mask`` on
    the same slice resolves to it exactly. The mask + image.png are still stored
    for offline U-Net training."""
    from PIL import Image
    h = image_hash or ahash(image)
    case = CACHE_DIR / h
    case.mkdir(parents=True, exist_ok=True)
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    Image.fromarray(m * 255).save(case / "mask.png")
    try:
        rgb = np.asarray(image) if isinstance(image, np.ndarray) else None
        if rgb is not None:
            Image.fromarray(rgb.astype(np.uint8)).save(case / "image.png")
    except Exception:
        pass
    (case / "meta.json").write_text(json.dumps({
        "hash": h, "label": label, "metrics": metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, default=str), encoding="utf-8")
    return case


def lookup_cached_mask(image, max_dist: int = 6,
                       image_hash: Optional[str] = None) -> Optional[Dict]:
    """Return the nearest clinician-verified mask for ``image`` (exact aHash or
    Hamming <= max_dist), or None. Result dict has keys: mask, label, hash, dist."""
    h = image_hash or ahash(image)
    best = None
    for meta_p in CACHE_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        d = 0 if meta.get("hash") == h else hamming(h, meta.get("hash", ""))
        if d <= max_dist:
            mp = meta_p.parent / "mask.png"
            if mp.exists() and (best is None or d < best["dist"]):
                from PIL import Image
                best = {"mask": np.asarray(Image.open(mp).convert("L")),
                        "label": meta.get("label"), "hash": meta.get("hash"), "dist": d}
    return best


# ---------------------------------------------------------------------------
# 1. Recompute structural metrics from a user-painted mask
# ---------------------------------------------------------------------------
def recompute_from_user_mask(image_rgb: np.ndarray, user_mask: np.ndarray,
                             *, predicted_class: Optional[str] = None,
                             confidence: Optional[float] = None) -> Dict:
    """Recompute geometry/morphometry/quadrant from a physician-supplied mask.

    ``image_rgb`` and ``user_mask`` must share spatial dimensions. Returns the
    full ``TumorReport.to_dict()`` plus a flat ``metrics`` block for the UI.
    """
    import cv2
    from .tumor_analysis import analyze_tumor

    mask = (np.asarray(user_mask) > 0).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    report = analyze_tumor(image_rgb, mask, predicted_class=predicted_class,
                           confidence=confidence)
    d = report.to_dict()
    morph = report.morphometry or {}
    geom = report.geometry or {}
    d["metrics"] = {
        "area_px": geom.get("tumor_area_px"),
        "area_pct_brain": geom.get("tumor_area_pct_brain"),
        "major_axis_px": morph.get("major_axis_px"),
        "minor_axis_px": morph.get("minor_axis_px"),
        "ellipsoidal_volume_px3_est": morph.get("ellipsoidal_volume_px3_est"),
        "bbox": {k: geom.get(f"bbox_{k}") for k in ("x", "y", "width", "height")}
                if geom else None,
        "quadrant": (report.quadrant or {}).get("quadrant"),
        "source": "physician_mask",
    }
    return d


# ---------------------------------------------------------------------------
# 2. Region-constrained re-classification (operator-guided override)
# ---------------------------------------------------------------------------
def region_constrained_reclassify(model, image_rgb: np.ndarray,
                                  user_mask: np.ndarray, *, img_size: int,
                                  class_names: List[str],
                                  rescale_to_unit: bool = False) -> Dict:
    """Multiply the user mask into the backbone feature maps and re-run the head.

    See the module-level warning: this is an OPERATOR-GUIDED OVERRIDE, not a valid
    independent prediction. Falls back to input-space masking if the model cannot
    be split into backbone/head.
    """
    import cv2
    from .preprocessing import preprocess_array
    from .xai import split_backbone_head

    x = preprocess_array(image_rgb, img_size, rescale_to_unit=rescale_to_unit)
    mask = (np.asarray(user_mask) > 0).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]

    method = "feature_space_masking"
    try:
        backbone, head = split_backbone_head(model)
        feat = backbone.predict(x, verbose=0)          # (1,h,w,C)
        h, w = feat.shape[1], feat.shape[2]
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
        m = (m > 0.05).astype(np.float32)[None, ..., None]    # broadcast over channels
        if m.sum() == 0:
            m = np.ones_like(m)                          # empty mask -> no constraint
        probs = head.predict(feat * m, verbose=0)[0]
    except Exception:
        method = "input_space_masking"                  # robust fallback
        mfull = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        masked_img = (image_rgb.astype(np.float32) *
                      cv2.resize(mfull, (image_rgb.shape[1], image_rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)[..., None])
        xm = preprocess_array(masked_img.astype(np.uint8), img_size,
                              rescale_to_unit=rescale_to_unit)
        probs = model.predict(xm, verbose=0)[0]

    probs = np.asarray(probs, dtype=np.float32)
    idx = int(np.argmax(probs))
    return {
        "prediction": class_names[idx],
        "confidence": float(probs[idx]),
        "per_class_probabilities": {class_names[i]: float(probs[i])
                                    for i in range(len(class_names))},
        "method": method,
        "operator_guided": True,
        "caveat": ("Operator-guided override: features outside the painted region "
                   "were suppressed, so this is NOT an independent model prediction "
                   "or a valid explainability result. For correction/teaching only."),
    }


# ---------------------------------------------------------------------------
# 3. Persist a correction for future fine-tuning / active learning
# ---------------------------------------------------------------------------
def log_feedback(*, image_path: str, image_rgb: Optional[np.ndarray],
                 user_mask: np.ndarray, model_prediction: Dict,
                 corrected_label: Optional[str] = None,
                 corrected_metrics: Optional[Dict] = None,
                 notes: str = "") -> Path:
    """Write a timestamped correction record under artifacts/clinical_feedback/.

    Stores: a copy of the source image, the physician mask (PNG), and a JSON
    record linking the initial model output to the physician's correction.
    """
    import cv2

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    case_dir = FEEDBACK_DIR / f"case_{ts}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # image
    img_name = "image" + Path(image_path).suffix if image_path else "image.png"
    try:
        if image_path and Path(image_path).exists():
            shutil.copy2(image_path, case_dir / img_name)
        elif image_rgb is not None:
            cv2.imwrite(str(case_dir / "image.png"),
                        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    except Exception:
        pass

    # mask
    m = (np.asarray(user_mask) > 0).astype(np.uint8) * 255
    if m.ndim == 3:
        m = m[..., 0]
    cv2.imwrite(str(case_dir / "physician_mask.png"), m)

    record = {
        "timestamp": ts,
        "image_path": str(image_path),
        "model_prediction": {
            "prediction": model_prediction.get("prediction"),
            "confidence": model_prediction.get("confidence"),
            "per_class_probabilities": model_prediction.get("per_class_probabilities"),
        },
        "physician_correction": {
            "corrected_label": corrected_label,
            "label_changed": (corrected_label is not None and
                              corrected_label != model_prediction.get("prediction")),
            "corrected_metrics": corrected_metrics,
            "notes": notes,
        },
        "use": "Future fine-tuning / active learning. The physician mask is a weak "
               "segmentation label; the corrected_label is a verified class label.",
    }
    (case_dir / "record.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"[feedback] logged correction -> {case_dir}")
    return case_dir


def list_feedback() -> List[Dict]:
    """Load all logged correction records (for building a fine-tuning set)."""
    out = []
    for rec in sorted(FEEDBACK_DIR.glob("case_*/record.json")):
        try:
            out.append(json.loads(rec.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# 4. Online "memorize the correction" micro-update  (SESSION-LOCAL, GUARDED)
# ---------------------------------------------------------------------------
#
#  *** Senior-engineer warning — read before enabling in production ***
#
#  Single-sample online SGD on a 95%-accurate model is intrinsically risky:
#  it causes catastrophic forgetting and is non-reproducible / non-auditable.
#  We make it SAFE the only way possible:
#    1. It NEVER touches the shared, process-cached model (that would corrupt
#       every other session). It operates on a CLONE created here.
#    2. The supervised signal is the corrected CLASS LABEL (a classifier cannot
#       backprop a paint mask - it has no spatial output). The mask only
#       optionally *focuses* the input; it does not "teach pixel locations".
#    3. Only the classification HEAD is unfrozen (backbone frozen), 1-3 steps,
#       lr 1e-5, so the conv feature extractor is preserved.
#    4. It is SESSION-SCOPED and REVERSIBLE (pipeline.reset_session restores the
#       pristine cached weights) and every correction is logged for offline QA.
#  The production-correct path remains OFFLINE BATCH fine-tuning on the
#  accumulated ``clinical_feedback/`` corrections + Attention-U-Net training on
#  accumulated masks. This online pass is an opt-in interactive aid, not a
#  validated learning procedure.
# ---------------------------------------------------------------------------
def clone_model_for_session(model):
    """Return a trainable deep copy of ``model`` (weights included).

    The clone is independent of the process-wide model cache, so fine-tuning it
    can never corrupt the shared/cached model used by other pipeline instances.
    """
    import tensorflow as tf
    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    return clone


def _set_head_trainable(model, also_last_block: bool = False):
    """Freeze the backbone sub-model; unfreeze the head (and optionally the last
    conv block). Returns the number of trainable layers."""
    import tensorflow as tf
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
    n = 0
    for layer in model.layers:
        if layer is backbone:
            layer.trainable = bool(also_last_block)
        else:
            layer.trainable = True
        n += int(getattr(layer, "trainable", False))
    if backbone is not None and also_last_block:
        # keep most of the backbone frozen; unfreeze only its deepest block + no BN
        for sub in backbone.layers:
            train_it = sub.name.startswith(("block7", "top"))
            if isinstance(sub, tf.keras.layers.BatchNormalization):
                train_it = False
            sub.trainable = train_it
    return n


def online_correction_update(model, image_rgb, user_mask, corrected_label,
                             class_names, *, img_size: int,
                             rescale_to_unit: bool = False, lr: float = 1e-5,
                             steps: int = 3, label_smoothing: float = 0.05,
                             focus_with_mask: bool = False,
                             also_last_block: bool = False) -> Dict:
    """Run a guarded micro-update on ``model`` (which MUST be a session clone).

    Returns before/after probabilities on this slice so the UI can show that the
    model moved toward the corrected label. Mutates ``model`` in place - pass a
    clone from :func:`clone_model_for_session`, never the cached model.
    """
    import numpy as np
    import tensorflow as tf
    from .preprocessing import preprocess_array, skull_strip

    if corrected_label not in class_names:
        raise ValueError(f"corrected_label {corrected_label!r} not in {class_names}")

    rgb = np.asarray(image_rgb)
    if focus_with_mask and user_mask is not None:
        import cv2
        m = (np.asarray(user_mask) > 0).astype(np.uint8)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != rgb.shape[:2]:
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgb = (rgb * m[..., None]).astype(np.uint8)

    x = preprocess_array(rgb, img_size, rescale_to_unit=rescale_to_unit)
    y = np.zeros((1, len(class_names)), dtype="float32")
    y[0, class_names.index(corrected_label)] = 1.0

    before = model.predict(x, verbose=0)[0].astype(float)

    n_train = _set_head_trainable(model, also_last_block=also_last_block)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(
                      label_smoothing=label_smoothing),
                  metrics=["accuracy"])
    losses = []
    for _ in range(max(1, int(steps))):
        losses.append(float(model.train_on_batch(x, y)[0]))

    after = model.predict(x, verbose=0)[0].astype(float)
    ci = class_names.index(corrected_label)
    return {
        "corrected_label": corrected_label,
        "trainable_layers": int(n_train),
        "lr": lr, "steps": int(steps),
        "loss_trace": losses,
        "before_probs": {class_names[i]: round(float(before[i]), 4) for i in range(len(class_names))},
        "after_probs": {class_names[i]: round(float(after[i]), 4) for i in range(len(class_names))},
        "target_prob_gain": round(float(after[ci] - before[ci]), 4),
        "method": "session_local_head_microupdate",
        "caveat": ("Session-local clone, head-only, single-slice micro-update toward "
                   "the corrected label. NOT validated learning; reset_session() "
                   "restores the cached weights. Use offline batch fine-tuning on "
                   "the logged corrections for the real model update."),
    }
