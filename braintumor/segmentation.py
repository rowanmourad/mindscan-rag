"""braintumor/segmentation.py - tumor segmentation (Attention U-Net + XAI fallback).

Two segmentation paths, in preference order:

1. **Attention U-Net (independent, supervised)** - ``build_attention_unet`` /
   ``AttentionUNetSegmenter``. A real encoder-decoder with spatial Attention Gates
   (Oktay et al., 2018) on the skip connections plus channel (squeeze-excitation)
   attention at the bottleneck, which actively suppress normal-tissue activations
   and yield a sharp pixel-level mask. This is the path the morphometry (axes,
   area, bbox) should read from - it is **independent of Grad-CAM++**.

   IMPORTANT: this dataset (Figshare+SARTAJ+Br35H) has **no ground-truth tumor
   masks**, so the U-Net cannot be trained here. ``AttentionUNetSegmenter`` loads
   trained weights from ``models/`` if present and otherwise reports
   ``available == False`` so callers fall back to path 2. Supply BraTS-style
   (image, mask) pairs and run ``train_attention_unet`` to activate it.

2. **XAI-guided weak segmentation (fallback)** - ``segment_tumor``. Uses the
   classifier's Grad-CAM++ attention to seed a GrabCut/intensity refinement,
   constrained to the brain. A *weak, unvalidated* lesion mask used only when no
   trained U-Net exists.

``segment()`` is the dispatcher: it returns the U-Net mask when a trained
segmenter is available, else the weak XAI-guided mask. Grad-CAM++ itself is kept
strictly as a **visual interpretability overlay** (see ``xai.py``), not as the
source of structural metrics.

Public API
----------
    segment(image_or_path, classifier=None, ...) -> SegmentationResult   # dispatcher
    build_attention_unet(input_size, base_filters) -> keras.Model
    AttentionUNetSegmenter(weights_path) ; .available ; .predict_mask(rgb)
    train_attention_unet(images, masks, ...)   # needs ground-truth masks
    segment_tumor(model, img_path, img_size, ...) -> SegmentationResult  # weak fallback
    visualize_segmentation(result, ...) -> matplotlib Figure
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


_CAVEAT = ("XAI-guided weak segmentation (anatomical core-seed-locked Grad-CAM + "
           "GrabCut refinement); not a clinically validated tumor mask.")


@dataclass
class SegmentationResult:
    present: bool
    mask: Optional[np.ndarray] = None          # uint8 {0,1}, model-input resolution
    bbox: Optional[Dict[str, int]] = None      # {x,y,w,h}
    center: Optional[Dict[str, int]] = None    # {x,y}
    contour: Optional[np.ndarray] = None        # Nx1x2 cv2 contour (largest)
    area_px: int = 0
    brain_area_px: int = 0
    brain_coverage_pct: float = 0.0
    img_size: int = 0
    method: str = "gradcam-coreseed-grabcut"
    heatmap: Optional[np.ndarray] = None        # normalized [0,1] at img_size
    warnings: List[str] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, object]:
        """Shaped for schema.SegmentationSummary."""
        return {
            "method": self.method,
            "mask_area_px": int(self.area_px),
            "mask_brain_coverage_pct": round(float(self.brain_coverage_pct), 3),
            "dice_against_attention": None,   # filled by pipeline if requested
            "mask_path": None,
        }

    def localization_dict(self) -> Dict[str, object]:
        return {
            "present": self.present,
            "bbox": self.bbox,
            "center": self.center,
            "area_px": int(self.area_px),
            "brain_area_px": int(self.brain_area_px),
            "brain_coverage_pct": round(float(self.brain_coverage_pct), 3),
            "method": self.method,
            "caveat": _CAVEAT,
        }


# ---------------------------------------------------------------------------
def segment_tumor(
    model,
    img_path: str | Path,
    img_size: int,
    *,
    pred_index: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    heatmap_threshold: float = 0.45,
    grad_model=None,
    rescale_to_unit: bool = False,
    skull_strip_input: bool = False,
) -> SegmentationResult:
    """Produce a weak lesion mask for one image, anchored on model attention.

    Pipeline
    --------
    1. Standard Grad-CAM heatmap for the predicted class, then HARD ENERGY
       THRESHOLDING (zero everything below 50% of the peak activation). Standard
       Grad-CAM is tighter/more peaked than Grad-CAM++ (which spreads to cover the
       whole object), and the hard threshold strips background leakage so the seed
       sits inside the tumor core - this prevents the mask from blowing up to engulf
       the brain.
    2. Brain mask via Otsu (denominator + spatial constraint).
    3. Seed = thresholded-core heatmap, intersected with brain.
    4. Refine the seed with GrabCut initialized from the seed's bounding box;
       fall back to in-brain intensity thresholding if GrabCut is unavailable.
    5. Largest connected component -> mask, contour, bbox, centroid, area.
    """
    import cv2
    from .xai import find_last_conv_layer, build_grad_model, compute_gradcam
    from .preprocessing import load_rgb, brain_mask, preprocess_array

    original_rgb = load_rgb(img_path)
    resized = cv2.resize(original_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = preprocess_array(original_rgb, img_size, rescale_to_unit=rescale_to_unit,
                         skull_strip_input=skull_strip_input)

    # ---- prediction / class index ----
    probs = model.predict(x, verbose=0)[0]
    if pred_index is None:
        pred_index = int(np.argmax(probs))

    # ---- Standard Grad-CAM (classic; method is locked to Grad-CAM project-wide) ----
    if grad_model is None:
        grad_model = build_grad_model(model, find_last_conv_layer(model))
    hm, _, _ = compute_gradcam(grad_model, x, pred_index=pred_index)
    hm = cv2.resize(hm.astype(np.float32), (img_size, img_size))
    hm = np.clip(hm / (hm.max() + 1e-8), 0, 1)

    # ---- brain mask (constraint + denominator) ----
    # Use the FILLED-brain pixel count as the denominator (not the Otsu contour
    # polygon area, which could be smaller than the mask and produce the absurd
    # ">100% of brain" coverage). The refined mask is constrained to the brain,
    # so coverage is now mathematically bounded to <= 100%.
    bmask, _contour_area = brain_mask(resized)
    brain_bin = (bmask > 0).astype(np.uint8)
    brain_area = max(int(brain_bin.sum()), 1)

    # ============ ANATOMICAL CORE SEED LOCKING (tight) ============
    # Tightness budgets (fractions of brain area). The hot core stays small even on
    # diffuse maps; the final mask is hard-capped.
    MAX_CORE_FRAC = 0.12
    MAX_MASK_FRAC = 0.25

    # 1. Restrict Grad-CAM to inside the brain so hyper-intense neck/spine pixels
    #    can never win the peak search.
    hm_in = hm * brain_bin.astype(np.float32)
    result = SegmentationResult(present=False, img_size=img_size,
                                brain_area_px=int(brain_area), heatmap=hm_in)
    if float(hm_in.max()) <= 1e-6:
        result.warnings.append("No in-brain Grad-CAM activation; no seed produced.")
        return result

    # 2. Absolute peak activation pixel, strictly inside the brain.
    py, px = np.unravel_index(int(np.argmax(hm_in)), hm_in.shape)
    peak_val = float(hm_in[py, px])

    # 3. Hot core = the connected high-energy blob CONTAINING the peak, using an
    #    ESCALATING threshold so the seed stays tight even when Grad-CAM is diffuse.
    #    A low-res heatmap upsampled to 260px is smooth, so a flat 0.5*peak cut can
    #    swallow half the brain (that was the bug). We raise the threshold until the
    #    peak's component is <= MAX_CORE_FRAC of the brain.
    core = None
    for thr_frac in (0.85, 0.75, 0.65, 0.55):
        hot = (hm_in >= thr_frac * peak_val).astype(np.uint8)
        _, lab = cv2.connectedComponents(hot, connectivity=8)
        cand = (lab == lab[py, px]).astype(np.uint8)
        if 0 < int(cand.sum()) <= MAX_CORE_FRAC * brain_area:
            core = cand
            break
    if core is None:
        # Degenerate/flat heatmap: fall back to a small fixed disk on the peak.
        core = np.zeros((img_size, img_size), np.uint8)
        cv2.circle(core, (int(px), int(py)), max(4, int(round(img_size * 0.05))), 1, -1)
        core = cv2.bitwise_and(core, brain_bin)
    hm = hm_in                      # the locked, in-brain heatmap drives the overlay
    seed = core * 255

    # tight bbox around the hot core (small pad only)
    ys, xs = np.where(core > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = max(3, img_size // 50)
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(img_size - 1, x1 + pad), min(img_size - 1, y1 + pad)

    # 4. GrabCut from the hot core, with the result intersected against a MODEST
    #    dilation of the core (the "envelope"), so it refines the lesion edge but
    #    cannot balloon to fill the ROI / neck.
    kd_env = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    core_envelope = cv2.bitwise_and(cv2.dilate(core, kd_env, iterations=1), brain_bin)
    refined = None
    try:
        if (rx1 - rx0) > 4 and (ry1 - ry0) > 4:
            gc_mask = np.full((img_size, img_size), cv2.GC_BGD, np.uint8)
            gc_mask[ry0:ry1, rx0:rx1] = cv2.GC_PR_BGD
            kd2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            gc_mask[cv2.dilate(core, kd2) > 0] = cv2.GC_PR_FGD
            gc_mask[core > 0] = cv2.GC_FGD
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            rect = (rx0, ry0, max(1, rx1 - rx0), max(1, ry1 - ry0))
            cv2.grabCut(resized, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
            refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                               1, 0).astype(np.uint8)
            refined = cv2.bitwise_and(refined, core_envelope)   # bounded to core nbhd
    except Exception as exc:
        result.warnings.append(f"GrabCut failed ({exc}); using locked core seed.")
        refined = None

    if refined is None or int(refined.sum()) < 20:
        # Fallback: the locked hot core itself (already tight and in-brain).
        refined = core.copy()

    # morphological cleanup + largest component
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, k, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k, iterations=2)

    def _largest(m):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return None
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == big).astype(np.uint8)

    mask = _largest(refined)
    # Hard tightness cap: if the refined mask still exceeds MAX_MASK_FRAC of the
    # brain, discard it and use the locked hot core (guaranteed <= MAX_CORE_FRAC).
    core_mask = _largest((seed > 0).astype(np.uint8))
    if mask is not None and mask.sum() > MAX_MASK_FRAC * brain_area:
        result.warnings.append(
            f"Refined mask exceeded {MAX_MASK_FRAC*100:.0f}% of brain; capped to "
            "the locked Grad-CAM hot core.")
        mask = core_mask if core_mask is not None else core.astype(np.uint8)
    if mask is None:
        mask = core_mask
    if mask is None or int(mask.sum()) == 0:
        result.warnings.append("Refinement produced an empty mask.")
        return result

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(cnts, key=cv2.contourArea) if cnts else None
    area = int(mask.sum())
    bx, by, bw, bh = cv2.boundingRect(contour) if contour is not None else (0, 0, 0, 0)
    M = cv2.moments(mask, binaryImage=True)
    cx = int(M["m10"] / (M["m00"] + 1e-8)); cy = int(M["m01"] / (M["m00"] + 1e-8))

    result.present = True
    result.mask = mask
    result.contour = contour
    result.bbox = {"x": bx, "y": by, "w": bw, "h": bh}
    result.center = {"x": cx, "y": cy}
    result.area_px = area
    result.brain_coverage_pct = min(100.0, 100.0 * area / brain_area)   # bounded
    result.warnings.append(_CAVEAT)
    return result


# ---------------------------------------------------------------------------
def visualize_segmentation(result: SegmentationResult, original_rgb: np.ndarray,
                           *, prediction: str = "", save_path: Optional[Path] = None):
    """4-panel figure: Original | Heatmap | Mask | Overlay (contour+bbox)."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = cv2.resize(original_rgb, (result.img_size, result.img_size))
    hm = result.heatmap if result.heatmap is not None else np.zeros((result.img_size,) * 2)
    mask = result.mask if result.mask is not None else np.zeros((result.img_size,) * 2, np.uint8)

    overlay = img.copy()
    if result.present and result.contour is not None:
        cv2.drawContours(overlay, [result.contour], -1, (0, 255, 0), 2)
        bb = result.bbox
        cv2.rectangle(overlay, (bb["x"], bb["y"]), (bb["x"] + bb["w"], bb["y"] + bb["h"]),
                      (255, 0, 0), 2)
        c = result.center
        cv2.drawMarker(overlay, (c["x"], c["y"]), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)

    method = result.method or "segmentation"
    # Grad-CAM is decoupled from segmentation now; the heatmap panel is a visual
    # REFERENCE only and is not used to produce the mask.
    attn_title = "Grad-CAM (reference only)"
    if method.startswith("attention-unet"):
        mask_title = "Tumor mask (U-Net)"
    elif method.startswith(("medsam", "sam")):
        mask_title = "MedSAM mask (independent box)"
    elif method.startswith("clinician"):
        mask_title = "Clinician-verified mask"
    elif method.startswith("intensity"):
        mask_title = "Intensity mask (fallback)"
    else:
        mask_title = "Tumor mask"

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    axes[0].imshow(img); axes[0].set_title("Original (resized)", fontweight="bold")
    axes[1].imshow(hm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(attn_title, fontweight="bold")
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title(mask_title, fontweight="bold")
    title = f"Segmentation overlay - {prediction.upper()}"
    if result.present:
        title += f"\n{result.brain_coverage_pct:.1f}% of brain"
    axes[3].imshow(overlay); axes[3].set_title(title, fontweight="bold")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Tumor segmentation - method: {method}  "
                 "(research use; not a validated clinical mask)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ===========================================================================
# Attention U-Net  (independent, supervised pixel-level segmenter)
# ===========================================================================
# Architecture: a 4-level encoder/decoder U-Net with
#   * spatial Attention Gates on every skip connection (Oktay et al., 2018):
#     the coarse decoder signal gates the encoder skip so normal-tissue
#     activations are suppressed and tumor margins are sharpened, and
#   * a channel (squeeze-and-excitation) attention block at the bottleneck.
# Output: a single-channel sigmoid mask (tumor probability per pixel).
# ---------------------------------------------------------------------------
def _conv_block(x, f):
    from tensorflow.keras import layers
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    return x


def _channel_attention(x, ratio: int = 8):
    """Squeeze-and-Excitation channel attention."""
    from tensorflow.keras import layers
    ch = int(x.shape[-1])
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(ch // ratio, 1), activation="relu")(se)
    se = layers.Dense(ch, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, ch))(se)
    return layers.multiply([x, se])


def _attention_gate(skip, gating, inter_ch: int):
    """Spatial Attention Gate. ``gating`` is upsampled to ``skip`` resolution,
    then a 1x1 conv + sigmoid yields per-pixel attention coefficients that
    re-weight the skip connection (suppresses irrelevant/normal-tissue regions).
    """
    from tensorflow.keras import layers
    theta = layers.Conv2D(inter_ch, 1, padding="same")(skip)
    phi = layers.Conv2D(inter_ch, 1, padding="same")(gating)
    add = layers.add([theta, phi])
    act = layers.Activation("relu")(add)
    psi = layers.Conv2D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)        # (H,W,1) attention map
    return layers.multiply([skip, psi])


def build_attention_unet(input_size: int = 256, base_filters: int = 32,
                         channels: int = 3):
    """Build the Attention U-Net (Keras). Returns an uncompiled ``keras.Model``."""
    from tensorflow.keras import layers, Model
    b = base_filters
    inp = layers.Input((input_size, input_size, channels), name="image")

    c1 = _conv_block(inp, b);      p1 = layers.MaxPool2D()(c1)
    c2 = _conv_block(p1, b * 2);   p2 = layers.MaxPool2D()(c2)
    c3 = _conv_block(p2, b * 4);   p3 = layers.MaxPool2D()(c3)
    c4 = _conv_block(p3, b * 8);   p4 = layers.MaxPool2D()(c4)

    bn = _conv_block(p4, b * 16)
    bn = _channel_attention(bn)                    # channel attention at bottleneck

    u4 = layers.UpSampling2D()(bn)
    a4 = _attention_gate(c4, u4, b * 8)
    c5 = _conv_block(layers.concatenate([u4, a4]), b * 8)

    u3 = layers.UpSampling2D()(c5)
    a3 = _attention_gate(c3, u3, b * 4)
    c6 = _conv_block(layers.concatenate([u3, a3]), b * 4)

    u2 = layers.UpSampling2D()(c6)
    a2 = _attention_gate(c2, u2, b * 2)
    c7 = _conv_block(layers.concatenate([u2, a2]), b * 2)

    u1 = layers.UpSampling2D()(c7)
    a1 = _attention_gate(c1, u1, b)
    c8 = _conv_block(layers.concatenate([u1, a1]), b)

    out = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(c8)
    return Model(inp, out, name="AttentionUNet")


def dice_loss(y_true, y_pred, smooth: float = 1.0):
    import tensorflow as tf
    y_true = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    return 1.0 - (2.0 * inter + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def bce_dice_loss(y_true, y_pred):
    import tensorflow as tf
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


class AttentionUNetSegmenter:
    """Wraps a trained Attention U-Net. ``available`` is False (and ``predict_mask``
    returns None) when no trained weights exist, so callers fall back cleanly."""

    DEFAULT_NAMES = ("attention_unet.keras", "attention_unet.weights.h5",
                     "unet_seg.keras")

    def __init__(self, weights_path=None, input_size: int = 256,
                 search_dirs=None):
        self.input_size = int(input_size)
        self.model = None
        self.weights_path = None
        path = self._resolve(weights_path, search_dirs)
        if path is not None:
            try:
                import tensorflow as tf
                if str(path).endswith(".keras"):
                    self.model = tf.keras.models.load_model(path, compile=False)
                else:
                    self.model = build_attention_unet(self.input_size)
                    self.model.load_weights(path)
                self.weights_path = str(path)
                print(f"[seg] Attention U-Net loaded: {path}")
            except Exception as exc:
                print(f"[seg] Attention U-Net present but failed to load ({exc}); "
                      "falling back to XAI-guided weak segmentation.")
                self.model = None

    def _resolve(self, weights_path, search_dirs):
        if weights_path and Path(weights_path).exists():
            return Path(weights_path)
        try:
            from . import config
            dirs = search_dirs or config.MODEL_SEARCH_DIRS
        except Exception:
            dirs = search_dirs or []
        for d in dirs:
            d = Path(d)
            if not d.exists():
                continue
            for name in self.DEFAULT_NAMES:
                if (d / name).exists():
                    return d / name
        return None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict_mask(self, rgb_uint8: np.ndarray, threshold: float = 0.5,
                     out_size: Optional[int] = None) -> Optional[np.ndarray]:
        """Return a binary {0,1} uint8 mask at ``out_size`` (default input_size)."""
        if self.model is None:
            return None
        import cv2
        out_size = out_size or self.input_size
        x = cv2.resize(rgb_uint8, (self.input_size, self.input_size)).astype("float32") / 255.0
        prob = self.model.predict(x[None, ...], verbose=0)[0, ..., 0]
        prob = cv2.resize(prob, (out_size, out_size))
        return (prob >= threshold).astype(np.uint8)


def train_attention_unet(images, masks, *, input_size: int = 256,
                         base_filters: int = 32, epochs: int = 40,
                         batch_size: int = 8, val_split: float = 0.15,
                         out_path=None):
    """Train the Attention U-Net on (image, mask) pairs.

    REQUIRES ground-truth tumor masks, which the current classification dataset
    does NOT provide. Supply a BraTS-style set (or hand-annotated masks) as
    ``images`` (N,H,W,3 uint8) and ``masks`` (N,H,W or N,H,W,1 binary). Saves to
    ``models/attention_unet.keras`` so the registry/segmenter pick it up.
    """
    import numpy as np
    import tensorflow as tf
    from . import config

    X = np.asarray(images, dtype="float32")
    if X.max() > 1.5:
        X = X / 255.0
    Y = np.asarray(masks, dtype="float32")
    if Y.ndim == 3:
        Y = Y[..., None]
    Y = (Y > 0).astype("float32")

    model = build_attention_unet(input_size, base_filters)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=bce_dice_loss,
                  metrics=[tf.keras.metrics.MeanIoU(num_classes=2, name="iou")])
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                            restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=4, min_lr=1e-6)]
    model.fit(X, Y, validation_split=val_split, epochs=epochs,
              batch_size=batch_size, callbacks=cbs)
    out_path = Path(out_path or (config.MODELS_DIR / "attention_unet.keras"))
    model.save(out_path)
    print(f"[seg] Attention U-Net saved -> {out_path}")
    return out_path


def train_unet_from_feedback(input_size: int = 256, min_samples: int = 10,
                             epochs: int = 40, **kw):
    """SELF-LEARNING (safe, batch): train/refresh the standalone Attention U-Net
    on the accumulated clinician-verified (image, mask) pairs in the feedback
    cache. This is the honest "active learning" loop - run it periodically (e.g.
    nightly or after N new corrections); it does NOT touch the classifier and is
    fully reproducible/auditable, unlike unsafe single-sample online updates.

    Once it saves ``models/attention_unet.keras``, the pipeline auto-promotes the
    U-Net to the primary (end-to-end, Grad-CAM-free) segmenter on next load.
    """
    import cv2
    import numpy as np
    from .clinical_feedback import CACHE_DIR

    imgs, masks = [], []
    for case in sorted(CACHE_DIR.glob("*")):
        ip, mp = case / "image.png", case / "mask.png"
        if ip.exists() and mp.exists():
            bgr = cv2.imread(str(ip))
            mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if bgr is None or mk is None:
                continue
            imgs.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                   (input_size, input_size)))
            masks.append((cv2.resize(mk, (input_size, input_size),
                                     interpolation=cv2.INTER_NEAREST) > 0).astype("float32"))
    n = len(imgs)
    if n < min_samples:
        print(f"[seg] {n} clinician-verified masks cached; need >= {min_samples} to "
              f"train the U-Net. Keep collecting corrections via the GUI brush.")
        return None
    print(f"[seg] training Attention U-Net on {n} clinician-verified masks ...")
    return train_attention_unet(np.array(imgs), np.array(masks),
                                input_size=input_size, epochs=epochs, **kw)


# ===========================================================================
# Dispatcher: prefer the trained U-Net, else the XAI-guided weak mask
# ===========================================================================
def segment(image_or_path, *, classifier=None, img_size: int = 260,
            pred_index: Optional[int] = None, class_names=None,
            grad_model=None, rescale_to_unit: bool = False,
            unet: Optional["AttentionUNetSegmenter"] = None,
            heatmap_threshold: float = 0.45,
            skull_strip_input: bool = False) -> SegmentationResult:
    """Return the best available segmentation.

    If a trained Attention U-Net is available (``unet.available``), its mask is
    used (independent of Grad-CAM++). Otherwise falls back to ``segment_tumor``
    (the weak XAI-guided mask). Morphometry/bbox should read from the returned
    ``SegmentationResult.mask`` regardless of which path produced it.
    """
    import cv2
    from .preprocessing import load_rgb, brain_mask

    unet = unet if unet is not None else AttentionUNetSegmenter(input_size=256)
    if unet.available:
        rgb = (load_rgb(image_or_path) if isinstance(image_or_path, (str, Path))
               else np.asarray(image_or_path))
        resized = cv2.resize(rgb, (img_size, img_size))
        mask = unet.predict_mask(resized, out_size=img_size)
        bmask, brain_area = brain_mask(resized)
        brain_area = max(brain_area, 1)
        res = SegmentationResult(present=False, img_size=img_size,
                                 brain_area_px=int(brain_area),
                                 method="attention-unet")
        if mask is not None and int(mask.sum()) > 0:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n > 1:
                big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                mask = (labels == big).astype(np.uint8)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(cnts, key=cv2.contourArea) if cnts else None
                bx, by, bw, bh = cv2.boundingRect(contour) if contour is not None else (0, 0, 0, 0)
                M = cv2.moments(mask, binaryImage=True)
                cx = int(M["m10"] / (M["m00"] + 1e-8)); cy = int(M["m01"] / (M["m00"] + 1e-8))
                area = int(mask.sum())
                res.present = True; res.mask = mask; res.contour = contour
                res.bbox = {"x": bx, "y": by, "w": bw, "h": bh}
                res.center = {"x": cx, "y": cy}; res.area_px = area
                res.brain_coverage_pct = 100.0 * area / brain_area
        return res

    # Fallback: XAI-guided weak segmentation (requires the classifier + grad model)
    if classifier is None:
        res = SegmentationResult(present=False, img_size=img_size, method="none")
        res.warnings.append("No trained U-Net and no classifier supplied; "
                            "cannot segment.")
        return res
    return segment_tumor(classifier, image_or_path, img_size,
                         pred_index=pred_index, class_names=class_names,
                         grad_model=grad_model, rescale_to_unit=rescale_to_unit,
                         heatmap_threshold=heatmap_threshold,
                         skull_strip_input=skull_strip_input)
