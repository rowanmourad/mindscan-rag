"""
xai_utils.py — Explainable AI utilities for the brain tumor classifier.

Provides:
    - find_last_conv_layer(model)
    - build_grad_model(model, last_conv_layer_name)
    - compute_gradcam(grad_model, img_tensor, pred_index=None)
    - compute_gradcam_plus_plus(grad_model, img_tensor, pred_index=None)
    - overlay_heatmap(rgb_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET)
    - focus_score(heatmap, threshold=0.5)
    - make_xai_figure(rgb_image, gradcam, gradcam_pp, title=None) -> matplotlib Figure

The key fix vs. the original notebook:
    The previous Grad-CAM created NEW random Dense layers inside the GradientTape.
    Those gradients were meaningless. Here we build a sub-model that reuses the
    actual trained weights: it taps the chosen conv layer's activations AND the
    model's real softmax output. That's the only way Grad-CAM tells you what
    your trained network actually looked at.
"""

from __future__ import annotations
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# LOCKED XAI METHOD.  The pipeline uses CLASSIC Grad-CAM for every model and
# every ensemble pathway, and the UI/report label is locked to this string, so
# clinicians never see the method toggle between "Grad-CAM" and "Grad-CAM++".
# ``compute_gradcam_plus_plus`` remains available for offline research/comparison
# but is intentionally OFF the default inference graph.
# ---------------------------------------------------------------------------
XAI_METHOD = "grad-cam"


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------
def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Walk the model (and its sub-models, e.g. Xception inside Sequential) and
    return the *name* of the deepest 4D conv layer.

    For Xception this resolves to `block14_sepconv2_act` (the activation after
    the last separable conv) when present, else the last `Conv2D`/`SeparableConv2D`.
    Using the activation layer gives smoother, better-localized heatmaps than
    the raw conv output.
    """
    # If the top-level is a Sequential wrapping a base model, descend into it.
    candidate_layers = []

    def _walk(m):
        for layer in m.layers:
            # Recurse into nested models (Sequential, Functional)
            if isinstance(layer, tf.keras.Model):
                _walk(layer)
                continue
            try:
                shape = layer.output.shape
            except AttributeError:
                continue
            if len(shape) == 4:  # (batch, H, W, C)
                candidate_layers.append(layer)

    _walk(model)

    if not candidate_layers:
        raise ValueError("No 4D convolutional layer found in the model.")

    # Prefer the last activation layer ending in '_act' (Xception convention),
    # otherwise just take the last 4D layer.
    for layer in reversed(candidate_layers):
        if layer.name.endswith("_act"):
            return layer.name

    return candidate_layers[-1].name


# ---------------------------------------------------------------------------
# Grad model — reuses TRAINED weights (this is what was broken before)
# ---------------------------------------------------------------------------
def build_grad_model(model: tf.keras.Model,
                     last_conv_layer_name: str) -> tf.keras.Model:
    """
    Build a functional model that outputs (conv_activations, final_predictions)
    in a single forward pass, reusing the existing trained weights.

    Handles the common case where `model` is a Sequential([base_model, head...])
    and the conv layer lives inside `base_model`.
    """
    # Locate the layer whether it's at the top level or inside a nested model.
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        # Top-level: easy case
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
    except ValueError:
        pass  # fall through to nested search

    # Search nested sub-models (handles the common case where the backbone is a
    # sub-Functional/Sequential at some index, with a head applied AFTER it).
    for backbone in model.layers:
        if not isinstance(backbone, tf.keras.Model):
            continue
        try:
            conv_layer = backbone.get_layer(last_conv_layer_name)
        except ValueError:
            continue

        # Sub-model returning conv activations AND the backbone's normal output.
        base_with_conv = tf.keras.Model(
            inputs=backbone.input,
            outputs=[conv_layer.output, backbone.output],
        )

        # Feed the REAL model input through the backbone, then replay ONLY the
        # layers that come AFTER the backbone in the top model. The previous
        # implementation replayed ``model.layers[1:]`` which re-included the
        # backbone itself (index 0 is the InputLayer, not the backbone), causing
        # a shape-mismatch crash on Functional models. This skips the backbone
        # by identity instead.
        inp = model.inputs[0] if getattr(model, "inputs", None) else \
            tf.keras.Input(shape=backbone.input_shape[1:])
        conv_out, base_out = base_with_conv(inp)

        x = base_out
        started = False
        for layer in model.layers:
            if layer is backbone:
                started = True
                continue
            if started:
                x = layer(x)
        return tf.keras.Model(inputs=inp, outputs=[conv_out, x])

    raise ValueError(
        f"Layer '{last_conv_layer_name}' not found in model or any sub-model."
    )


def split_backbone_head(model: "tf.keras.Model"):
    """Split a nested-backbone classifier into (backbone_model, head_model).

    Returns:
        backbone_model : image -> final conv feature map (the backbone output)
        head_model     : conv feature map -> class probabilities (GAP + dense head)

    Both reuse the model's TRAINED weights. This is used by the clinician
    feedback loop to inject an operator-supplied spatial mask into the conv
    feature maps before the head. Raises ValueError for flat (non-nested) models.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
    if backbone is None:
        raise ValueError("split_backbone_head: no nested backbone sub-model found.")

    inp = model.inputs[0]
    feat = backbone(inp)
    backbone_model = tf.keras.Model(inp, feat, name="backbone_feat")

    h_in = tf.keras.Input(shape=backbone.output_shape[1:])
    x = h_in
    started = False
    for layer in model.layers:
        if layer is backbone:
            started = True
            continue
        if started:
            x = layer(x)
    head_model = tf.keras.Model(h_in, x, name="classifier_head")
    return backbone_model, head_model


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_gradcam(grad_model: tf.keras.Model,
                    img_tensor: np.ndarray,
                    pred_index: int | None = None) -> tuple[np.ndarray, int, float]:
    """
    Standard Grad-CAM (Selvaraju et al., 2017).

    Args:
        grad_model: model returning (conv_outputs, predictions)
        img_tensor: preprocessed input, shape (1, H, W, 3), already normalized
        pred_index: which class to explain; defaults to argmax

    Returns:
        heatmap (H_conv, W_conv) in [0, 1]
        pred_index (int)
        confidence (float, the softmax prob of pred_index)
    """
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Gradients are None. The conv layer is not connected to the chosen "
            "output, or the layer name resolved to a non-trainable path."
        )

    # Global-average-pool the gradients over spatial dims -> channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]                         # (H, W, C)
    # Weight each channel by its pooled gradient and sum
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                          # (H, W)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-10)

    confidence = float(predictions[0, pred_index].numpy())
    return heatmap.numpy(), int(pred_index), confidence


# ---------------------------------------------------------------------------
# Grad-CAM++
# ---------------------------------------------------------------------------
def compute_gradcam_plus_plus(grad_model: tf.keras.Model,
                              img_tensor: np.ndarray,
                              pred_index: int | None = None
                              ) -> tuple[np.ndarray, int, float]:
    """
    Grad-CAM++ (Chattopadhay et al., 2018).

    Uses 1st, 2nd, and 3rd-order gradients of the score wrt conv activations to
    derive per-pixel weights, giving sharper and more complete object coverage
    than vanilla Grad-CAM — especially when multiple instances of the target
    feature are present (common in MRIs).

    Args/Returns: same shape as compute_gradcam.
    """
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    with tf.GradientTape() as t3:
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                conv_outputs, predictions = grad_model(img_tensor, training=False)
                if pred_index is None:
                    pred_index = int(tf.argmax(predictions[0]))
                # Grad-CAM++ uses the unnormalized exp(score). Using the softmax
                # output directly works well in practice and is numerically safer.
                class_channel = predictions[:, pred_index]
            first = t1.gradient(class_channel, conv_outputs)   # (1,H,W,C)
        second = t2.gradient(first, conv_outputs)              # (1,H,W,C)
    third = t3.gradient(second, conv_outputs)                  # (1,H,W,C)

    if first is None or second is None or third is None:
        raise RuntimeError("Higher-order gradients are None — model graph issue.")

    # Sum of conv activations over spatial dims (per channel)
    global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)  # (1,1,1,C)

    # α weights — see eq. (19) of the paper
    alpha_num = second
    alpha_denom = 2.0 * second + global_sum * third
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom                              # (1,H,W,C)

    # Only positive gradients contribute (ReLU on first)
    weights = tf.maximum(first, 0.0) * alphas                     # (1,H,W,C)
    # Sum over spatial dims to get per-channel weights
    deep_linearization_weights = tf.reduce_sum(weights, axis=(1, 2))  # (1,C)

    conv_outputs0 = conv_outputs[0]                               # (H,W,C)
    heatmap = tf.reduce_sum(
        deep_linearization_weights[0] * conv_outputs0, axis=-1
    )                                                             # (H,W)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-10)

    confidence = float(predictions[0, pred_index].numpy())
    return heatmap.numpy(), int(pred_index), confidence


# ---------------------------------------------------------------------------
# Hard energy thresholding (peak-relative suppression)
# ---------------------------------------------------------------------------
def hard_energy_threshold(heatmap: np.ndarray, frac: float = 0.5,
                          renormalize: bool = True) -> np.ndarray:
    """Zero every pixel below ``frac * peak`` activation, keeping only the core.

    This strips peripheral / background leakage (eyes, skull margins) from a
    Grad-CAM map so the red hotspot - and the segmentation seed derived from it -
    sits strictly inside the high-energy tumor core. ``frac=0.5`` keeps only
    activations >= 50% of the single hottest pixel. Vectorized; no Python loops.
    """
    h = np.asarray(heatmap, dtype=np.float32)
    peak = float(h.max())
    if peak <= 0.0:
        return h
    core = np.where(h >= frac * peak, h, 0.0).astype(np.float32)
    if renormalize:
        m = float(core.max())
        if m > 0:
            core = core / m
    return core


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def overlay_heatmap(rgb_image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.45,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Resize `heatmap` to match `rgb_image`, apply a colormap, and alpha-blend.

    Args:
        rgb_image: uint8 RGB array (H, W, 3)
        heatmap: float array in [0, 1], any shape
        alpha: heatmap weight (0..1). Image weight is (1 - alpha).
        colormap: any cv2.COLORMAP_*

    Returns:
        uint8 RGB array, same shape as rgb_image.
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    h, w = rgb_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # Slight Gaussian smoothing — kills the blocky low-res look from the conv map
    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=3)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb_image, 1 - alpha, colored, alpha, 0)
    return overlay


def focus_score(heatmap: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Quantify how *focused* a heatmap is. Lower 'spread' = tighter localization.

    Returns dict with:
        - peak: max activation value
        - mean: mean activation
        - spread_pct: percentage of pixels above `threshold` (lower is tighter)
        - center_of_mass: (row, col) of weighted centroid
    """
    h = np.asarray(heatmap, dtype=np.float64)
    h = h / (h.max() + 1e-10)

    mask = h > threshold
    spread_pct = 100.0 * mask.sum() / h.size

    # Weighted centroid
    ys, xs = np.indices(h.shape)
    total = h.sum() + 1e-10
    cy = float((ys * h).sum() / total)
    cx = float((xs * h).sum() / total)

    return {
        "peak": float(h.max()),
        "mean": float(h.mean()),
        "spread_pct": float(spread_pct),
        "center_of_mass": (cy, cx),
    }


def make_xai_figure(rgb_image: np.ndarray,
                    gradcam_heatmap: np.ndarray,
                    gradcam_pp_heatmap: np.ndarray,
                    predicted_class: str | None = None,
                    confidence: float | None = None,
                    figsize: tuple[float, float] = (16, 5)) -> Figure:
    """
    Build a 1x4 medical-style figure: Original | Grad-CAM overlay |
    Grad-CAM++ overlay | side-by-side comparison verdict.

    Returns the matplotlib Figure (not shown — caller decides).
    """
    overlay_cam = overlay_heatmap(rgb_image, gradcam_heatmap, alpha=0.45)
    overlay_pp = overlay_heatmap(rgb_image, gradcam_pp_heatmap, alpha=0.45)

    score_cam = focus_score(gradcam_heatmap)
    score_pp = focus_score(gradcam_pp_heatmap)

    # The method with the smaller spread_pct is typically the better localizer
    if score_pp["spread_pct"] < score_cam["spread_pct"]:
        winner = "Grad-CAM++"
        winner_color = "#10B981"
    elif score_cam["spread_pct"] < score_pp["spread_pct"]:
        winner = "Grad-CAM"
        winner_color = "#10B981"
    else:
        winner = "Tie"
        winner_color = "#6B7280"

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(1, 4, wspace=0.12)

    # --- 1. Original ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_image)
    ax1.set_title("Original MRI", fontsize=13, fontweight="bold", pad=10)
    ax1.axis("off")

    # --- 2. Grad-CAM overlay ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(overlay_cam)
    ax2.set_title(
        f"Grad-CAM\nspread: {score_cam['spread_pct']:.1f}% • peak: {score_cam['peak']:.2f}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax2.axis("off")

    # --- 3. Grad-CAM++ overlay ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay_pp)
    ax3.set_title(
        f"Grad-CAM++\nspread: {score_pp['spread_pct']:.1f}% • peak: {score_pp['peak']:.2f}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax3.axis("off")

    # --- 4. Verdict panel (pure matplotlib, no extra image) ---
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    title_lines = []
    if predicted_class is not None:
        title_lines.append(f"Prediction: {predicted_class.upper()}")
    if confidence is not None:
        title_lines.append(f"Confidence: {confidence * 100:.2f}%")

    ax4.text(0.5, 0.95, "\n".join(title_lines) if title_lines else "Summary",
             ha="center", va="top", fontsize=12, fontweight="bold",
             transform=ax4.transAxes)

    ax4.text(0.5, 0.72,
             "Better localization:",
             ha="center", va="center", fontsize=10, color="#374151",
             transform=ax4.transAxes)
    ax4.text(0.5, 0.62, winner,
             ha="center", va="center", fontsize=18, fontweight="bold",
             color=winner_color, transform=ax4.transAxes)

    # Small metric block
    metrics = (
        "Spread % (lower = tighter focus):\n"
        f"  • Grad-CAM    : {score_cam['spread_pct']:.1f}%\n"
        f"  • Grad-CAM++ : {score_pp['spread_pct']:.1f}%\n\n"
        "Peak activation:\n"
        f"  • Grad-CAM    : {score_cam['peak']:.2f}\n"
        f"  • Grad-CAM++ : {score_pp['peak']:.2f}"
    )
    ax4.text(0.05, 0.35, metrics, ha="left", va="center",
             fontsize=9, family="monospace", color="#1F2937",
             transform=ax4.transAxes)

    fig.suptitle(
        "Explainable AI — Grad-CAM vs Grad-CAM++ Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig
