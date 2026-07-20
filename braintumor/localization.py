"""improvements/localization.py - Honest Grad-CAM-based tumor region localization.

REPLACES the original `tumor_analysis()` in notebook cell 31, whose Otsu+contour
pipeline measured BRAIN AREA, not tumor area (the largest contour after Otsu on
an MRI is the brain against the dark background, not a lesion).

This module localizes the region the model attended to when making its
prediction. The output is described as "model attention region" - which it
literally is - rather than as a ground-truth tumor mask, which would require a
real segmentation model trained on BraTS-style annotations.

Pipeline:
    1. Compute Grad-CAM (or Grad-CAM++) heatmap on the predicted class.
    2. Threshold the heatmap at `heatmap_threshold * max` to get an attention mask.
    3. Find contours in the attention mask; keep the largest with area >= min_area.
    4. Compute bbox, center, area for that region.
    5. Independently estimate a brain mask via Otsu on grayscale; report
       attention-region coverage as a % of brain area (not % of image),
       which is the clinically more meaningful denominator.
    6. Return a structured dict + render a 4-panel visualization.

Public API:
    analyze_attention_region(model, img_path, **kwargs) -> dict
    visualize_localization(...) -> matplotlib Figure

Both are designed to be called from improvements.predict.advanced_predict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Size buckets (% of brain area)
_SIZE_BUCKETS = [
    (5, "very_small"),
    (15, "small"),
    (30, "medium"),
    (101, "large"),
]


def _size_category(brain_coverage_pct: float) -> str:
    for thr, name in _SIZE_BUCKETS:
        if brain_coverage_pct < thr:
            return name
    return "large"


# ---------------------------------------------------------------------------
# Brain mask via Otsu (used as the denominator for coverage %)
# ---------------------------------------------------------------------------
def _brain_mask(rgb_uint8: np.ndarray) -> Tuple[np.ndarray, int]:
    """Quick brain-region mask. Otsu on grayscale, take the largest contour.

    Returns (mask_uint8 with values {0,255}, brain_area_in_pixels).
    """
    import cv2

    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Fill holes (skull stripping leaves dark interior in some sequences)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(otsu)
    brain_area = 0
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        brain_area = int(cv2.contourArea(biggest))
        cv2.drawContours(mask, [biggest], -1, color=255, thickness=cv2.FILLED)
    return mask, brain_area


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze_attention_region(
    model,
    img_path: str,
    *,
    img_size: int = 299,
    class_names: Optional[List[str]] = None,
    heatmap_threshold: float = 0.5,
    min_area_pct_of_brain: float = 0.3,
    method: str = "gradcam++",
    target_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> Dict:
    """Localize the model's attention region for a single image.

    Parameters
    ----------
    model : tf.keras.Model
        Trained classifier (4-class or 3-class softmax).
    img_path : str
        Path to MRI image.
    img_size : int
        Resize target for the model input.
    class_names : list[str] | None
        Names matching the model output. Defaults to config.CLASSES_4 when len=4.
    heatmap_threshold : float
        Threshold on normalized heatmap to form the attention mask (0.5 = top half).
    min_area_pct_of_brain : float
        Discard attention contours smaller than this % of the brain mask area.
    method : "gradcam" | "gradcam++"
        Which Grad-CAM variant to use.
    target_layer_name : str | None
        Name of the conv layer for Grad-CAM. Auto-detected if None.
    pred_index : int | None
        Class index to explain. If None, uses the model's argmax prediction.
    save_path : Path | None
        If given, also writes a 4-panel visualization to this path.

    Returns
    -------
    dict with keys:
        prediction, confidence, per_class_probabilities,
        attention_region:
            present (bool), bbox{x,y,w,h}, center{x,y}, area_px,
            image_coverage_pct, brain_coverage_pct,
            brain_area_px,
        size: { area_px, brain_coverage_pct, category },
        xai: { method, focus_spread_pct, peak_activation,
               mean_activation, center_of_mass{x,y} },
        meta: { img_size, heatmap_threshold, target_layer, ... }
    """
    import cv2
    from PIL import Image
    import tensorflow as tf

    # Local imports: keep braintumor.xai as the single source of truth for Grad-CAM
    try:
        from .xai import (
            find_last_conv_layer, build_grad_model,
            compute_gradcam, compute_gradcam_plus_plus,
            focus_score,
        )
    except ImportError:
        # Fallback to the in-package xai
        from .xai import explain  # type: ignore
        find_last_conv_layer = build_grad_model = None
        compute_gradcam = compute_gradcam_plus_plus = focus_score = None
        _legacy = True
    else:
        _legacy = False

    if class_names is None:
        from . import config
        n_out = int(model.output_shape[-1])
        class_names = config.CLASSES_4 if n_out == 4 else config.CLASSES_3

    # ---- load image ----
    pil = Image.open(img_path).convert("RGB")
    pil_resized = pil.resize((img_size, img_size))
    x_in = np.asarray(pil_resized, dtype=np.float32)[None, ...]  # raw [0,255]
    rgb_full = np.asarray(pil, dtype=np.uint8)
    rgb_resized = np.asarray(pil_resized, dtype=np.uint8)

    # ---- prediction ----
    probs = model.predict(x_in, verbose=0)[0]
    if pred_index is None:
        pred_index = int(np.argmax(probs))
    pred_class = class_names[pred_index]
    confidence = float(probs[pred_index])

    # ---- Grad-CAM ----
    if not _legacy:
        layer_name = target_layer_name or find_last_conv_layer(model)
        grad_model = build_grad_model(model, layer_name)
        if method.replace("-", "").lower() in ("gradcampp", "gradcam++", "++"):
            heatmap, _, _ = compute_gradcam_plus_plus(grad_model, x_in,
                                                     pred_index=pred_index)
            method_used = "grad-cam++"
        else:
            heatmap, _, _ = compute_gradcam(grad_model, x_in, pred_index=pred_index)
            method_used = "grad-cam"
        cam_focus = focus_score(heatmap)
    else:
        heatmap, _ = explain(model, img_path, img_size, class_names,
                             method=method, class_idx=pred_index,
                             target_layer_name=target_layer_name)
        method_used = "grad-cam++" if "++" in method else "grad-cam"
        cam_focus = {"peak": float(heatmap.max()),
                     "mean": float(heatmap.mean()),
                     "spread_pct": float(100 * (heatmap > 0.5).sum() / heatmap.size),
                     "center_of_mass": (0.0, 0.0)}
        layer_name = target_layer_name or "auto"

    # ---- resize heatmap to RESIZED image space (so coords match the input the model saw) ----
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_resized = np.clip(heatmap_resized, 0, 1).astype(np.float32)

    # ---- attention mask via threshold ----
    attention_mask = (heatmap_resized >= heatmap_threshold).astype(np.uint8) * 255

    # ---- brain mask (in resized space) ----
    brain_mask, brain_area = _brain_mask(rgb_resized)
    brain_area = max(brain_area, 1)

    # Constrain attention to within the brain to avoid edge artifacts
    attention_mask = cv2.bitwise_and(attention_mask, brain_mask)

    # ---- contour extraction ----
    contours, _ = cv2.findContours(attention_mask,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(50, int(min_area_pct_of_brain / 100.0 * brain_area))
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    region_info: Dict
    if not valid:
        region_info = {
            "present": False, "bbox": None, "center": None,
            "area_px": 0, "image_coverage_pct": 0.0,
            "brain_coverage_pct": 0.0, "brain_area_px": brain_area,
            "note": "No attention region above threshold within brain mask.",
        }
        size_info = None
    else:
        biggest = max(valid, key=cv2.contourArea)
        area = int(cv2.contourArea(biggest))
        x, y, w, h = cv2.boundingRect(biggest)
        M = cv2.moments(biggest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        img_area = img_size * img_size
        brain_cov = 100.0 * area / brain_area
        region_info = {
            "present": True,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "center": {"x": int(cx), "y": int(cy)},
            "area_px": area,
            "image_coverage_pct": round(100.0 * area / img_area, 3),
            "brain_coverage_pct": round(brain_cov, 3),
            "brain_area_px": int(brain_area),
        }
        size_info = {
            "area_px": area,
            "brain_coverage_pct": round(brain_cov, 3),
            "category": _size_category(brain_cov),
        }

    # ---- XAI summary ----
    cy_cm, cx_cm = cam_focus["center_of_mass"] if "center_of_mass" in cam_focus else (0.0, 0.0)
    xai_info = {
        "method": method_used,
        "focus_spread_pct": round(float(cam_focus["spread_pct"]), 2),
        "peak_activation": round(float(cam_focus["peak"]), 4),
        "mean_activation": round(float(cam_focus["mean"]), 4),
        "center_of_mass": {"x": float(cx_cm), "y": float(cy_cm)},
        "target_layer": str(layer_name),
    }

    result = {
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "is_tumor": pred_class != "notumor",
        "per_class_probabilities": {
            class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
        },
        "attention_region": region_info,
        "size": size_info,
        "xai": xai_info,
        "meta": {
            "img_size": img_size,
            "heatmap_threshold": heatmap_threshold,
            "method": method_used,
            "image_path": str(img_path),
        },
    }

    if save_path is not None:
        try:
            fig = visualize_localization(
                rgb_resized, heatmap_resized, attention_mask, region_info,
                prediction=pred_class, confidence=confidence,
            )
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            result["meta"]["visualization_error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Visualization (4-panel)
# ---------------------------------------------------------------------------
def visualize_localization(
    rgb_uint8: np.ndarray,
    heatmap: np.ndarray,
    attention_mask: np.ndarray,
    region_info: Dict,
    *,
    prediction: str,
    confidence: float,
):
    """4-panel figure: Original | Heatmap | Attention mask | Overlay with bbox+center."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build heatmap-colored overlay
    hm_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_uint8, 0.6, hm_color, 0.4, 0)

    # Annotate overlay with bbox + center if present
    if region_info["present"]:
        bb = region_info["bbox"]
        cv2.rectangle(overlay, (bb["x"], bb["y"]),
                      (bb["x"] + bb["w"], bb["y"] + bb["h"]),
                      color=(0, 255, 0), thickness=2)
        c = region_info["center"]
        cv2.drawMarker(overlay, (c["x"], c["y"]), color=(255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    axes[0].imshow(rgb_uint8)
    axes[0].set_title("Original (resized)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(attention_mask, cmap="gray")
    axes[2].set_title("Attention mask\n(thresholded, brain-constrained)",
                      fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    title = f"Localization - {prediction.upper()}"
    if region_info["present"]:
        title += f"\nbrain coverage: {region_info['brain_coverage_pct']:.1f}%"
    else:
        title += "\n(no region above threshold)"
    axes[3].set_title(title, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle(
        f"Model attention localization  |  predicted {prediction.upper()}  "
        f"(confidence {confidence*100:.1f}%)\n"
        "Note: this shows where the MODEL ATTENDED, not a clinically validated tumor mask.",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig
