"""improvements/predict.py - Unified inference pipeline (replaces notebook cell 31).

Combines, in a single clean entry point:
    1. Image loading + preprocessing
    2. Classifier prediction
    3. Faithful Grad-CAM and Grad-CAM++ via xai_utils (NOT the broken random-head
       version from notebook cell 31)
    4. Proper attention-region localization via improvements.localization (NOT
       the Otsu-on-whole-brain version that mismeasured brain area as tumor area)
    5. A structured PredictionResult that maps to reporting.schema for LLM use

To use from the notebook, replace cell 31 with:

    from improvements.predict import Predictor
    predictor = Predictor(
        model_path=r"D:\\Gradeproject\\stadge2\\brain_tumor_model.keras",
        class_indices_path="class_indices.json",
        img_size=299,
    )
    result = predictor.predict(
        r"D:\\Gradeproject\\Datasets\\archive (2)\\Testing\\meningioma\\Te-me_3.jpg",
        show=True,
    )

Programmatic use (GUI, FastAPI, batch jobs):

    predictor = Predictor(model_path="...keras", img_size=299)
    result = predictor.predict("path/to/image.jpg", show=False)
    json_out = result["json"]               # JSON-serializable dict
    fig1 = result["prob_figure"]            # matplotlib Figure
    fig2 = result["localization_figure"]    # matplotlib Figure
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Class-indices loading: NEVER hardcode; load from disk and fall back to config.
# ---------------------------------------------------------------------------
def _load_class_indices(path: Optional[str | Path]) -> List[str]:
    """Returns class names in model output-index order (0..N-1)."""
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            # Accept either {"0": "glioma", ...} or {"glioma": 0, ...}
            if all(str(k).isdigit() for k in d.keys()):
                return [d[str(i)] for i in range(len(d))]
            return [k for k, _ in sorted(d.items(), key=lambda kv: int(kv[1]))]
    # Fallback
    try:
        from . import config
        return list(config.CLASSES_4)
    except Exception:
        return ["glioma", "meningioma", "notumor", "pituitary"]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class Predictor:
    """Single-model inference with faithful XAI and proper localization."""

    def __init__(
        self,
        model_path: str | Path,
        class_indices_path: Optional[str | Path] = "class_indices.json",
        img_size: int = 299,
        rescale_to_unit: bool = False,
    ):
        """
        Parameters
        ----------
        model_path : path to a .keras model
        class_indices_path : optional path to class_indices.json (see notebook cell A)
        img_size : input resolution the model expects
        rescale_to_unit : True for the OLD Xception notebook model that used /255 inside
                          training; False (default) for EfficientNet-family models which
                          do their own preprocessing on raw [0, 255].
        """
        import tensorflow as tf

        self.model_path = Path(model_path)
        self.img_size = int(img_size)
        self.rescale_to_unit = bool(rescale_to_unit)
        self.class_names = _load_class_indices(class_indices_path)

        print(f"[predict] loading model: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

        n_out = int(self.model.output_shape[-1])
        if n_out != len(self.class_names):
            print(f"[predict] WARNING: model output={n_out} but class_names has "
                  f"{len(self.class_names)} entries. Truncating to model output.")
            self.class_names = self.class_names[:n_out]

        # Pre-build Grad-CAM model once (expensive otherwise)
        try:
            from .xai import find_last_conv_layer, build_grad_model
            self._last_conv = find_last_conv_layer(self.model)
            self._grad_model = build_grad_model(self.model, self._last_conv)
            print(f"[predict] Grad-CAM tap: {self._last_conv}")
        except Exception as exc:
            print(f"[predict] WARNING: could not build Grad-CAM model: {exc}")
            self._last_conv = None
            self._grad_model = None

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    def _preprocess(self, img_path: str):
        pil = Image.open(img_path).convert("RGB")
        pil_resized = pil.resize((self.img_size, self.img_size))
        arr = np.asarray(pil_resized, dtype=np.float32)
        if self.rescale_to_unit:
            arr = arr / 255.0
        return pil, pil_resized, arr[None, ...]  # batch dim

    def predict(
        self, img_path: str, *,
        show: bool = False,
        run_xai: bool = True,
        run_localization: bool = True,
        heatmap_threshold: float = 0.5,
    ) -> Dict:
        """Full pipeline. Returns dict ready for reporting / GUI / API.

        Output keys:
            "json": structured prediction dict (JSON-serializable, LLM-ready)
            "prob_figure": matplotlib Figure with the image + probability bars
            "localization_figure": matplotlib Figure (only if run_localization)
            "xai_figure": matplotlib Figure (only if run_xai)
            "model_path", "class_names"
        """
        if not Path(img_path).is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        pil, pil_resized, x_in = self._preprocess(img_path)

        # ---- prediction ----
        probs = self.model.predict(x_in, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        per_class = {self.class_names[i]: float(probs[i])
                     for i in range(len(self.class_names))}

        # ---- localization (Phase 6) ----
        loc_result = None
        loc_fig = None
        if run_localization and pred_class != "notumor":
            try:
                from .localization import analyze_attention_region, visualize_localization
                loc_result = analyze_attention_region(
                    self.model, img_path,
                    img_size=self.img_size,
                    class_names=self.class_names,
                    heatmap_threshold=heatmap_threshold,
                    pred_index=pred_idx,
                )
            except Exception as exc:
                loc_result = {"error": str(exc)}

        # ---- XAI side-by-side (Grad-CAM vs Grad-CAM++) ----
        xai_fig = None
        xai_summary = None
        if run_xai and pred_class != "notumor" and self._grad_model is not None:
            try:
                from .xai import (
                    compute_gradcam, compute_gradcam_plus_plus,
                    focus_score, make_xai_figure,
                )
                hm_cam, _, _ = compute_gradcam(self._grad_model, x_in, pred_index=pred_idx)
                hm_pp, _, _ = compute_gradcam_plus_plus(self._grad_model, x_in, pred_index=pred_idx)
                f_cam = focus_score(hm_cam)
                f_pp = focus_score(hm_pp)
                better = ("Grad-CAM++" if f_pp["spread_pct"] < f_cam["spread_pct"]
                          else "Grad-CAM" if f_cam["spread_pct"] < f_pp["spread_pct"]
                          else "Tie")
                xai_summary = {
                    "gradcam_focus":   {k: float(v) if not isinstance(v, tuple) else list(v)
                                        for k, v in f_cam.items()},
                    "gradcampp_focus": {k: float(v) if not isinstance(v, tuple) else list(v)
                                        for k, v in f_pp.items()},
                    "better_method": better,
                }
                if show:
                    xai_fig = make_xai_figure(
                        rgb_image=np.asarray(pil, dtype=np.uint8),
                        gradcam_heatmap=hm_cam,
                        gradcam_pp_heatmap=hm_pp,
                        predicted_class=pred_class,
                        confidence=confidence,
                    )
            except Exception as exc:
                xai_summary = {"error": str(exc)}

        # ---- probability figure ----
        prob_fig = None
        if show:
            prob_fig = self._render_prob_figure(pil_resized, probs, pred_idx)

        # ---- structured JSON output ----
        out_json = {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "is_tumor": pred_class != "notumor",
            "tumor_type": None if pred_class == "notumor" else pred_class,
            "per_class_probabilities": {k: round(v, 4) for k, v in per_class.items()},
            "attention_region": (loc_result or {}).get("attention_region"),
            "size": (loc_result or {}).get("size"),
            "xai_summary": xai_summary,
            "segmentation_summary": None,
            "model_info": {
                "model_path": str(self.model_path),
                "img_size": self.img_size,
                "class_names": list(self.class_names),
                "rescale_to_unit": self.rescale_to_unit,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "image_path": str(img_path),
        }

        # ---- show (notebook mode) ----
        if show:
            import matplotlib.pyplot as plt
            plt.show()
            self._print_summary(out_json)

        return {
            "json": out_json,
            "prob_figure": prob_fig,
            "localization_figure": loc_fig,
            "xai_figure": xai_fig,
            "model_path": str(self.model_path),
            "class_names": list(self.class_names),
        }

    # ----------------------------------------------------------------
    # Rendering helpers
    # ----------------------------------------------------------------
    def _render_prob_figure(self, pil_resized, probs, pred_idx):
        import matplotlib.pyplot as plt

        labels = list(self.class_names)
        pred_class = labels[pred_idx]
        conf = float(probs[pred_idx])

        fig = plt.figure(figsize=(14, 5), facecolor="white")
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(pil_resized)
        color = "#10B981" if pred_class == "notumor" else "#DC2626"
        symbol = "OK" if pred_class == "notumor" else "!"
        ax1.set_title(f"[{symbol}] {pred_class.upper()}  ({conf*100:.1f}%)",
                      fontsize=13, fontweight="bold", color=color, pad=10)
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        bar_colors = ["#DC2626" if i == pred_idx else "#3B82F6" for i in range(len(labels))]
        bars = ax2.barh(labels, probs, color=bar_colors)
        ax2.set_xlim(0, 1)
        ax2.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
        ax2.set_title("Class probabilities", fontsize=13, fontweight="bold", pad=10)
        ax2.set_xlabel("Probability")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    @staticmethod
    def _print_summary(out_json):
        print("=" * 60)
        print(" BRAIN TUMOR PREDICTION")
        print("=" * 60)
        print(f"  Predicted     : {out_json['prediction'].upper()}")
        print(f"  Confidence    : {out_json['confidence']*100:.2f}%")
        if out_json["is_tumor"] and out_json.get("size"):
            sz = out_json["size"]
            print(f"  Region size   : {sz['category']} ({sz['brain_coverage_pct']:.1f}% of brain)")
        if out_json.get("xai_summary") and "better_method" in out_json["xai_summary"]:
            print(f"  Better XAI    : {out_json['xai_summary']['better_method']}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Notebook drop-in: backward-compatible `advanced_predict(img_path, show=True)`
# ---------------------------------------------------------------------------
def advanced_predict(
    img_path: str,
    model_path: str = r"D:\Gradeproject\stadge2\brain_tumor_model.keras",
    class_indices_path: str = "class_indices.json",
    img_size: int = 299,
    rescale_to_unit: bool = True,
    show: bool = True,
) -> Dict:
    """Drop-in replacement for the notebook's `advanced_predict`.

    Loads the model and class indices once per call (slow for batch use; use the
    Predictor class directly for that).

    The default `rescale_to_unit=True` matches the original notebook's Xception
    model, which was trained with `rescale=1/255` in the data generator.
    """
    predictor = Predictor(
        model_path=model_path,
        class_indices_path=class_indices_path,
        img_size=img_size,
        rescale_to_unit=rescale_to_unit,
    )
    return predictor.predict(img_path, show=show)
