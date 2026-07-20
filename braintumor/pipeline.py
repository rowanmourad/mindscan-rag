"""braintumor/pipeline.py - The single end-to-end orchestrator.

Wires every stage of the project into one connected flow:

    MRI image
      -> preprocessing
      -> classification sources (single 4-class  +  optional two-stage binary->subtype
                                  +  optional dual-backbone fusion model)
      -> decision fusion  (probability averaging over available sources)
      -> final diagnosis
      -> XAI  (Grad-CAM + Grad-CAM++ on the final class)
      -> segmentation  (XAI-guided weak mask)
      -> tumor analysis  (geometry / location / shape / intensity from the mask)
      -> structured PredictionResult  (schema)
      -> medical-style markdown report  (generate_report)
      -> LLM-ready payload  (llm_report)

It is robust to whatever is trained: with only the single 4-class model present
(the shipped default), it runs the full XAI/segmentation/analysis/report flow on
that model. When binary+subtype and/or a fusion model are also available, they
join the decision-fusion automatically - no code change needed.

Usage
-----
    from braintumor.pipeline import BrainTumorPipeline
    pipe = BrainTumorPipeline()                 # auto-loads best models
    out = pipe.analyze("path/to/mri.jpg", save_dir="artifacts/cases/demo")
    print(out["report_markdown"])
    out["result"]            # schema.PredictionResult
    out["llm_payload"]       # dict for the local LLM
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from . import config
from .registry import ModelRegistry, ResolvedModel


# Coverage(%) -> coarse size bucket (shared with localization)
def _size_category(brain_coverage_pct: float) -> str:
    for thr, name in [(5, "very_small"), (15, "small"), (30, "medium")]:
        if brain_coverage_pct < thr:
            return name
    return "large"


@dataclass
class _Source4c:
    """A source that yields a 4-class probability vector for one image."""
    name: str
    img_size: int
    rescale_to_unit: bool
    kind: str                       # "single" | "fusion" | "two_stage"
    model: object = None            # keras model (single/fusion)
    two_stage: object = None        # TwoStageClassifier (two_stage)
    weight: float = 1.0

    def predict(self, img_path: str, class_names: List[str]) -> np.ndarray:
        from .preprocessing import preprocess_array, load_rgb
        if self.kind == "two_stage":
            res = self.two_stage.predict_image(img_path)
            fp = res["full_probs"]
            return np.array([fp[c] for c in class_names], dtype=np.float32)
        rgb = load_rgb(img_path)
        x = preprocess_array(rgb, self.img_size, rescale_to_unit=self.rescale_to_unit)
        return self.model.predict(x, verbose=0)[0].astype(np.float32)


class BrainTumorPipeline:
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        *,
        enhance_input: bool = False,
        skull_strip: bool = True,
        llm_model_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.reg = registry or ModelRegistry()
        self.enhance_input = enhance_input
        # Skull-strip / neck-artifact removal before the model. Empirically
        # neutral-to-positive for the shipped B2 (95.0% vs 93.8% raw on a 80-img
        # sample) and removes background gradient distractors, so it defaults ON.
        # Set skull_strip=False to feed raw images (matches original training).
        self.skull_strip = skull_strip
        self.llm_model_dir = llm_model_dir
        self.verbose = verbose
        if verbose:
            print(self.reg.summary())

        self.class_names = self.reg.class_names(4)

        # --- primary single 4-class model (the XAI/segmentation backbone) ---
        primary = self.reg.latest_4class()
        if primary is None:
            raise RuntimeError(
                "No 4-class model found. Train one (python -m braintumor.train_4class) "
                "or place a .keras model in artifacts/models / stadge2.")
        self.primary: ResolvedModel = primary
        self.primary_model = primary.load()

        # Build the Grad-CAM model once on the primary backbone.
        from .xai import find_last_conv_layer, build_grad_model
        try:
            self._last_conv = find_last_conv_layer(self.primary_model)
            self._grad_model = build_grad_model(self.primary_model, self._last_conv)
            if verbose:
                print(f"[pipeline] Grad-CAM tap: {self._last_conv}")
        except Exception as exc:
            print(f"[pipeline] WARNING: Grad-CAM unavailable: {exc}")
            self._last_conv, self._grad_model = None, None

        # Attention U-Net segmenter (independent of Grad-CAM). Loads trained
        # weights from models/ if present; otherwise .available == False and the
        # pipeline falls back to the XAI-guided weak mask.
        self._primary_x = None       # cached preprocessed primary input (per scan)
        self._last_rgb = None         # cached decoded RGB (per scan)
        self._session_active = False  # True once a session-local clone is in use
        from .segmentation import AttentionUNetSegmenter
        from .segmentation_sam import SamSegmenter
        self._unet = AttentionUNetSegmenter(input_size=256)
        # High-tier segmenter: MedSAM/SAM prompted by the Grad-CAM box. Decoupled
        # from the classifier weights. available==False (no crash) until a SAM
        # backend + weights are installed, in which case it becomes the default.
        self._sam = SamSegmenter(verbose=verbose)
        if verbose:
            if self._sam.available:
                tier = f"MedSAM/SAM ({self._sam.name})"
            elif self._unet.available:
                tier = "Attention U-Net"
            else:
                tier = "core-seed fallback (install MedSAM/SAM for SOTA delineation)"
            print(f"[pipeline] segmenter tier: {tier}")

        # --- assemble decision-fusion sources ---
        self.sources: List[_Source4c] = [
            _Source4c(name=f"single:{primary.path.name}", img_size=primary.img_size,
                      rescale_to_unit=primary.rescale_to_unit, kind="single",
                      model=self.primary_model)
        ]
        fusion = self.reg.latest_fusion()
        if fusion is not None:
            try:
                fmodel = fusion.load()
                self.sources.append(_Source4c(
                    name=f"fusion:{fusion.path.name}", img_size=fusion.img_size,
                    rescale_to_unit=fusion.rescale_to_unit, kind="fusion",
                    model=fmodel))
            except Exception as exc:
                print(f"[pipeline] skipping fusion model (could not load): "
                      f"{type(exc).__name__}. It was saved with bare Lambda "
                      f"preprocess layers; re-save with weights or "
                      f"@register_keras_serializable to enable it.")
        if self.reg.has_two_stage():
            try:
                from .two_stage import TwoStageClassifier
                b, s = self.reg.latest_binary(), self.reg.latest_subtype()
                ts = TwoStageClassifier(
                    binary_model_path=b.path, subtype_model_path=s.path,
                    binary_img_size=b.img_size, subtype_img_size=s.img_size)
                self.sources.append(_Source4c(
                    name="two_stage(binary->subtype)", img_size=b.img_size,
                    rescale_to_unit=False, kind="two_stage", two_stage=ts))
            except Exception as exc:
                print(f"[pipeline] skipping two-stage source: {exc}")
        if verbose:
            print(f"[pipeline] decision-fusion sources: {[s.name for s in self.sources]}")

    # ------------------------------------------------------------------
    def predict_probs(self, img_path: str) -> Dict[str, object]:
        """Run every source, fuse by probability averaging -> final 4-class probs.

        The image is decoded ONCE and preprocessing is cached per (img_size,
        rescale), so two same-resolution sources (e.g. single B2 + fusion at 260)
        share a single resize. The primary source's input tensor is stashed in
        ``self._primary_x`` for reuse by the XAI stage (no redundant preprocess).
        """
        from .preprocessing import load_rgb, preprocess_array
        rgb = load_rgb(img_path)
        self._last_rgb = rgb
        _pre_cache: Dict[tuple, np.ndarray] = {}

        def _get_x(size, rescale):
            key = (int(size), bool(rescale))
            if key not in _pre_cache:
                _pre_cache[key] = preprocess_array(
                    rgb, size, rescale_to_unit=rescale,
                    skull_strip_input=self.skull_strip)
            return _pre_cache[key]

        per_source = {}
        vecs = []
        for src in self.sources:
            try:
                if src.kind == "two_stage":
                    p = src.predict(img_path, self.class_names)
                else:
                    x = _get_x(src.img_size, src.rescale_to_unit)
                    p = src.model.predict(x, verbose=0)[0].astype(np.float32)
                    if src is self.sources[0]:        # cache the primary input for XAI
                        self._primary_x = x
                per_source[src.name] = {self.class_names[i]: float(p[i]) for i in range(4)}
                vecs.append(src.weight * p)
            except Exception as exc:
                per_source[src.name] = {"error": str(exc)}
        if not vecs:
            raise RuntimeError("All classification sources failed.")
        fused = np.mean(np.stack(vecs, axis=0), axis=0)
        fused = fused / (fused.sum() + 1e-12)
        idx = int(np.argmax(fused))
        return {
            "fused_probs": fused,
            "pred_index": idx,
            "pred_class": self.class_names[idx],
            "confidence": float(fused[idx]),
            "per_source": per_source,
            "fusion_method": "probability_average",
            "n_sources": len(vecs),
        }

    # ------------------------------------------------------------------
    def analyze(
        self,
        img_path: str,
        *,
        run_xai: bool = True,
        run_segmentation: bool = True,
        run_tumor_analysis: bool = True,
        make_figures: bool = True,
        save_dir: Optional[str | Path] = None,
        patient_info: Optional[Dict] = None,
    ) -> Dict[str, object]:
        """Full end-to-end analysis of one MRI image."""
        import cv2
        from .preprocessing import load_rgb
        from .schema import (PredictionResult, TumorLocation, TumorSize,
                             XAISummary, SegmentationSummary, ModelInfo)

        img_path = str(img_path)
        if not Path(img_path).is_file():
            raise FileNotFoundError(img_path)

        # ---- 1. classification + decision fusion ----
        self._primary_x = None                  # reset per-scan caches
        clf = self.predict_probs(img_path)
        pred_idx, pred_class = clf["pred_index"], clf["pred_class"]
        confidence = clf["confidence"]
        is_tumor = pred_class != "notumor"
        # Reuse the RGB decoded inside predict_probs (avoid a redundant decode).
        original_rgb = self._last_rgb if self._last_rgb is not None else load_rgb(img_path)

        figures: Dict[str, object] = {}
        xai_summary_obj = None
        seg_summary_obj = None
        tumor_loc_obj = None
        tumor_size_obj = None
        tumor_analysis_payload = None
        seg_result = None
        xai_hm = None                 # brain-masked Grad-CAM heatmap (for the seg figure)

        # ---- 2. XAI (only meaningful for tumor classes) ----
        if run_xai and is_tumor and self._grad_model is not None:
            try:
                # Standard Grad-CAM + hard energy thresholding (tight tumor core).
                # Grad-CAM++ is intentionally OFF the default graph: it spreads to
                # full-object coverage (background leakage) and costs 3 backward
                # passes vs 1 for Grad-CAM. ``compute_gradcam_plus_plus`` remains
                # available in xai.py for an explicit comparison figure if desired.
                import cv2
                from .xai import (compute_gradcam, focus_score,
                                  hard_energy_threshold, overlay_heatmap,
                                  XAI_METHOD)
                from .preprocessing import brain_mask
                # Reuse the already-preprocessed primary input if present.
                x = self._primary_x if self._primary_x is not None else None
                if x is None:
                    from .preprocessing import preprocess_array
                    x = preprocess_array(original_rgb, self.primary.img_size,
                                         rescale_to_unit=self.primary.rescale_to_unit,
                                         skull_strip_input=self.skull_strip)
                sz = self.primary.img_size
                hm, _, _ = compute_gradcam(self._grad_model, x, pred_index=pred_idx)
                hm = cv2.resize(hm.astype("float32"), (sz, sz))
                # Brain-mask the heatmap: zero any activation outside the
                # intracranial region so Grad-CAM cannot drift onto neck/skull
                # (safe regardless of whether the input was skull-stripped).
                bmask, _ = brain_mask(cv2.resize(original_rgb, (sz, sz)))
                hm = hm * (bmask > 0).astype("float32")
                hm = hard_energy_threshold(hm, frac=0.5)   # zero < 50% of peak
                xai_hm = hm                                 # reused by the seg figure
                f = focus_score(hm)
                cy, cx = f["center_of_mass"]
                xai_summary_obj = XAISummary(
                    method=XAI_METHOD,                  # locked to "grad-cam"
                    focus_spread_pct=f["spread_pct"],
                    peak_activation=f["peak"], mean_activation=f["mean"],
                    center_of_mass_x=cx, center_of_mass_y=cy,
                    target_layer=str(self._last_conv), better_method=None)
                if make_figures:
                    figures["xai"] = self._xai_figure(
                        original_rgb, hm, pred_class, confidence, f["spread_pct"])
            except Exception as exc:
                print(f"[pipeline] XAI failed: {exc}")

        # ---- 3. segmentation: DECOUPLED from Grad-CAM. Tier order:
        #   1. clinician-verified mask (aHash cache)  -> authoritative
        #   2. trained Attention U-Net (end-to-end, no prompt)
        #   3. MedSAM prompted by an INDEPENDENT intensity box (not Grad-CAM)
        #   4. intensity blob fallback (Grad-CAM-free)
        # Grad-CAM is NOT used here at all; it remains visual-only in step 2.
        if run_segmentation and is_tumor:
            try:
                import cv2
                from .segmentation import visualize_segmentation
                from .segmentation_sam import independent_tumor_box, intensity_tumor_mask
                from .preprocessing import brain_mask
                from .clinical_feedback import lookup_cached_mask
                sz = self.primary.img_size
                resized = cv2.resize(original_rgb, (sz, sz))
                bmask, _ = brain_mask(resized)
                brain_bin = (bmask > 0).astype("uint8")
                brain_area = max(int(brain_bin.sum()), 1)

                seg_result = None
                # Tier 1: clinician-verified mask (never repeat a corrected mistake).
                hit = lookup_cached_mask(original_rgb)
                if hit is not None:
                    seg_result = self._mask_to_result(
                        hit["mask"], sz, brain_bin, brain_area,
                        f"clinician-verified(aHash d={hit['dist']})")
                # Tier 2: trained Attention U-Net (standalone, end-to-end).
                if seg_result is None and self._unet.available:
                    um = self._unet.predict_mask(resized, out_size=sz)
                    if um is not None and int(um.sum()) > 0:
                        seg_result = self._mask_to_result(
                            cv2.bitwise_and(um, brain_bin), sz, brain_bin, brain_area,
                            "attention-unet")
                # Tier 3: MedSAM prompted by a tight intensity box (no Grad-CAM).
                if (seg_result is None or not seg_result.present) and self._sam.available:
                    box = independent_tumor_box(resized, brain_bin)
                    if box is not None:
                        cand = self._sam.segment(resized, box)
                        # Reject MedSAM over-segmentation (whole-hemisphere leak):
                        # if it still covers > 40% of the brain, discard it and use
                        # the tighter intensity mask instead.
                        if (cand is not None and cand.present
                                and cand.brain_coverage_pct <= 40.0):
                            cand.method = "medsam(independent-box)"
                            seg_result = cand
                        elif cand is not None and cand.present:
                            print(f"[pipeline] MedSAM over-segmented "
                                  f"({cand.brain_coverage_pct:.0f}% of brain); "
                                  "falling back to tight intensity mask.")
                # Tier 4: intensity blob fallback (Grad-CAM-free, tight).
                if seg_result is None or not seg_result.present:
                    seg_result = self._mask_to_result(
                        intensity_tumor_mask(resized, brain_bin), sz, brain_bin,
                        brain_area, "intensity-fallback")
                # heatmap for the figure: the visual Grad-CAM from step 2 (display only)
                if seg_result is not None and seg_result.heatmap is None and xai_hm is not None:
                    seg_result.heatmap = xai_hm

                if seg_result.present:
                    sz = self.primary.img_size
                    bb, ctr = seg_result.bbox, seg_result.center
                    tumor_loc_obj = TumorLocation(
                        bbox_x=bb["x"], bbox_y=bb["y"], bbox_w=bb["w"], bbox_h=bb["h"],
                        center_x=ctr["x"], center_y=ctr["y"],
                        image_coverage_pct=100.0 * seg_result.area_px / (sz * sz),
                        brain_coverage_pct=min(seg_result.brain_coverage_pct, 100.0),
                        brain_area_px=seg_result.brain_area_px,
                        region_area_px=seg_result.area_px)
                    tumor_size_obj = TumorSize(
                        area_px=seg_result.area_px,
                        brain_coverage_pct=min(seg_result.brain_coverage_pct, 100.0),
                        category=_size_category(seg_result.brain_coverage_pct))
                    seg_summary_obj = SegmentationSummary(**{
                        k: v for k, v in seg_result.summary_dict().items()
                        if k in ("method", "mask_area_px", "mask_brain_coverage_pct",
                                 "dice_against_attention", "mask_path")})
                if make_figures and seg_result is not None:
                    figures["segmentation"] = visualize_segmentation(
                        seg_result, original_rgb, prediction=pred_class)
            except Exception as exc:
                print(f"[pipeline] segmentation failed: {exc}")

        # ---- 4. tumor analysis (geometry/shape/intensity from the mask) ----
        if run_tumor_analysis and is_tumor and seg_result is not None and seg_result.present:
            try:
                from .tumor_analysis import analyze_tumor, build_dashboard
                resized = cv2.resize(original_rgb, (self.primary.img_size,) * 2)
                report = analyze_tumor(
                    resized, seg_result.mask, predicted_class=pred_class,
                    confidence=confidence)
                # Pass the FULL report (geometry + morphometry + quadrant + shape +
                # intensity), not just the compact schema, so the LLM sees axes,
                # ellipsoidal volume estimate and quadrant localization.
                tumor_analysis_payload = report.to_dict()
                if make_figures:
                    figures["tumor_analysis"] = build_dashboard(resized, report)
            except Exception as exc:
                print(f"[pipeline] tumor analysis failed: {exc}")

        # ---- 5. probability bar figure ----
        if make_figures:
            figures["probabilities"] = self._prob_figure(
                original_rgb, clf["fused_probs"], pred_idx)

        # ---- 6. structured result ----
        result = PredictionResult(
            prediction=pred_class, confidence=confidence, is_tumor=is_tumor,
            tumor_type=None if not is_tumor else pred_class,
            per_class_probabilities={self.class_names[i]: float(clf["fused_probs"][i])
                                     for i in range(4)},
            tumor_location=tumor_loc_obj, tumor_size=tumor_size_obj,
            xai_summary=xai_summary_obj, segmentation_summary=seg_summary_obj,
            model_info=ModelInfo(
                model_path=str(self.primary.path), img_size=self.primary.img_size,
                class_names=list(self.class_names),
                rescale_to_unit=self.primary.rescale_to_unit,
                architecture=self.primary.spec.architecture,
                notes=f"decision-fusion over {clf['n_sources']} source(s): "
                      f"{', '.join(self.sources[i].name for i in range(clf['n_sources']))}"),
            image_path=img_path,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"))

        # ---- 7. report + LLM payload ----
        from .generate_report import generate_markdown_report
        from .llm_report import build_llm_payload
        report_md = generate_markdown_report(result, patient_info=patient_info,
                                             model_version=config_version())
        llm_payload = build_llm_payload(
            result, tumor_analysis=tumor_analysis_payload,
            segmentation=seg_summary_obj.__dict__ if seg_summary_obj else None,
            patient_info=patient_info)

        out = {
            "result": result,
            "result_json": result.to_dict(),
            "fusion": clf,
            "report_markdown": report_md,
            "llm_payload": llm_payload,
            "figures": figures,
            # raw arrays for downstream UIs (Streamlit canvas, overlays):
            "segmentation_mask": (seg_result.mask if seg_result is not None
                                  and seg_result.present else None),
            "segmentation_method": (seg_result.method if seg_result is not None else None),
            "image_resized": cv2.resize(original_rgb, (self.primary.img_size,) * 2),
        }

        if save_dir is not None:
            out["saved"] = self._save(out, save_dir)
        return out

    # ------------------------------------------------------------------
    def generate_llm_report(self, analyze_out: Dict[str, object]) -> str:
        """Run the optional local LLM on a previous analyze() output."""
        from .llm_report import generate_report_with_llm
        return generate_report_with_llm(
            analyze_out["llm_payload"], model_dir=self.llm_model_dir,
            fallback_result=analyze_out["result"])

    # ------------------------------------------------------------------
    # Online clinician-correction learning (SESSION-LOCAL, reversible)
    # ------------------------------------------------------------------
    def apply_session_correction(self, image_path_or_rgb, user_mask,
                                 corrected_label: str, *, steps: int = 3,
                                 lr: float = 1e-5, log: bool = True, **kw) -> Dict:
        """Guarded micro-update so the model "memorizes" a doctor's correction for
        the rest of the session.

        On first call the PRIMARY model is replaced by a session-local CLONE (the
        shared process cache stays pristine), and that clone is fine-tuned (head
        only, lr 1e-5) toward ``corrected_label``. The Grad-CAM model is rebuilt on
        the clone so subsequent overlays/segmentation reflect the updated weights.
        ``reset_session()`` restores the original cached model.
        """
        import numpy as np
        from .clinical_feedback import (clone_model_for_session,
                                        online_correction_update, log_feedback)
        from .xai import build_grad_model
        from .preprocessing import load_rgb

        rgb = (load_rgb(image_path_or_rgb)
               if isinstance(image_path_or_rgb, (str, Path)) else np.asarray(image_path_or_rgb))

        # First correction this session: clone + swap in (cache untouched).
        if not getattr(self, "_session_active", False):
            print("[session] cloning primary model for session-local learning "
                  "(shared cache untouched).")
            clone = clone_model_for_session(self.primary_model)
            self.primary_model = clone
            self.sources[0].model = clone
            self._session_active = True

        result = online_correction_update(
            self.primary_model, rgb, user_mask, corrected_label, self.class_names,
            img_size=self.primary.img_size,
            rescale_to_unit=self.primary.rescale_to_unit,
            steps=steps, lr=lr, **kw)

        # Rebuild Grad-CAM on the updated clone (head weights changed).
        try:
            self._grad_model = build_grad_model(self.primary_model, self._last_conv)
        except Exception as exc:
            print(f"[session] grad-model rebuild warning: {exc}")

        if log:
            log_feedback(
                image_path=image_path_or_rgb if isinstance(image_path_or_rgb, str) else "",
                image_rgb=None if isinstance(image_path_or_rgb, str) else rgb,
                user_mask=user_mask,
                model_prediction={"prediction": None, "confidence": None,
                                  "per_class_probabilities": result["before_probs"]},
                corrected_label=corrected_label,
                corrected_metrics={"online_update": result},
                notes="Session-local online correction (head micro-update).")
        return result

    def reset_session(self) -> None:
        """Discard session-local fine-tuning; restore the pristine cached model."""
        if not getattr(self, "_session_active", False):
            return
        from .xai import build_grad_model
        self.primary_model = self.primary.load()       # cached original (untouched)
        self.sources[0].model = self.primary_model
        try:
            self._grad_model = build_grad_model(self.primary_model, self._last_conv)
        except Exception:
            pass
        self._session_active = False
        print("[session] reset to pristine cached model.")

    # ------------------------------------------------------------------
    def _mask_to_result(self, mask, sz, brain_bin, brain_area, method):
        """Build a SegmentationResult (bbox/center/contour/area/coverage) from any
        binary mask. Used by the decoupled segmentation tiers."""
        import cv2
        from .segmentation import SegmentationResult
        res = SegmentationResult(present=False, img_size=sz,
                                 brain_area_px=int(brain_area), method=method)
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != (sz, sz):
            m = cv2.resize(m, (sz, sz), interpolation=cv2.INTER_NEAREST)
        m = cv2.bitwise_and(m, brain_bin.astype(np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            res.warnings.append("empty mask")
            return res
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(cnts, key=cv2.contourArea) if cnts else None
        bx, by, bw, bh = cv2.boundingRect(contour) if contour is not None else (0, 0, 0, 0)
        M = cv2.moments(m, binaryImage=True)
        cx = int(M["m10"] / (M["m00"] + 1e-8)); cy = int(M["m01"] / (M["m00"] + 1e-8))
        area = int(m.sum())
        res.present = True
        res.mask = m; res.contour = contour
        res.bbox = {"x": bx, "y": by, "w": bw, "h": bh}
        res.center = {"x": cx, "y": cy}
        res.area_px = area
        res.brain_coverage_pct = min(100.0, 100.0 * area / brain_area)
        return res

    # ------------------------------------------------------------------
    def _xai_figure(self, original_rgb, heatmap, pred_class, confidence, spread):
        """Fast single-map figure: original | hard-thresholded Grad-CAM overlay."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .xai import overlay_heatmap
        overlay = overlay_heatmap(original_rgb, heatmap, alpha=0.45)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
        axes[0].imshow(original_rgb); axes[0].set_title("Original MRI", fontweight="bold")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM (hard-thresholded, core)\n{pred_class.upper()} "
                          f"({confidence*100:.1f}%) - focus {spread:.1f}%",
                          fontweight="bold")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def _prob_figure(self, original_rgb, probs, pred_idx):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list(self.class_names)
        pred_class = labels[pred_idx]
        conf = float(probs[pred_idx])
        fig = plt.figure(figsize=(13, 5), facecolor="white")
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(original_rgb)
        color = "#10B981" if pred_class == "notumor" else "#DC2626"
        ax1.set_title(f"{pred_class.upper()}  ({conf*100:.1f}%)",
                      fontsize=13, fontweight="bold", color=color)
        ax1.axis("off")
        ax2 = fig.add_subplot(1, 2, 2)
        colors = ["#DC2626" if i == pred_idx else "#3B82F6" for i in range(len(labels))]
        bars = ax2.barh(labels, probs, color=colors)
        ax2.set_xlim(0, 1); ax2.bar_label(bars, fmt="%.3f", padding=4)
        ax2.set_title("Fused class probabilities", fontweight="bold")
        ax2.invert_yaxis()
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        fig.tight_layout()
        return fig

    def _save(self, out: Dict[str, object], save_dir) -> List[str]:
        import json
        import matplotlib.pyplot as plt
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        (save_dir / "report.md").write_text(out["report_markdown"], encoding="utf-8")
        saved.append(str(save_dir / "report.md"))
        (save_dir / "result.json").write_text(
            out["result"].to_json(indent=2), encoding="utf-8")
        saved.append(str(save_dir / "result.json"))
        (save_dir / "llm_payload.json").write_text(
            json.dumps(out["llm_payload"], indent=2, default=str), encoding="utf-8")
        saved.append(str(save_dir / "llm_payload.json"))
        for name, fig in out.get("figures", {}).items():
            if fig is None:
                continue
            p = save_dir / f"{name}.png"
            try:
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(str(p))
            except Exception:
                pass
        return saved


def config_version() -> str:
    from . import __version__
    return f"braintumor v{__version__}"
