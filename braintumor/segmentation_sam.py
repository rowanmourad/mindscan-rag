"""braintumor/segmentation_sam.py - MedSAM / SAM box-prompted delineation.

The "localize with Grad-CAM, delineate with SAM" cascade. Grad-CAM gives a
reliable tumor LOCATION (a bounding box) but a poor boundary; this module hands
that box to a foundation segmenter (MedSAM / SAM), which returns a tight,
edge-accurate binary mask. The classifier weights are never touched, so this is
fully decoupled (no catastrophic-forgetting risk).

Backends, tried in order (first available wins):
  1. HuggingFace ``transformers`` (``SamModel`` + ``SamProcessor``) - works with
     ``facebook/sam-vit-base`` and MedSAM HF ports (e.g. ``wanglab/medsam-vit-base``).
  2. ``segment_anything`` package + a local checkpoint (SAM or MedSAM vit_b .pth).
If neither is present, ``available`` is False and callers fall back to the
classical segmenter - so the app boots even with no SAM weights downloaded.

Config (args take precedence over env):
    BRAINTUMOR_SAM_HF_ID   HF model id (default tries MedSAM then SAM-base)
    BRAINTUMOR_SAM_CKPT    path to a local .pth checkpoint (segment_anything)
    BRAINTUMOR_SAM_TYPE    model_type for segment_anything (default 'vit_b')

To enable (one of):
    pip install "transformers>=4.40"        # then a SAM/MedSAM HF id auto-downloads
    pip install segment-anything            # + set BRAINTUMOR_SAM_CKPT to a .pth
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .segmentation import SegmentationResult
from .preprocessing import brain_mask

_DEFAULT_HF_IDS = ["wanglab/medsam-vit-base", "facebook/sam-vit-base"]


def _pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SamSegmenter:
    """Box-promptable MedSAM/SAM wrapper. ``available`` is False when no backend
    can be loaded, so callers can fall back cleanly."""

    def __init__(self, hf_id: Optional[str] = None, checkpoint: Optional[str] = None,
                 model_type: Optional[str] = None, device: Optional[str] = None,
                 verbose: bool = True):
        self.device = _pick_device(device)
        self.backend: Optional[str] = None
        self.name = "sam"
        self.model = self.processor = self.predictor = None

        hf_id = hf_id or os.environ.get("BRAINTUMOR_SAM_HF_ID")
        checkpoint = checkpoint or os.environ.get("BRAINTUMOR_SAM_CKPT")
        model_type = model_type or os.environ.get("BRAINTUMOR_SAM_TYPE", "vit_b")

        if self._try_hf(hf_id, verbose):
            return
        if self._try_sam_pkg(checkpoint, model_type, verbose):
            return
        if verbose:
            print("[sam] no MedSAM/SAM backend (install transformers + a SAM/MedSAM "
                  "id, or segment-anything + a checkpoint). Using classical fallback.")

    # ------------------------------------------------------------------
    def _try_hf(self, hf_id, verbose) -> bool:
        try:
            from transformers import SamModel, SamProcessor
        except Exception:
            return False
        ids = [hf_id] if hf_id else _DEFAULT_HF_IDS
        for mid in ids:
            try:
                self.model = SamModel.from_pretrained(mid).to(self.device).eval()
                self.processor = SamProcessor.from_pretrained(mid)
                self.backend, self.name = "transformers", mid
                if verbose:
                    print(f"[sam] loaded '{mid}' via transformers on {self.device}")
                return True
            except Exception as exc:
                if verbose:
                    print(f"[sam] transformers load failed for '{mid}': {type(exc).__name__}")
        return False

    def _try_sam_pkg(self, checkpoint, model_type, verbose) -> bool:
        if not checkpoint or not Path(checkpoint).exists():
            return False
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(self.device).eval()
            self.predictor = SamPredictor(sam)
            self.backend = "segment_anything"
            self.name = f"{model_type}:{Path(checkpoint).name}"
            if verbose:
                print(f"[sam] loaded '{self.name}' via segment_anything on {self.device}")
            return True
        except Exception as exc:
            if verbose:
                print(f"[sam] segment_anything load failed: {exc}")
            return False

    @property
    def available(self) -> bool:
        return self.backend is not None

    # ------------------------------------------------------------------
    def _raw_mask(self, rgb: np.ndarray, box_xyxy) -> np.ndarray:
        """Run the backend and return a uint8 {0,1} mask at ``rgb`` resolution."""
        import torch
        if self.backend == "transformers":
            inputs = self.processor(rgb, input_boxes=[[list(map(float, box_xyxy))]],
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            masks = self.processor.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu())[0][0]      # (num_masks, H, W)
            scores = out.iou_scores.cpu().numpy().reshape(-1)
            best = int(scores.argmax())
            return masks[best].numpy().astype(np.uint8)
        # segment_anything
        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            box=np.array(box_xyxy, dtype=np.float32), multimask_output=True)
        return masks[int(np.argmax(scores))].astype(np.uint8)

    def segment(self, image_rgb: np.ndarray, bbox_xywh,
                *, constrain_to_brain: bool = True) -> SegmentationResult:
        """Delineate the tumor inside ``bbox_xywh`` (Grad-CAM box). Returns a
        ``SegmentationResult`` so the rest of the pipeline is unchanged."""
        import cv2
        rgb = np.asarray(image_rgb)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        H, W = rgb.shape[:2]
        x, y, w, h = (int(v) for v in bbox_xywh)
        box_xyxy = [max(0, x), max(0, y), min(W - 1, x + w), min(H - 1, y + h)]

        res = SegmentationResult(present=False, img_size=W,
                                 method=f"medsam[{self.backend}]")
        bmask, _ = brain_mask(rgb)
        brain_bin = (bmask > 0).astype(np.uint8)
        res.brain_area_px = max(int(brain_bin.sum()), 1)

        try:
            m = (self._raw_mask(rgb, box_xyxy) > 0).astype(np.uint8)
        except Exception as exc:
            res.warnings.append(f"SAM inference failed: {exc}")
            return res
        if constrain_to_brain:
            m = cv2.bitwise_and(m, brain_bin)

        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            res.warnings.append("SAM returned an empty in-brain mask.")
            return res
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == big).astype(np.uint8)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(cnts, key=cv2.contourArea) if cnts else None
        bx, by, bw, bh = cv2.boundingRect(contour) if contour is not None else (0, 0, 0, 0)
        M = cv2.moments(mask, binaryImage=True)
        cx = int(M["m10"] / (M["m00"] + 1e-8)); cy = int(M["m01"] / (M["m00"] + 1e-8))
        area = int(mask.sum())

        res.present = True
        res.mask = mask
        res.contour = contour
        res.bbox = {"x": bx, "y": by, "w": bw, "h": bh}
        res.center = {"x": cx, "y": cy}
        res.area_px = area
        res.brain_coverage_pct = min(100.0, 100.0 * area / res.brain_area_px)
        res.warnings.append("MedSAM/SAM delineation prompted by the Grad-CAM box "
                            "(research use; not a clinically validated mask).")
        return res


# ---------------------------------------------------------------------------
# Grad-CAM -> localization bbox  (the reliable "red box")
# ---------------------------------------------------------------------------
def gradcam_localization_box(grad_model, x, pred_index, img_size: int,
                             brain_bin: Optional[np.ndarray] = None,
                             frac: float = 0.50, pad_frac: float = 0.05
                             ) -> Optional[Tuple[int, int, int, int]]:
    """Derive a (generous) localization box from the in-brain Grad-CAM peak blob.

    Grad-CAM is poor at boundaries but good at WHERE - this returns a box around
    the peak's connected component (threshold ``frac*peak``) with padding, to be
    used purely as the SAM prompt (SAM does the tight delineation inside it).
    Returns (x, y, w, h) or None.
    """
    import cv2
    from .xai import compute_gradcam
    hm, _, _ = compute_gradcam(grad_model, x, pred_index=pred_index)
    hm = cv2.resize(hm.astype(np.float32), (img_size, img_size))
    hm = np.clip(hm / (hm.max() + 1e-8), 0, 1)
    if brain_bin is not None:
        hm = hm * brain_bin.astype(np.float32)
    if float(hm.max()) <= 1e-6:
        return None
    py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
    peak = float(hm[py, px])
    hot = (hm >= frac * peak).astype(np.uint8)
    _, lab = cv2.connectedComponents(hot, connectivity=8)
    comp = (lab == lab[py, px]).astype(np.uint8)
    ys, xs = np.where(comp > 0)
    pad = int(pad_frac * img_size)
    x0 = max(0, int(xs.min()) - pad); y0 = max(0, int(ys.min()) - pad)
    x1 = min(img_size, int(xs.max()) + pad); y1 = min(img_size, int(ys.max()) + pad)
    return (x0, y0, x1 - x0, y1 - y0)


# ---------------------------------------------------------------------------
# Grad-CAM-FREE localization box (intensity saliency) - decouples MedSAM from XAI
# ---------------------------------------------------------------------------
def _brain_interior(gray_shape, brain_bin, erode_frac: float = 0.08):
    """Erode the brain mask inward to DROP the hyperintense skull/scalp rim, which
    otherwise dominates an intensity search (the skull is usually the brightest
    structure). Returns the interior brain mask uint8 {0,1}."""
    import cv2
    r = max(3, int(erode_frac * min(gray_shape[:2])))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode((brain_bin > 0).astype(np.uint8), k, iterations=1)
    return inner


def intensity_tumor_mask(image_rgb: np.ndarray, brain_bin: Optional[np.ndarray] = None,
                         erode_frac: float = 0.08) -> np.ndarray:
    """Grad-CAM-FREE tumor candidate (BALANCED - neither a point nor the hemisphere).

    Root-cause-stable design: instead of a single global threshold + largest blob +
    hard cap (which swings between whole-brain and a microscopic point), we:
      1. skull-strip by eroding the brain inward (drops the bright skull/scalp rim),
      2. threshold the interior at a MODERATE level (top ~20% / mean+std) to capture
         the tumor BODY, not just the peak pixel,
      3. score every connected component by SOLIDITY (compactness) within a sane
         SIZE BAND [~0.5%, 30%] of brain, and return the best-scoring one. Compact,
         mid-sized blobs (tumors) win; ring-like skull remnants and tiny specks lose.
    Returns a uint8 {0,1} mask of the tumor BODY (zeros if none qualifies)."""
    import cv2
    rgb = np.asarray(image_rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    if brain_bin is None:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_bin = (th > 0).astype(np.uint8)
    out = np.zeros(gray.shape, np.uint8)

    inner = _brain_interior(gray.shape, brain_bin, erode_frac)
    if int(inner.sum()) < 50:
        inner = (brain_bin > 0).astype(np.uint8)
    brain_area = max(int((brain_bin > 0).sum()), 1)
    vals = gray[inner > 0].astype(np.float32)
    if vals.size == 0:
        return out

    # moderate threshold -> capture the tumor BODY (not just the peak)
    thr = max(float(vals.mean() + 1.0 * vals.std()), float(np.percentile(vals, 80)))
    hot = ((gray >= thr).astype(np.uint8) & inner)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, k, iterations=1)
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, k, iterations=2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    if n <= 1:
        return out
    lo, hi = 0.005 * brain_area, 0.30 * brain_area
    best_lab, best_score = None, -1.0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        in_band = lo <= area <= hi
        # prefer compact, in-band blobs; mildly favor larger within the band
        score = solidity * (1.0 + 0.5 * (area / hi)) * (1.0 if in_band else 0.15)
        if score > best_score:
            best_score, best_lab = score, i
    if best_lab is None:
        return out
    return (lab == best_lab).astype(np.uint8)


def independent_tumor_box(image_rgb: np.ndarray, brain_bin: Optional[np.ndarray] = None,
                          pad_frac: float = 0.18, min_pad: int = 6,
                          max_frac: float = 0.45
                          ) -> Optional[Tuple[int, int, int, int]]:
    """BALANCED Grad-CAM-FREE MedSAM prompt box: the bounding box of the tumor-body
    blob (:func:`intensity_tumor_mask`) expanded by PROPORTIONAL context padding
    (``pad_frac`` of the blob size, >= ``min_pad`` px). No point-collapse - MedSAM
    gets the tumor body with surrounding context. Returns None (so the caller falls
    back) only if no plausible blob exists or the box would exceed ``max_frac`` of
    the brain. Returns (x, y, w, h)."""
    import cv2
    rgb = np.asarray(image_rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    if brain_bin is None:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        brain_bin = (th > 0).astype(np.uint8)
    brain_area = max(int((brain_bin > 0).sum()), 1)

    m = intensity_tumor_mask(image_rgb, brain_bin)
    if int(m.sum()) == 0:
        return None
    H, W = m.shape[:2]
    ys, xs = np.where(m > 0)
    bw = int(xs.max() - xs.min() + 1); bh = int(ys.max() - ys.min() + 1)
    pad = max(min_pad, int(pad_frac * max(bw, bh)))
    x0 = max(0, int(xs.min()) - pad); y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + 1 + pad); y1 = min(H, int(ys.max()) + 1 + pad)
    if (x1 - x0) * (y1 - y0) > max_frac * brain_area:
        return None                            # implausible -> let the caller fall back
    return (x0, y0, x1 - x0, y1 - y0)


# ---------------------------------------------------------------------------
# Module-level convenience (cached default segmenter)
# ---------------------------------------------------------------------------
_DEFAULT: Optional[SamSegmenter] = None


def default_sam_segmenter(**kw) -> SamSegmenter:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SamSegmenter(**kw)
    return _DEFAULT


def segment_with_sam(image, bbox_prompt, **kw) -> SegmentationResult:
    """Convenience: delineate ``image`` inside ``bbox_prompt`` (x,y,w,h) with SAM.
    Falls back to an empty result (``present=False``) when no SAM backend exists."""
    seg = default_sam_segmenter(verbose=kw.pop("verbose", False), **kw)
    if not seg.available:
        r = SegmentationResult(present=False, method="sam-unavailable")
        r.warnings.append("MedSAM/SAM not installed; cannot segment with SAM.")
        return r
    return seg.segment(image, bbox_prompt)
