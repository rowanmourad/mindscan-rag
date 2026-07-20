"""Quantitative tumor-region analysis (image-analysis only — NOT diagnosis).

This module turns a **binary segmentation mask** + the original MRI slice into a
structured set of quantitative descriptors (geometry, location, shape, intensity,
size band), a set of visualizations, and an LLM-ready JSON payload.

────────────────────────────────────────────────────────────────────────────
READ THIS FIRST — what this module is and is NOT
────────────────────────────────────────────────────────────────────────────
* It is a **mask analyzer**, not a segmenter. It does not decide *where* the
  tumor is; it measures whatever mask you hand it. **Every number it returns is
  only as valid as that mask.** Garbage mask -> confident garbage measurements.
* This project currently has **no validated tumor-segmentation model**. The
  ``heuristic_tumor_mask`` fallback below (Otsu + brain mask + largest blob) is a
  ROUGH stand-in for demos only: on brain MRI, Otsu often latches onto skull,
  fat, or ventricle rather than the tumor. For real measurements, feed a mask
  from a trained segmenter (e.g. a U-Net trained on BraTS) or, as a weak
  localizer, a thresholded Grad-CAM from ``improvements.xai``.
* All outputs are **image-analysis findings, not a medical diagnosis**. The
  size band is a geometric descriptor, not a clinical risk score. The disclaimer
  is embedded in every report and every LLM payload on purpose; do not strip it.

Design goals: TensorFlow-compatible (optional classifier hook), OpenCV-based,
modular pure functions, and outputs shaped for a downstream LLM reporting system.

Suggested location: ``improvements/tumor_analysis.py`` (additive; nothing here
imports or modifies your existing modules, so it is safe to drop in).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Embedded in every report. Do not remove — keeps the non-diagnostic framing
#: attached to the data wherever it travels (including into an LLM prompt).
DISCLAIMER = (
    "Image-analysis findings only. These are geometric/intensity measurements "
    "of a provided segmentation mask and do NOT constitute a medical diagnosis, "
    "tumor grade, or clinical risk assessment. Validity depends entirely on the "
    "accuracy of the input mask. For clinical use, a qualified radiologist and "
    "validated tooling are required."
)

#: Size bands as tumor area expressed as a percentage of the **brain** area
#: (not the whole image). These cut-points are an explainable, tunable
#: convention for description only — they are not clinical thresholds.
SIZE_BANDS = {
    "small": (0.0, 2.0),      # < 2% of brain area
    "medium": (2.0, 8.0),     # 2–8%
    "large": (8.0, 100.0),    # > 8%
}

#: Shape-classification thresholds (explainable rule-based; tunable).
#: solidity = area / convex-hull area  (1.0 = convex, lower = more concave)
#: circularity = 4*pi*area / perimeter^2  (1.0 = perfect circle)
SHAPE_RULES = {
    "regular": dict(min_solidity=0.90, min_circularity=0.65),
    "irregular": dict(min_solidity=0.75, min_circularity=0.40),
    # anything below the "irregular" floor is "highly_irregular"
}

#: Fraction of half-brain-width around the midline that counts as "central".
CENTRAL_BAND_FRACTION = 0.10

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of ``image`` (H, W)."""
    img = np.asarray(image)
    if img.ndim == 3:
        # If it looks like RGB/BGR, convert; if it's a 1-channel-in-3 stack, mean is fine.
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img[..., 0]
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Coerce any mask-like array to a {0,1} uint8 array."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected foreground component.

    Tumor masks frequently contain small specks; analyzing the largest blob
    avoids mixing several regions into one set of descriptors. If you expect
    multifocal tumors, run the analysis per-component instead (see
    :func:`iter_components`).
    """
    m = _binarize(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    # index 0 is background; pick the largest by area among the rest
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def iter_components(mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
    """Yield each connected foreground component as its own binary mask."""
    m = _binarize(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append((labels == i).astype(np.uint8))
    return out


def _largest_contour(bin_mask: np.ndarray):
    """Return the largest external contour of a binary mask, or None."""
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# --------------------------------------------------------------------------- #
# Brain mask (denominator for area %, and reference frame for location)
# --------------------------------------------------------------------------- #

def compute_brain_mask(image: np.ndarray) -> np.ndarray:
    """Approximate the intracranial (brain) region of an MRI slice.

    Used as the denominator for "area % of brain" (more meaningful than % of the
    whole image, which is dominated by black background) and as the reference
    frame for hemisphere/midline location.

    Method: Otsu threshold of the non-black foreground, morphological closing to
    fill holes, then the largest connected component. This is a standard,
    dependency-light approximation — it is NOT skull-stripping and will include
    skull on many sequences. It is good enough as an area reference; document it
    as an approximation in any writeup.
    """
    gray = _to_uint8_gray(image)
    # Slight blur stabilizes Otsu against speckle.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    brain = largest_component(closed)
    # Fill internal holes so ventricles don't subtract from the brain area.
    contour = _largest_contour(brain)
    if contour is not None:
        filled = np.zeros_like(brain)
        cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
        brain = filled
    return brain


# --------------------------------------------------------------------------- #
# Heuristic tumor mask — DEMO FALLBACK ONLY (clearly not a real segmenter)
# --------------------------------------------------------------------------- #

def heuristic_tumor_mask(image: np.ndarray,
                         brain_mask: Optional[np.ndarray] = None,
                         min_area_frac: float = 0.002) -> Tuple[np.ndarray, str]:
    """ROUGH tumor-region guess for demos when no segmenter is available.

    WARNING — this is intentionally simple and is **not** a validated tumor
    segmenter. It picks the brightest in-brain blob via Otsu within the brain
    mask. On many MRI sequences the brightest region is skull, fat, or a
    ventricle, not the tumor, so the resulting measurements may describe the
    wrong structure. Returned alongside a warning string so callers can surface
    the caveat. Replace with a trained segmentation model for real use.

    Returns (tumor_mask, warning_message).
    """
    gray = _to_uint8_gray(image)
    if brain_mask is None:
        brain_mask = compute_brain_mask(gray)

    inside = cv2.bitwise_and(gray, gray, mask=brain_mask)
    # Threshold relative to in-brain intensities (upper region = "bright").
    vals = inside[brain_mask > 0]
    if vals.size == 0:
        return np.zeros_like(gray), "empty brain mask; no tumor mask produced"
    hi = int(np.percentile(vals, 85))
    _, th = cv2.threshold(inside, hi, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    tumor = largest_component(th)

    brain_area = int(brain_mask.sum())
    if tumor.sum() < min_area_frac * max(brain_area, 1):
        return np.zeros_like(gray), "no region passed the heuristic size floor"

    warn = ("HEURISTIC MASK (Otsu brightest-blob) — not a validated tumor "
            "segmenter; measurements may describe a non-tumor structure.")
    return tumor, warn


# --------------------------------------------------------------------------- #
# Feature extractors (pure functions over a binary tumor mask)
# --------------------------------------------------------------------------- #

def geometric_features(tumor_mask: np.ndarray, brain_area_px: int) -> Dict[str, float]:
    """Compute geometric/morphometric descriptors of the tumor mask.

    Formulas (documented to avoid the usual circularity/compactness ambiguity):
      area              = foreground pixel count
      perimeter         = arc length of the external contour
      circularity       = 4*pi*area / perimeter^2          (1.0 = circle, 0..1)
      compactness       = perimeter^2 / area               (dimensionless; smaller = more compact)
      solidity          = area / convex_hull_area          (1.0 = convex)
      equivalent_diam   = sqrt(4*area/pi)                  (diameter of equal-area circle)
      aspect_ratio      = bbox_width / bbox_height         (axis-aligned bbox)
    """
    m = _binarize(tumor_mask)
    area = float(int(m.sum()))
    if area == 0:
        return {}

    contour = _largest_contour(m)
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0

    x, y, w, h = cv2.boundingRect(contour) if contour is not None else (0, 0, 0, 0)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0

    hull = cv2.convexHull(contour) if contour is not None else None
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0

    circularity = (4.0 * math.pi * area) / (perimeter ** 2 + _EPS)
    circularity = float(min(circularity, 1.0))  # discretization can nudge >1
    compactness = (perimeter ** 2) / (area + _EPS)
    equivalent_diameter = math.sqrt(4.0 * area / math.pi)

    area_pct_brain = 100.0 * area / (brain_area_px + _EPS) if brain_area_px else 0.0

    return {
        "tumor_area_px": round(area, 2),
        "tumor_area_pct_brain": round(area_pct_brain, 3),
        "perimeter_px": round(perimeter, 2),
        "bbox_x": int(x), "bbox_y": int(y),
        "bbox_width": int(w), "bbox_height": int(h),
        "aspect_ratio": round(aspect_ratio, 3),
        "circularity": round(circularity, 3),
        "compactness": round(compactness, 3),
        "solidity": round(solidity, 3),
        "convex_hull_area_px": round(hull_area, 2),
        "equivalent_diameter_px": round(equivalent_diameter, 2),
    }


def location_analysis(tumor_mask: np.ndarray,
                      brain_mask: np.ndarray) -> Dict[str, object]:
    """Locate the tumor centroid and assign a hemisphere relative to the brain
    midline.

    The midline is approximated by the brain-mask centroid x (more robust than
    the image center, since the brain is often off-center). A central band of
    +/- ``CENTRAL_BAND_FRACTION`` of the half-brain-width counts as "central".

    IMPORTANT — radiological convention: in standard radiological display,
    image-left is the patient's RIGHT and vice-versa. This function reports the
    *image* side and flags the convention; do not translate to patient side
    without confirming the display orientation.
    """
    m = _binarize(tumor_mask)
    if m.sum() == 0:
        return {"hemisphere_image_side": None, "center_xy": None,
                "midline_x": None, "note": "empty mask"}

    M = cv2.moments(m, binaryImage=True)
    cx = M["m10"] / (M["m00"] + _EPS)
    cy = M["m01"] / (M["m00"] + _EPS)

    bx, by, bw, bh = cv2.boundingRect(_binarize(brain_mask))
    bM = cv2.moments(_binarize(brain_mask), binaryImage=True)
    midline_x = bM["m10"] / (bM["m00"] + _EPS) if bM["m00"] > 0 else (bx + bw / 2.0)

    half_w = bw / 2.0
    band = CENTRAL_BAND_FRACTION * half_w
    if abs(cx - midline_x) <= band:
        side = "central"
    elif cx < midline_x:
        side = "image-left (radiological: patient-right)"
    else:
        side = "image-right (radiological: patient-left)"

    return {
        "hemisphere_image_side": side,
        "center_xy": [round(float(cx), 1), round(float(cy), 1)],
        "midline_x": round(float(midline_x), 1),
        "radiological_convention_note": (
            "Image-left = patient-right by radiological convention; verify "
            "display orientation before mapping to patient anatomy."
        ),
    }


def shape_analysis(geom: Dict[str, float]) -> Dict[str, str]:
    """Classify boundary regularity from solidity + circularity (rule-based).

    Returns the category plus a human-readable reason, so the decision is fully
    explainable (and easy for a downstream LLM to narrate).
    """
    if not geom:
        return {"shape_type": None, "reason": "no tumor region"}

    sol = geom.get("solidity", 0.0)
    circ = geom.get("circularity", 0.0)
    reg, irr = SHAPE_RULES["regular"], SHAPE_RULES["irregular"]

    if sol >= reg["min_solidity"] and circ >= reg["min_circularity"]:
        shape = "regular"
    elif sol >= irr["min_solidity"] and circ >= irr["min_circularity"]:
        shape = "irregular"
    else:
        shape = "highly_irregular"

    reason = (
        f"solidity={sol:.2f} (1.0=convex), circularity={circ:.2f} (1.0=circle); "
        f"thresholds — regular>= (sol {reg['min_solidity']}, circ "
        f"{reg['min_circularity']}), irregular>= (sol {irr['min_solidity']}, "
        f"circ {irr['min_circularity']}); lower => more lobulated/spiculated "
        f"boundary in the mask."
    )
    return {"shape_type": shape, "reason": reason}


def intensity_analysis(image: np.ndarray, tumor_mask: np.ndarray) -> Dict[str, float]:
    """Intensity statistics of the original grayscale image inside the mask."""
    gray = _to_uint8_gray(image)
    m = _binarize(tumor_mask)
    pix = gray[m > 0]
    if pix.size == 0:
        return {}
    return {
        "mean_intensity": round(float(pix.mean()), 2),
        "std_intensity": round(float(pix.std()), 2),
        "min_intensity": int(pix.min()),
        "max_intensity": int(pix.max()),
        "n_pixels": int(pix.size),
        "intensity_scale": "0-255 (normalized grayscale of the input slice)",
    }


def morphometric_features(tumor_mask: np.ndarray,
                          px_spacing_mm: Optional[float] = None) -> Dict[str, float]:
    """Axis lengths, orientation, elongation and an ellipsoidal VOLUME ESTIMATE.

    Fits an ellipse to the largest contour (``cv2.fitEllipse``) to obtain the
    in-plane **major** and **minor** axis lengths and the orientation angle.

    Volume estimate uses the standard ellipsoid formula

        V = (4/3) * pi * a * b * c

    where ``a`` and ``b`` are the in-plane semi-axes (major/2, minor/2). The
    out-of-plane semi-axis ``c`` is UNOBSERVABLE from a single 2D slice, so we
    assume a prolate-spheroid model with ``c = b`` (the minor in-plane semi-axis).
    **This is an order-of-magnitude estimate, NOT a true 3D tumor volume** - a
    real volume needs the full 3D MRI stack. The caveat travels with the numbers.

    If ``px_spacing_mm`` (mm per pixel) is provided, physical mm / mm^3 values are
    added; otherwise everything is in pixels / pixel^3.
    """
    m = _binarize(tumor_mask)
    contour = _largest_contour(m)
    if contour is None or len(contour) < 5:   # fitEllipse needs >=5 points
        return {}

    (_cx, _cy), (axis1, axis2), angle = cv2.fitEllipse(contour)
    major = float(max(axis1, axis2))
    minor = float(min(axis1, axis2))
    a, b = major / 2.0, minor / 2.0
    c = b                                    # spheroid assumption (see docstring)
    volume_px3 = (4.0 / 3.0) * math.pi * a * b * c
    elongation = round(minor / major, 3) if major > 0 else 0.0

    out = {
        "major_axis_px": round(major, 2),
        "minor_axis_px": round(minor, 2),
        "semi_major_a_px": round(a, 2),
        "semi_minor_b_px": round(b, 2),
        "assumed_semi_c_px": round(c, 2),
        "orientation_deg": round(float(angle), 1),
        "elongation": elongation,                       # 1.0 = circular, ->0 = elongated
        "ellipsoidal_volume_px3_est": round(volume_px3, 1),
        "volume_method": "ellipsoid V=4/3*pi*a*b*c, single-slice (c assumed = minor semi-axis)",
        "volume_caveat": ("Estimate from ONE 2D slice with a spheroid assumption; "
                          "NOT a validated 3D tumor volume. Use the full MRI stack "
                          "for a real volume."),
    }
    if px_spacing_mm:
        s = float(px_spacing_mm)
        out.update({
            "major_axis_mm": round(major * s, 2),
            "minor_axis_mm": round(minor * s, 2),
            "ellipsoidal_volume_mm3_est": round(volume_px3 * (s ** 3), 1),
            "px_spacing_mm": s,
        })
    return out


def quadrant_localization(tumor_mask: np.ndarray,
                          brain_mask: np.ndarray) -> Dict[str, object]:
    """Map the tumor centroid to a brain quadrant (upper/lower x left/right).

    The reference frame is the brain's bounding box: its horizontal mid-line and
    vertical mid-line split the brain into four quadrants. Reports the *image*
    side and flags the radiological convention (image-left = patient-right).
    """
    m = _binarize(tumor_mask)
    if m.sum() == 0:
        return {"quadrant": None, "note": "empty mask"}
    M = cv2.moments(m, binaryImage=True)
    cx = M["m10"] / (M["m00"] + _EPS)
    cy = M["m01"] / (M["m00"] + _EPS)
    bx, by, bw, bh = cv2.boundingRect(_binarize(brain_mask))
    midx, midy = bx + bw / 2.0, by + bh / 2.0
    vert = "superior" if cy < midy else "inferior"
    horiz = "image-left" if cx < midx else "image-right"
    return {
        "quadrant": f"{vert} / {horiz}",
        "centroid_xy": [round(float(cx), 1), round(float(cy), 1)],
        "brain_center_xy": [round(float(midx), 1), round(float(midy), 1)],
        "radiological_convention_note": (
            "image-left = patient-right by radiological convention; confirm display "
            "orientation before mapping to patient anatomy."),
    }


def size_category(area_pct_brain: float) -> Dict[str, str]:
    """Map area-%-of-brain to a small/medium/large *descriptor* (NOT clinical risk)."""
    for band, (lo, hi) in SIZE_BANDS.items():
        if lo <= area_pct_brain < hi:
            return {
                "size_band": band,
                "basis": f"tumor occupies {area_pct_brain:.2f}% of brain area; "
                         f"band '{band}' = [{lo}, {hi})% (descriptive, not clinical)",
            }
    return {"size_band": "large",
            "basis": f"{area_pct_brain:.2f}% of brain area (>= {SIZE_BANDS['large'][0]}%)"}


# --------------------------------------------------------------------------- #
# Optional classifier hook (TensorFlow / Keras) — fully optional
# --------------------------------------------------------------------------- #

def classify_image(model, image: np.ndarray, class_names: Sequence[str],
                   img_size: int = 224,
                   preprocess=None) -> Tuple[str, float, List[float]]:
    """Run a trained Keras classifier on one image -> (class, confidence, probs).

    ``preprocess`` is an optional callable applied to the resized float image
    (e.g. ``tensorflow.keras.applications.efficientnet.preprocess_input``).
    Kept optional so this module never hard-depends on TensorFlow.
    """
    gray = _to_uint8_gray(image)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (img_size, img_size)).astype("float32")
    x = preprocess(resized) if preprocess is not None else resized
    probs = model.predict(x[None, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), [float(p) for p in probs]


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

@dataclass
class TumorReport:
    """Container that also serializes to the requested JSON schema."""
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    tumor_present: bool = False
    geometry: Dict[str, float] = field(default_factory=dict)
    location: Dict[str, object] = field(default_factory=dict)
    shape: Dict[str, str] = field(default_factory=dict)
    intensity: Dict[str, float] = field(default_factory=dict)
    size: Dict[str, str] = field(default_factory=dict)
    morphometry: Dict[str, float] = field(default_factory=dict)
    quadrant: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> Dict[str, object]:
        """Full, nested report."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "tumor_present": self.tumor_present,
            "tumor_area_px": self.geometry.get("tumor_area_px"),
            "tumor_area_pct_brain": self.geometry.get("tumor_area_pct_brain"),
            "tumor_location": self.location.get("hemisphere_image_side"),
            "shape_type": self.shape.get("shape_type"),
            "size_band": self.size.get("size_band"),
            "tumor_major_axis_px": self.morphometry.get("major_axis_px"),
            "tumor_minor_axis_px": self.morphometry.get("minor_axis_px"),
            "tumor_volume_px3_est": self.morphometry.get("ellipsoidal_volume_px3_est"),
            "tumor_quadrant": self.quadrant.get("quadrant"),
            "geometry": self.geometry,
            "location": self.location,
            "shape": self.shape,
            "intensity_statistics": self.intensity,
            "size": self.size,
            "morphometry": self.morphometry,
            "quadrant": self.quadrant,
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
        }

    def to_schema(self) -> Dict[str, object]:
        """Exactly the JSON keys requested in the brief (+ disclaimer)."""
        return {
            "predicted_class": self.predicted_class or "",
            "confidence": "" if self.confidence is None else round(self.confidence, 4),
            "tumor_area": self.geometry.get("tumor_area_px", ""),
            "tumor_percentage": self.geometry.get("tumor_area_pct_brain", ""),
            "tumor_location": self.location.get("hemisphere_image_side", "") or "",
            "shape_type": self.shape.get("shape_type", "") or "",
            "risk_category": self.size.get("size_band", "") or "",
            "intensity_statistics": self.intensity,
            "disclaimer": self.disclaimer,
        }


def analyze_tumor(
    image: np.ndarray,
    tumor_mask: Optional[np.ndarray] = None,
    *,
    predicted_class: Optional[str] = None,
    confidence: Optional[float] = None,
    brain_mask: Optional[np.ndarray] = None,
    no_tumor_labels: Sequence[str] = ("notumor", "no_tumor", "none"),
    allow_heuristic_mask: bool = False,
) -> TumorReport:
    """Run the full analysis pipeline over a slice + tumor mask.

    Parameters
    ----------
    image : np.ndarray
        The MRI slice (grayscale or RGB).
    tumor_mask : np.ndarray | None
        Binary tumor mask. If None and ``allow_heuristic_mask`` is True, a ROUGH
        Otsu fallback is used (with a warning attached). Strongly prefer passing
        a mask from a trained segmenter.
    predicted_class, confidence : optional
        Classifier outputs, if available (e.g. from :func:`classify_image`).
    brain_mask : np.ndarray | None
        Optional precomputed brain mask; otherwise estimated.
    no_tumor_labels : sequence
        If ``predicted_class`` is one of these, analysis is skipped and an empty
        (tumor_present=False) report is returned — no fabricated features.
    """
    report = TumorReport(predicted_class=predicted_class, confidence=confidence)

    if predicted_class is not None and predicted_class.lower() in {s.lower() for s in no_tumor_labels}:
        report.warnings.append("predicted_class indicates no tumor; analysis skipped.")
        return report

    if brain_mask is None:
        brain_mask = compute_brain_mask(image)
    brain_area = int(brain_mask.sum())

    if tumor_mask is None:
        if not allow_heuristic_mask:
            report.warnings.append(
                "No tumor_mask provided and heuristic fallback disabled; nothing "
                "to analyze. Pass a mask from a trained segmenter.")
            return report
        tumor_mask, warn = heuristic_tumor_mask(image, brain_mask)
        report.warnings.append(warn)

    tumor_mask = largest_component(tumor_mask)
    if int(tumor_mask.sum()) == 0:
        report.warnings.append("Tumor mask is empty after cleanup; no region to analyze.")
        return report

    report.tumor_present = True
    report.geometry = geometric_features(tumor_mask, brain_area)
    report.location = location_analysis(tumor_mask, brain_mask)
    report.shape = shape_analysis(report.geometry)
    report.intensity = intensity_analysis(image, tumor_mask)
    report.size = size_category(report.geometry.get("tumor_area_pct_brain", 0.0))
    report.morphometry = morphometric_features(tumor_mask)
    report.quadrant = quadrant_localization(tumor_mask, brain_mask)
    # stash masks for the visualizer (not serialized)
    report.__dict__["_tumor_mask"] = tumor_mask
    report.__dict__["_brain_mask"] = brain_mask
    return report


# --------------------------------------------------------------------------- #
# Visualization (6 panels + dashboard)
# --------------------------------------------------------------------------- #

def build_dashboard(image: np.ndarray, report: TumorReport,
                    save_path: Optional[str] = None,
                    show: bool = False):
    """Render the 6 required views + a metrics panel into one figure.

    Panels: (1) original, (2) mask, (3) boundary overlay, (4) bbox overlay,
    (5) center marker, (6) text dashboard of the computed features.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = _to_uint8_gray(image)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    mask = report.__dict__.get("_tumor_mask")
    if mask is None:
        mask = np.zeros_like(gray)

    contour = _largest_contour(_binarize(mask))

    boundary = rgb.copy()
    if contour is not None:
        cv2.drawContours(boundary, [contour], -1, (0, 255, 0), 2)

    bbox = rgb.copy()
    g = report.geometry
    if g:
        x, y, w, h = g["bbox_x"], g["bbox_y"], g["bbox_width"], g["bbox_height"]
        cv2.rectangle(bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)

    center = boundary.copy()
    loc = report.location
    if loc.get("center_xy"):
        cx, cy = map(int, loc["center_xy"])
        cv2.drawMarker(center, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
        mid = int(loc.get("midline_x") or cx)
        cv2.line(center, (mid, 0), (mid, center.shape[0]), (255, 0, 255), 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        ("Original MRI", rgb),
        ("Segmentation Mask", np.dstack([mask * 255] * 3)),
        ("Tumor Boundary", boundary),
        ("Bounding Box", bbox),
        ("Center + Midline", center),
    ]
    for ax, (title, img) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Panel 6: text dashboard
    ax = axes.flat[5]
    ax.axis("off")
    lines = _dashboard_text(report)
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10,
            family="monospace", transform=ax.transAxes)
    ax.set_title("Analysis Dashboard", fontsize=12, fontweight="bold")

    fig.suptitle("Tumor Image-Analysis (NOT a medical diagnosis)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=7, wrap=True, color="0.3")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def _dashboard_text(report: TumorReport) -> List[str]:
    g, loc, sh, inten, sz = (report.geometry, report.location, report.shape,
                             report.intensity, report.size)
    if not report.tumor_present:
        return ["No tumor region analyzed.",
                *[f"! {w}" for w in report.warnings]]
    L = [
        f"class      : {report.predicted_class}  "
        f"({'' if report.confidence is None else f'{report.confidence*100:.1f}%'})",
        f"size band  : {sz.get('size_band')}",
        f"location   : {loc.get('hemisphere_image_side')}",
        f"shape      : {sh.get('shape_type')}",
        "-" * 34,
        f"area       : {g.get('tumor_area_px')} px "
        f"({g.get('tumor_area_pct_brain')}% of brain)",
        f"perimeter  : {g.get('perimeter_px')} px",
        f"equiv diam : {g.get('equivalent_diameter_px')} px",
        f"bbox WxH   : {g.get('bbox_width')} x {g.get('bbox_height')}",
        f"aspect     : {g.get('aspect_ratio')}",
        f"circularity: {g.get('circularity')}",
        f"solidity   : {g.get('solidity')}",
        f"compactness: {g.get('compactness')}",
        "-" * 34,
        f"intensity  : mean {inten.get('mean_intensity')} "
        f"std {inten.get('std_intensity')}",
        f"           : min {inten.get('min_intensity')} "
        f"max {inten.get('max_intensity')}",
    ]
    if report.warnings:
        L += ["-" * 34] + [f"! {w}" for w in report.warnings]
    return L


# --------------------------------------------------------------------------- #
# LLM-ready payload
# --------------------------------------------------------------------------- #

def to_llm_payload(report: TumorReport) -> Dict[str, object]:
    """Compact, text-friendly payload for a downstream LLM reporting system.

    Keeps the disclaimer attached and instructs the LLM to remain
    image-analysis-only. Contains no images — just the numbers and labels.
    """
    return {
        "task": "Generate an image-analysis description (NOT a diagnosis) from "
                "the structured tumor measurements below.",
        "guardrails": [
            "Do not state or imply a diagnosis, grade, or prognosis.",
            "Refer to findings as measurements of a segmentation mask.",
            "Preserve the disclaimer in any generated text.",
            "If 'warnings' mention a heuristic mask, state that measurements are "
            "approximate and unvalidated.",
        ],
        "findings": report.to_schema(),
        "details": {
            "geometry": report.geometry,
            "location": report.location,
            "shape": report.shape,
            "intensity": report.intensity,
            "size": report.size,
            "morphometry": report.morphometry,
            "quadrant": report.quadrant,
        },
        "warnings": report.warnings,
        "disclaimer": DISCLAIMER,
    }


def save_report_json(report: TumorReport, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Demo / smoke test on a synthetic phantom (no model or data required)
# --------------------------------------------------------------------------- #

def _demo() -> None:
    """Build a synthetic 'brain + tumor' phantom and run the full pipeline.

    Lets you verify the module end-to-end without TensorFlow, models, or MRI
    data. The geometry is known, so it doubles as a sanity check.
    """
    H = W = 256
    img = np.zeros((H, W), np.uint8)
    cv2.circle(img, (W // 2, H // 2), 90, 70, -1)        # "brain" disc
    cv2.circle(img, (W // 2 + 35, H // 2 - 20), 22, 210, -1)  # bright "tumor"
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Use the bright blob as a clean ground-truth mask (bypasses the heuristic).
    mask = (img > 150).astype(np.uint8)

    report = analyze_tumor(img, mask, predicted_class="glioma", confidence=0.83)
    print(json.dumps(report.to_schema(), indent=2))
    out = Path("improvements/artifacts/reports/tumor_analysis_demo.png")
    build_dashboard(img, report, save_path=str(out))
    print(f"[demo] dashboard -> {out}")
    print(f"[demo] LLM payload keys: {list(to_llm_payload(report).keys())}")


if __name__ == "__main__":
    _demo()
