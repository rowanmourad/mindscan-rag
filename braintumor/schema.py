"""reporting/schema.py - Structured output schema for the brain-tumor pipeline.

The pipeline (improvements.predict.Predictor) emits a plain dict; this module
provides a typed schema for that dict so it can be:
    1. Validated programmatically  (PredictionResult.from_dict / from_predictor_output)
    2. Serialized cleanly to JSON   (.to_dict / .to_json)
    3. Consumed by an LLM with a clear contract  (JSON_SCHEMA)
    4. Rendered into a case report  (reporting.generate_report)

Target output shape (the user-requested fields, plus required structure for
each):

    {
        "prediction": "glioma",
        "confidence": 0.9342,
        "tumor_type": "glioma",                  # null if notumor
        "tumor_location": { ... },               # null if notumor
        "tumor_size":     { ... },               # null if notumor
        "xai_summary":    { ... },               # null if notumor or unavailable
        "segmentation_summary": { ... | null },  # reserved for future BraTS model
        "per_class_probabilities": { ... },
        "model_info": { ... },
        "timestamp": "2026-06-16T05:59:11+00:00",
        "image_path": "...",
        "is_tumor": true
    }

Implementation: plain Python dataclasses (no pydantic dependency). Use
`PredictionResult.from_predictor_output(predictor.predict(img)["json"])` to
convert the predictor's dict into a typed object.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Allowed prediction labels (extend if your model has different classes)
ALLOWED_PREDICTIONS = ("glioma", "meningioma", "notumor", "pituitary")
ALLOWED_TUMOR_TYPES = ("glioma", "meningioma", "pituitary")
ALLOWED_SIZE_CATEGORIES = ("very_small", "small", "medium", "large")
ALLOWED_XAI_METHODS = ("grad-cam", "grad-cam++", "gradcam", "gradcam++")


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------
@dataclass
class TumorLocation:
    """Bounding box + centroid of the model-attended region, in resized-image
    pixel coordinates (i.e. the resolution the model saw).

    Coordinates use image-standard convention: origin top-left, x right, y down.
    """
    bbox_x: int                              # top-left x of bounding box
    bbox_y: int                              # top-left y of bounding box
    bbox_w: int                              # width of bounding box
    bbox_h: int                              # height of bounding box
    center_x: int                            # centroid x
    center_y: int                            # centroid y
    image_coverage_pct: float                # area / image area * 100
    brain_coverage_pct: float                # area / brain area * 100 (clinically useful)
    brain_area_px: int                       # for reproducibility
    region_area_px: int                      # attention-region area in pixels

    def __post_init__(self):
        for f in ("bbox_x", "bbox_y", "bbox_w", "bbox_h",
                  "center_x", "center_y", "brain_area_px", "region_area_px"):
            v = getattr(self, f)
            if v is None or int(v) < 0:
                raise ValueError(f"TumorLocation.{f} must be a non-negative int (got {v}).")
            setattr(self, f, int(v))
        for f in ("image_coverage_pct", "brain_coverage_pct"):
            v = getattr(self, f)
            if v is None or not 0 <= float(v) <= 100:
                raise ValueError(f"TumorLocation.{f} must be in [0,100] (got {v}).")
            setattr(self, f, round(float(v), 3))


@dataclass
class TumorSize:
    """Coarse size descriptor derived from attention-region area."""
    area_px: int
    brain_coverage_pct: float
    category: str                            # very_small | small | medium | large

    def __post_init__(self):
        self.area_px = int(self.area_px)
        self.brain_coverage_pct = round(float(self.brain_coverage_pct), 3)
        if self.category not in ALLOWED_SIZE_CATEGORIES:
            raise ValueError(
                f"TumorSize.category must be one of {ALLOWED_SIZE_CATEGORIES} "
                f"(got {self.category!r})."
            )


@dataclass
class XAISummary:
    """Numerical summary of the XAI explanation used for this prediction."""
    method: str                              # "grad-cam" | "grad-cam++"
    focus_spread_pct: float                  # lower = tighter focus
    peak_activation: float                   # max heatmap value (after normalize)
    mean_activation: float                   # mean heatmap value
    center_of_mass_x: float
    center_of_mass_y: float
    target_layer: Optional[str] = None       # e.g. "block7a_project_conv"
    better_method: Optional[str] = None      # if Grad-CAM vs Grad-CAM++ were compared

    def __post_init__(self):
        m = self.method.lower()
        if m not in ALLOWED_XAI_METHODS:
            raise ValueError(
                f"XAISummary.method must be one of {ALLOWED_XAI_METHODS} "
                f"(got {self.method!r})."
            )
        # Canonicalize
        self.method = "grad-cam++" if "++" in m else "grad-cam"
        if not 0 <= self.focus_spread_pct <= 100:
            raise ValueError("focus_spread_pct must be in [0,100].")
        self.focus_spread_pct = round(float(self.focus_spread_pct), 3)
        self.peak_activation = round(float(self.peak_activation), 4)
        self.mean_activation = round(float(self.mean_activation), 4)
        self.center_of_mass_x = round(float(self.center_of_mass_x), 2)
        self.center_of_mass_y = round(float(self.center_of_mass_y), 2)


@dataclass
class SegmentationSummary:
    """Reserved for true segmentation output (BraTS-style model). The current
    pipeline does NOT produce a clinically validated mask; this slot is filled
    only when a dedicated segmentation model is added downstream."""
    method: str                              # e.g. "unet-brats", "manual"
    mask_area_px: int
    mask_brain_coverage_pct: float
    dice_against_attention: Optional[float] = None
    mask_path: Optional[str] = None

    def __post_init__(self):
        self.mask_area_px = int(self.mask_area_px)
        self.mask_brain_coverage_pct = round(float(self.mask_brain_coverage_pct), 3)
        if self.dice_against_attention is not None:
            self.dice_against_attention = round(float(self.dice_against_attention), 4)


@dataclass
class ModelInfo:
    """Provenance for the prediction. Helps the LLM (and clinicians) reason
    about which model produced which result."""
    model_path: str
    img_size: int
    class_names: List[str]
    rescale_to_unit: bool = False
    architecture: Optional[str] = None       # e.g. "EfficientNetB2", "Two-stage(B+S)"
    version: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        self.img_size = int(self.img_size)
        self.class_names = [str(c) for c in self.class_names]
        self.rescale_to_unit = bool(self.rescale_to_unit)


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """The structured prediction record. This is the single source of truth
    for everything downstream: report generation, GUI, LLM consumers, FastAPI.
    """
    prediction: str
    confidence: float
    is_tumor: bool
    tumor_type: Optional[str] = None
    per_class_probabilities: Dict[str, float] = field(default_factory=dict)
    tumor_location: Optional[TumorLocation] = None
    tumor_size: Optional[TumorSize] = None
    xai_summary: Optional[XAISummary] = None
    segmentation_summary: Optional[SegmentationSummary] = None
    model_info: Optional[ModelInfo] = None
    timestamp: str = ""
    image_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.prediction not in ALLOWED_PREDICTIONS:
            raise ValueError(
                f"prediction must be one of {ALLOWED_PREDICTIONS} "
                f"(got {self.prediction!r})."
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0,1] (got {self.confidence}).")
        self.confidence = round(float(self.confidence), 4)
        self.is_tumor = bool(self.is_tumor)
        if self.tumor_type is not None and self.tumor_type not in ALLOWED_TUMOR_TYPES:
            raise ValueError(
                f"tumor_type must be one of {ALLOWED_TUMOR_TYPES} or None "
                f"(got {self.tumor_type!r})."
            )
        # Consistency: notumor -> no tumor_type, no location, no size
        if self.prediction == "notumor":
            self.is_tumor = False
            self.tumor_type = None
            if self.tumor_location is not None or self.tumor_size is not None:
                self.warnings.append(
                    "Prediction is 'notumor' but tumor_location/size were "
                    "supplied; ignoring them."
                )
                self.tumor_location = None
                self.tumor_size = None
        else:
            self.is_tumor = True
            if self.tumor_type is None:
                self.tumor_type = self.prediction
        # Probability dict: round + validate
        if self.per_class_probabilities:
            self.per_class_probabilities = {
                str(k): round(float(v), 4)
                for k, v in self.per_class_probabilities.items()
            }
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Produce a clean nested dict for JSON / LLM consumption."""
        d = {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "is_tumor": self.is_tumor,
            "tumor_type": self.tumor_type,
            "tumor_location": asdict(self.tumor_location) if self.tumor_location else None,
            "tumor_size": asdict(self.tumor_size) if self.tumor_size else None,
            "xai_summary": asdict(self.xai_summary) if self.xai_summary else None,
            "segmentation_summary": (
                asdict(self.segmentation_summary) if self.segmentation_summary else None
            ),
            "per_class_probabilities": dict(self.per_class_probabilities),
            "model_info": asdict(self.model_info) if self.model_info else None,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
        }
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_minimal_dict(self) -> Dict[str, Any]:
        """The 7-field minimal output requested in the spec.

        Returns:
            {
              "prediction": "...",
              "confidence": "...",
              "tumor_type": "...",
              "tumor_location": "...",
              "tumor_size": "...",
              "xai_summary": "...",
              "segmentation_summary": "..."
            }
        Each value is a human-readable string; fields not applicable are "".
        """
        if self.tumor_location is not None:
            loc = (
                f"bbox=({self.tumor_location.bbox_x}, {self.tumor_location.bbox_y}, "
                f"{self.tumor_location.bbox_w}x{self.tumor_location.bbox_h}); "
                f"center=({self.tumor_location.center_x}, {self.tumor_location.center_y}); "
                f"brain coverage {self.tumor_location.brain_coverage_pct:.1f}%"
            )
        else:
            loc = ""

        if self.tumor_size is not None:
            sz = (f"{self.tumor_size.category} "
                  f"({self.tumor_size.brain_coverage_pct:.1f}% of brain, "
                  f"{self.tumor_size.area_px} px)")
        else:
            sz = ""

        if self.xai_summary is not None:
            xai = (
                f"{self.xai_summary.method}; focus spread "
                f"{self.xai_summary.focus_spread_pct:.1f}%; "
                f"peak {self.xai_summary.peak_activation:.3f}; "
                f"center of mass ({self.xai_summary.center_of_mass_x:.0f}, "
                f"{self.xai_summary.center_of_mass_y:.0f})"
            )
            if self.xai_summary.better_method:
                xai += f"; comparator winner: {self.xai_summary.better_method}"
        else:
            xai = ""

        if self.segmentation_summary is not None:
            seg = (f"{self.segmentation_summary.method}; "
                   f"mask area {self.segmentation_summary.mask_area_px} px "
                   f"({self.segmentation_summary.mask_brain_coverage_pct:.1f}% of brain)")
            if self.segmentation_summary.dice_against_attention is not None:
                seg += f"; Dice vs attention {self.segmentation_summary.dice_against_attention:.3f}"
        else:
            seg = ""

        return {
            "prediction": self.prediction,
            "confidence": f"{self.confidence:.4f}",
            "tumor_type": self.tumor_type or "",
            "tumor_location": loc,
            "tumor_size": sz,
            "xai_summary": xai,
            "segmentation_summary": seg,
        }

    # ----------------------------------------------------------------
    # Deserialization
    # ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionResult":
        """Re-hydrate a PredictionResult from a serialized dict."""
        def _maybe(sub_cls, payload):
            if payload is None:
                return None
            if is_dataclass(payload):
                return payload
            # Filter unknown keys defensively
            allowed = {f.name for f in fields(sub_cls)}
            clean = {k: v for k, v in payload.items() if k in allowed}
            return sub_cls(**clean)

        return cls(
            prediction=d["prediction"],
            confidence=float(d["confidence"]),
            is_tumor=bool(d.get("is_tumor", d["prediction"] != "notumor")),
            tumor_type=d.get("tumor_type"),
            per_class_probabilities=dict(d.get("per_class_probabilities") or {}),
            tumor_location=_maybe(TumorLocation, d.get("tumor_location")),
            tumor_size=_maybe(TumorSize, d.get("tumor_size")),
            xai_summary=_maybe(XAISummary, d.get("xai_summary")),
            segmentation_summary=_maybe(SegmentationSummary, d.get("segmentation_summary")),
            model_info=_maybe(ModelInfo, d.get("model_info")),
            timestamp=d.get("timestamp", ""),
            image_path=d.get("image_path"),
            warnings=list(d.get("warnings") or []),
        )

    @classmethod
    def from_predictor_output(cls, predictor_dict: Dict[str, Any]) -> "PredictionResult":
        """Convert the dict produced by improvements.predict.Predictor.predict()
        (the "json" key of its return value) into a typed PredictionResult.

        This is the integration seam between Phase 7 and Phase 8.
        """
        # The predictor uses nested dicts for attention_region; flatten into our
        # TumorLocation dataclass.
        ar = predictor_dict.get("attention_region")
        tloc = None
        if ar and ar.get("present"):
            bbox = ar.get("bbox") or {}
            center = ar.get("center") or {}
            tloc = TumorLocation(
                bbox_x=int(bbox.get("x", 0)),
                bbox_y=int(bbox.get("y", 0)),
                bbox_w=int(bbox.get("w", 0)),
                bbox_h=int(bbox.get("h", 0)),
                center_x=int(center.get("x", 0)),
                center_y=int(center.get("y", 0)),
                image_coverage_pct=float(ar.get("image_coverage_pct", 0.0)),
                brain_coverage_pct=float(ar.get("brain_coverage_pct", 0.0)),
                brain_area_px=int(ar.get("brain_area_px", 0)),
                region_area_px=int(ar.get("area_px", 0)),
            )

        sz = predictor_dict.get("size")
        tsize = TumorSize(
            area_px=int(sz["area_px"]),
            brain_coverage_pct=float(sz["brain_coverage_pct"]),
            category=str(sz["category"]),
        ) if sz else None

        xs = predictor_dict.get("xai_summary")
        xai = None
        if xs and "error" not in xs:
            # Predictor produces both a comparison summary and a single explainer.
            # Prefer the comparison if available; otherwise use whatever's present.
            if "gradcampp_focus" in xs:
                f = xs["gradcampp_focus"]
                com = f.get("center_of_mass", [0.0, 0.0])
                if isinstance(com, dict):
                    com = [com.get("x", 0.0), com.get("y", 0.0)]
                xai = XAISummary(
                    method="grad-cam++",
                    focus_spread_pct=float(f.get("spread_pct", 0.0)),
                    peak_activation=float(f.get("peak", 0.0)),
                    mean_activation=float(f.get("mean", 0.0)),
                    center_of_mass_x=float(com[1] if len(com) > 1 else 0.0),
                    center_of_mass_y=float(com[0] if len(com) > 0 else 0.0),
                    better_method=xs.get("better_method"),
                )
            elif "method" in xs:
                com = xs.get("center_of_mass", {"x": 0.0, "y": 0.0})
                xai = XAISummary(
                    method=xs.get("method", "grad-cam"),
                    focus_spread_pct=float(xs.get("focus_spread_pct", 0.0)),
                    peak_activation=float(xs.get("peak_activation", 0.0)),
                    mean_activation=float(xs.get("mean_activation", 0.0)),
                    center_of_mass_x=float(com.get("x", 0.0) if isinstance(com, dict) else 0.0),
                    center_of_mass_y=float(com.get("y", 0.0) if isinstance(com, dict) else 0.0),
                    target_layer=xs.get("target_layer"),
                )

        ss = predictor_dict.get("segmentation_summary")
        seg = None
        if ss:
            seg = SegmentationSummary(
                method=str(ss.get("method", "unknown")),
                mask_area_px=int(ss.get("mask_area_px", 0)),
                mask_brain_coverage_pct=float(ss.get("mask_brain_coverage_pct", 0.0)),
                dice_against_attention=ss.get("dice_against_attention"),
                mask_path=ss.get("mask_path"),
            )

        mi = predictor_dict.get("model_info")
        model_info = ModelInfo(
            model_path=str(mi.get("model_path", "")) if mi else "",
            img_size=int(mi.get("img_size", 0)) if mi else 0,
            class_names=list(mi.get("class_names", [])) if mi else [],
            rescale_to_unit=bool(mi.get("rescale_to_unit", False)) if mi else False,
        ) if mi else None

        return cls(
            prediction=predictor_dict["prediction"],
            confidence=float(predictor_dict["confidence"]),
            is_tumor=bool(predictor_dict.get("is_tumor", False)),
            tumor_type=predictor_dict.get("tumor_type"),
            per_class_probabilities=dict(predictor_dict.get("per_class_probabilities") or {}),
            tumor_location=tloc,
            tumor_size=tsize,
            xai_summary=xai,
            segmentation_summary=seg,
            model_info=model_info,
            timestamp=predictor_dict.get("timestamp", ""),
            image_path=predictor_dict.get("image_path"),
        )


# ---------------------------------------------------------------------------
# JSON Schema (for LLM-side validation / system prompts)
# ---------------------------------------------------------------------------
JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BrainTumorPredictionResult",
    "type": "object",
    "required": ["prediction", "confidence", "is_tumor"],
    "properties": {
        "prediction":           {"type": "string", "enum": list(ALLOWED_PREDICTIONS)},
        "confidence":           {"type": "number", "minimum": 0, "maximum": 1},
        "is_tumor":             {"type": "boolean"},
        "tumor_type": {
            "type": ["string", "null"],
            "enum": list(ALLOWED_TUMOR_TYPES) + [None],
        },
        "per_class_probabilities": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "tumor_location": {
            "type": ["object", "null"],
            "properties": {
                "bbox_x": {"type": "integer", "minimum": 0},
                "bbox_y": {"type": "integer", "minimum": 0},
                "bbox_w": {"type": "integer", "minimum": 0},
                "bbox_h": {"type": "integer", "minimum": 0},
                "center_x": {"type": "integer", "minimum": 0},
                "center_y": {"type": "integer", "minimum": 0},
                "region_area_px": {"type": "integer", "minimum": 0},
                "brain_area_px": {"type": "integer", "minimum": 0},
                "image_coverage_pct": {"type": "number", "minimum": 0, "maximum": 100},
                "brain_coverage_pct": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "center_x",
                         "center_y", "brain_coverage_pct", "region_area_px"],
        },
        "tumor_size": {
            "type": ["object", "null"],
            "properties": {
                "area_px": {"type": "integer", "minimum": 0},
                "brain_coverage_pct": {"type": "number", "minimum": 0, "maximum": 100},
                "category": {"type": "string", "enum": list(ALLOWED_SIZE_CATEGORIES)},
            },
            "required": ["area_px", "brain_coverage_pct", "category"],
        },
        "xai_summary": {
            "type": ["object", "null"],
            "properties": {
                "method": {"type": "string"},
                "focus_spread_pct": {"type": "number", "minimum": 0, "maximum": 100},
                "peak_activation": {"type": "number"},
                "mean_activation": {"type": "number"},
                "center_of_mass_x": {"type": "number"},
                "center_of_mass_y": {"type": "number"},
                "target_layer": {"type": ["string", "null"]},
                "better_method": {"type": ["string", "null"]},
            },
            "required": ["method", "focus_spread_pct"],
        },
        "segmentation_summary": {"type": ["object", "null"]},
        "model_info": {"type": ["object", "null"]},
        "timestamp": {"type": "string"},
        "image_path": {"type": ["string", "null"]},
    },
}


# ---------------------------------------------------------------------------
# Convenience: minimal-schema constant (matches the spec verbatim)
# ---------------------------------------------------------------------------
MINIMAL_OUTPUT_TEMPLATE: Dict[str, str] = {
    "prediction": "",
    "confidence": "",
    "tumor_type": "",
    "tumor_location": "",
    "tumor_size": "",
    "xai_summary": "",
    "segmentation_summary": "",
}
