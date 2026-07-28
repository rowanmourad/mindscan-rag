"""braintumor/seg_eval.py - Automated Segmentation Evaluator (framework-agnostic).

Pure logic for scoring a predicted mask against the clinician-verified
(brush-corrected) ground truth, isolating neck/skull-artifact performance, and
tracking how the system learns over time. No FastAPI dependency, so it is unit-
testable on its own; ``api/main.py`` is a thin HTTP wrapper over this module.

Metrics use numpy (exact, dependency-light). The formulas match MONAI's
``compute_dice`` / ``MeanIoU``; an optional MONAI cross-check is provided.

Registry: a structured JSON file at ``artifacts/evaluation/registry.json``
recording, per evaluation, the timestamp, patient id, model used, the
neck/skull flag, and the four metrics. This is the same ``artifacts/`` tree the
clinician-feedback cache uses, keeping all learning signals in one place.
"""
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from . import config

EVAL_DIR = config.OUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = EVAL_DIR / "registry.json"

# A segmentation is counted as "successful" at or above this Dice score. Neck/skull
# reporting uses this to express a success *rate* for the Compact-Blob algorithm.
SUCCESS_DICE = 0.70

MaskLike = Union[np.ndarray, list, tuple, str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Mask decoding (accept many transport formats)
# ---------------------------------------------------------------------------
def decode_mask(obj: MaskLike) -> np.ndarray:
    """Return a uint8 {0,1} 2D mask from: ndarray, 2D list, base64-PNG string, or
    a ``data:image/...;base64,`` data URI."""
    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, (list, tuple)):
        arr = np.asarray(obj)
    elif isinstance(obj, str):
        s = obj.split(",", 1)[-1] if obj.strip().startswith("data:") else obj
        raw = base64.b64decode(s)
        from PIL import Image
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
    else:
        raise ValueError(f"Unsupported mask type: {type(obj).__name__}")
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D; got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def _match_shapes(pred: np.ndarray, gt: np.ndarray):
    if pred.shape != gt.shape:
        import cv2
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    return pred, (gt > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics engine: Dice, IoU, Precision, Recall
# ---------------------------------------------------------------------------
def compute_segmentation_metrics(pred: MaskLike, gt: MaskLike) -> Dict[str, float]:
    """Dice (DSC), IoU (Jaccard), Precision, Recall for two binary masks.

    Convention: two empty masks agree perfectly (all metrics = 1.0), which is the
    correct behaviour for a true-negative (healthy) slice.
    """
    p = decode_mask(pred) if not isinstance(pred, np.ndarray) else (pred > 0).astype(np.uint8)
    g = decode_mask(gt) if not isinstance(gt, np.ndarray) else (gt > 0).astype(np.uint8)
    p, g = _match_shapes(p, g)

    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))

    if p.sum() == 0 and g.sum() == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0,
                "tp": 0, "fp": 0, "fn": 0, "empty_both": True}

    eps = 1e-8
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    return {"dice": round(dice, 4), "iou": round(iou, 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn, "empty_both": False}


def compute_metrics_monai(pred: MaskLike, gt: MaskLike) -> Dict[str, float]:
    """Optional MONAI-backed Dice/IoU (parity check). Requires monai + torch."""
    import torch
    from monai.metrics import compute_dice, compute_iou
    p = torch.as_tensor(decode_mask(pred))[None, None].float()
    g = torch.as_tensor(decode_mask(gt))[None, None].float()
    d = float(compute_dice(p, g, include_background=True).nan_to_num(1.0).item())
    j = float(compute_iou(p, g, include_background=True).nan_to_num(1.0).item())
    return {"dice": round(d, 4), "iou": round(j, 4)}


# ---------------------------------------------------------------------------
# Learning registry (atomic JSON append)
# ---------------------------------------------------------------------------
def load_registry() -> List[Dict]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("records", [])
        except Exception:
            return []
    return []


def _save_registry(records: List[Dict]) -> None:
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"records": records, "updated": _now()}, indent=2),
                   encoding="utf-8")
    os.replace(tmp, REGISTRY_PATH)              # atomic replace


def append_evaluation(metrics: Dict, metadata: Dict) -> Dict:
    """Append one evaluation record to the registry and return it."""
    metadata = metadata or {}
    record = {
        "timestamp": _now(),
        "patient_id": metadata.get("patient_id"),
        "model_used": metadata.get("model_used"),
        "has_neck_or_skull": bool(metadata.get("has_neck_or_skull", False)),
        "metrics": metrics,
        "success": bool(metrics.get("dice", 0.0) >= SUCCESS_DICE),
    }
    records = load_registry()
    records.append(record)
    _save_registry(records)
    return record


def evaluate_and_log(predicted_mask: MaskLike, ground_truth_mask: MaskLike,
                     slice_metadata: Optional[Dict] = None) -> Dict:
    """Decode -> score -> log. Single entry point used by the API route."""
    metrics = compute_segmentation_metrics(decode_mask(predicted_mask),
                                            decode_mask(ground_truth_mask))
    return append_evaluation(metrics, slice_metadata or {})


# ---------------------------------------------------------------------------
# Analytics: neck/skull vulnerability isolation
# ---------------------------------------------------------------------------
def _aggregate(subset: List[Dict]) -> Dict:
    if not subset:
        return {"n": 0, "mean_dice": None, "mean_iou": None, "success_rate": None}
    dice = np.array([r["metrics"]["dice"] for r in subset], dtype=float)
    iou = np.array([r["metrics"]["iou"] for r in subset], dtype=float)
    succ = np.array([r["success"] for r in subset], dtype=float)
    return {"n": len(subset),
            "mean_dice": round(float(dice.mean()), 4),
            "mean_iou": round(float(iou.mean()), 4),
            "success_rate": round(float(succ.mean()), 4)}


def neck_skull_report() -> Dict:
    """Isolate Compact-Blob performance on neck/skull-bearing slices vs. the rest,
    so the specific robustness against hyperintense neck artifacts is measurable."""
    recs = load_registry()
    neck = [r for r in recs if r.get("has_neck_or_skull")]
    clean = [r for r in recs if not r.get("has_neck_or_skull")]
    rep = {
        "total_evaluations": len(recs),
        "success_dice_threshold": SUCCESS_DICE,
        "neck_or_skull": _aggregate(neck),
        "no_neck_or_skull": _aggregate(clean),
    }
    a, b = rep["neck_or_skull"], rep["no_neck_or_skull"]
    if a["n"] and b["n"] and a["mean_dice"] is not None:
        rep["neck_gap_dice"] = round(b["mean_dice"] - a["mean_dice"], 4)
    return rep


# ---------------------------------------------------------------------------
# Learning Velocity: step-by-step Dice improvement over time
# ---------------------------------------------------------------------------
def _velocity(seq: List[Dict]) -> Dict:
    d = [r["metrics"]["dice"] for r in seq]
    deltas = [round(d[i] - d[i - 1], 4) for i in range(1, len(d))]
    return {
        "n": len(d),
        "dice_series": d,
        "deltas": deltas,
        "mean_delta": round(float(np.mean(deltas)), 4) if deltas else 0.0,
        "net_gain": round(d[-1] - d[0], 4) if len(d) > 1 else 0.0,
        "improving": bool(deltas and np.mean(deltas) > 0),
    }


def learning_velocity(group_by: Optional[str] = "patient_id") -> Dict:
    """Quantify how the human-in-the-loop system improves over time.

    ``group_by`` in {"patient_id", "model_used", None}. Records are ordered by
    timestamp; ``deltas`` are the consecutive Dice differences (positive = the
    system is learning), ``net_gain`` is last minus first.
    """
    recs = sorted(load_registry(), key=lambda r: r.get("timestamp", ""))
    result: Dict = {"overall": _velocity(recs)}
    if group_by:
        groups: Dict[str, List[Dict]] = {}
        for r in recs:
            groups.setdefault(str(r.get(group_by)), []).append(r)
        result["group_by"] = group_by
        result["groups"] = {k: _velocity(v) for k, v in groups.items()}
    return result
