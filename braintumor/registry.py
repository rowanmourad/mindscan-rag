"""braintumor/registry.py - Discover and load the best available trained models.

The project has accumulated several trained Keras models across different
folders. Rather than hard-coding a path in the GUI / pipeline, this registry
scans ``config.MODEL_SEARCH_DIRS`` and resolves, for each role, the strongest
model that actually exists on disk. The GUI calls ``ModelRegistry.latest_4class()``
and always gets the best model without code changes.

Roles
-----
    4class   : single-stage glioma/meningioma/notumor/pituitary  (primary path)
    3class   : tumor-subtype glioma/meningioma/pituitary
    binary   : healthy/tumor                                    (stage 1)
    subtype  : == 3class trained on tumor-only                   (stage 2)
    fusion   : dual-backbone 4-class

Each known model carries metadata (input size, whether it needs /255 rescaling,
architecture) so the inference code preprocesses correctly. Unknown ``.keras``
files are still usable: input size is read from the model and rescaling defaults
to raw [0,255] (the EfficientNet convention used by every recent model here).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import config


# Process-wide model cache (keyed by absolute path). Keras-loaded models are held
# statically here so they are loaded at most once per file per process, no matter
# how many ModelRegistry / pipeline instances are created. Cleared only on exit.
_MODEL_CACHE: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Known-model metadata. Higher ``priority`` wins when several models fill the
# same role. Values reflect how each model was actually trained.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    filename: str
    role: str                 # 4class | 3class | binary | subtype | fusion
    img_size: int
    rescale_to_unit: bool     # True only for the original /255 Xception model
    architecture: str
    priority: int             # higher = preferred for its role
    note: str = ""


KNOWN_MODELS: List[ModelSpec] = [
    # --- best single 4-class model (cleaned B2) -> primary path ---
    ModelSpec("effnet_b2_4c_clean.keras", "4class", 260, False,
              "EfficientNetB2", 100, "Best 4-class (~94.9% test, 1 glioma quarantined)."),
    ModelSpec("fusion_eff_xcep_4c_clean.keras", "fusion", 260, False,
              "EfficientNetB2+Xception late fusion", 90, "Dual-backbone fusion model."),
    # --- original notebook models ---
    ModelSpec("brain_tumor_efficientnet.keras", "4class", 224, False,
              "EfficientNetB0", 60, "Original 4-class EfficientNetB0 (~90%)."),
    ModelSpec("brain_tumor_model.keras", "4class", 299, True,
              "Xception", 50, "Original Xception 4-class; trained with /255 rescale."),
    ModelSpec("brain_tumor_efficientnet_3class.keras", "3class", 224, False,
              "EfficientNetB0", 60, "Original 3-class (tumor subtype) EfficientNetB0."),
    # --- hierarchical Keras models (produced by train_binary / train_subtype) ---
    ModelSpec("binary_b2_clean.keras", "binary", 260, False, "EfficientNetB2", 100, ""),
    ModelSpec("binary_b2.keras", "binary", 260, False, "EfficientNetB2", 90, ""),
    ModelSpec("subtype_b2_clean.keras", "subtype", 260, False, "EfficientNetB2", 100, ""),
    ModelSpec("subtype_b2.keras", "subtype", 260, False, "EfficientNetB2", 90, ""),
]

_BY_NAME: Dict[str, ModelSpec] = {m.filename: m for m in KNOWN_MODELS}


@dataclass
class ResolvedModel:
    path: Path
    spec: ModelSpec
    _model: object = None     # lazily loaded Keras model

    @property
    def img_size(self) -> int:
        return self.spec.img_size

    @property
    def rescale_to_unit(self) -> bool:
        return self.spec.rescale_to_unit

    def load(self):
        """Load (and cache) the Keras model.

        Caching is two-tier: per-instance (``self._model``) and PROCESS-WIDE
        (``_MODEL_CACHE`` keyed by absolute path). The process-wide cache means
        the (expensive) Keras load happens at most once per file per process - any
        later ``ModelRegistry`` / ``BrainTumorPipeline`` instance reuses the
        already-resident weights, so subsequent scans skip model loading entirely.

        Fusion models saved with bare Lambda preprocessing layers cannot be
        deserialized directly by Keras 3; for those we fall back to the
        reconstruct-and-load-weights path in ``braintumor.fusion``.
        """
        if self._model is not None:
            return self._model

        key = str(self.path.resolve())
        if key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[key]
            return self._model

        import tensorflow as tf
        print(f"[registry] loading {self.spec.role} model: {self.path.name} "
              f"({self.spec.architecture}, {self.spec.img_size}px)")
        try:
            model = tf.keras.models.load_model(self.path)
        except Exception:
            if self.spec.role == "fusion":
                from .fusion import load_fusion_model
                model = load_fusion_model(self.path, n_classes=4,
                                          img_size=self.spec.img_size)
            else:
                raise
        _MODEL_CACHE[key] = model
        self._model = model
        return self._model


class ModelRegistry:
    """Scans the search dirs once and answers role queries."""

    def __init__(self, search_dirs: Optional[List[Path]] = None,
                 class_indices_path: Optional[Path] = None):
        self.search_dirs = [Path(d) for d in (search_dirs or config.MODEL_SEARCH_DIRS)]
        self.class_indices_path = (
            Path(class_indices_path) if class_indices_path
            else config.ROOT / "class_indices.json"
        )
        self._found: Dict[str, ResolvedModel] = {}   # role -> best ResolvedModel
        self._unknown: List[Path] = []
        self._scan()

    # ------------------------------------------------------------------
    def _scan(self) -> None:
        best_priority: Dict[str, int] = {}
        seen: set = set()
        for d in self.search_dirs:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.keras")):
                if p.name in seen:
                    continue
                seen.add(p.name)
                spec = _BY_NAME.get(p.name)
                if spec is None:
                    self._unknown.append(p)
                    continue
                if spec.priority > best_priority.get(spec.role, -1):
                    best_priority[spec.role] = spec.priority
                    self._found[spec.role] = ResolvedModel(path=p, spec=spec)

    # ------------------------------------------------------------------
    # Role accessors (return None if no model for that role exists yet)
    # ------------------------------------------------------------------
    def get(self, role: str) -> Optional[ResolvedModel]:
        return self._found.get(role)

    def latest_4class(self) -> Optional[ResolvedModel]:
        return self._found.get("4class")

    def latest_3class(self) -> Optional[ResolvedModel]:
        return self._found.get("3class") or self._found.get("subtype")

    def latest_binary(self) -> Optional[ResolvedModel]:
        return self._found.get("binary")

    def latest_subtype(self) -> Optional[ResolvedModel]:
        return self._found.get("subtype") or self._found.get("3class")

    def latest_fusion(self) -> Optional[ResolvedModel]:
        return self._found.get("fusion")

    # Short aliases (match the public API spec) ------------------------
    def four_class(self) -> Optional[ResolvedModel]:
        return self.latest_4class()

    def three_class(self) -> Optional[ResolvedModel]:
        return self.latest_3class()

    def binary(self) -> Optional[ResolvedModel]:
        return self.latest_binary()

    def subtype(self) -> Optional[ResolvedModel]:
        return self.latest_subtype()

    def fusion(self) -> Optional[ResolvedModel]:
        return self.latest_fusion()

    def has_two_stage(self) -> bool:
        return self.latest_binary() is not None and self.latest_subtype() is not None

    # ------------------------------------------------------------------
    def class_names(self, n: int = 4) -> List[str]:
        """Canonical class order. Reads class_indices.json if present, else config.
        Normalizes the legacy 'no_tumor' label to 'notumor'."""
        names = None
        if self.class_indices_path.exists():
            try:
                with open(self.class_indices_path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                if all(str(k).isdigit() for k in d.keys()):
                    names = [d[str(i)] for i in range(len(d))]
                else:
                    names = [k for k, _ in sorted(d.items(), key=lambda kv: int(kv[1]))]
            except Exception:
                names = None
        if not names:
            names = list(config.CLASSES_4) if n == 4 else list(config.CLASSES_3)
        names = ["notumor" if c in ("no_tumor", "no tumor", "none") else c for c in names]
        return names

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = ["Model registry scan:"]
        for role in ("4class", "fusion", "binary", "subtype", "3class"):
            rm = self._found.get(role)
            if rm:
                lines.append(f"  [{role:<8}] {rm.path.name}  "
                             f"({rm.spec.architecture}, {rm.spec.img_size}px)"
                             f"{'  /255' if rm.spec.rescale_to_unit else ''}")
            else:
                lines.append(f"  [{role:<8}] (none found)")
        if self._unknown:
            lines.append("  unknown .keras files (usable, metadata inferred):")
            for p in self._unknown:
                lines.append(f"     - {p}")
        # Note the PyTorch alternative track if present
        pth = list((config.ROOT / "models").glob("*.pth")) if (config.ROOT / "models").exists() else []
        if pth:
            lines.append("  PyTorch checkpoints (alternative track, not loaded by Keras pipeline):")
            for p in pth:
                lines.append(f"     - {p.name}")
        return "\n".join(lines)


# Convenience singleton-style accessor -------------------------------------
_DEFAULT: Optional[ModelRegistry] = None


def default_registry() -> ModelRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ModelRegistry()
    return _DEFAULT


if __name__ == "__main__":
    print(ModelRegistry().summary())
