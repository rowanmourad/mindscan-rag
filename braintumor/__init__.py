"""braintumor — end-to-end brain-tumor MRI analysis pipeline.

One consolidated package covering the full clinical-decision-support flow:

    MRI image
      -> preprocessing            (braintumor.preprocessing)
      -> classification           (braintumor.predict / two_stage / decision_fusion)
      -> XAI  (Grad-CAM/++)       (braintumor.xai)
      -> tumor analysis           (braintumor.localization, braintumor.tumor_analysis)
      -> segmentation             (braintumor.segmentation)
      -> structured result        (braintumor.schema)
      -> medical-style report     (braintumor.generate_report)
      -> LLM-ready payload         (braintumor.llm_report)

The single orchestrator that wires all of the above together is
``braintumor.pipeline.BrainTumorPipeline``.

Design note: heavy dependencies (TensorFlow, OpenCV, matplotlib) are imported
lazily inside the submodules, so ``import braintumor`` is cheap. Public names are
resolved on first access via ``__getattr__`` below.
"""
from __future__ import annotations

from . import config  # cheap, no TF

__version__ = "1.0.0"

# Map public attribute name -> (submodule, attribute) for lazy resolution.
_LAZY = {
    "BrainTumorPipeline": ("pipeline", "BrainTumorPipeline"),
    "ModelRegistry": ("registry", "ModelRegistry"),
    "Predictor": ("predict", "Predictor"),
    "TwoStageClassifier": ("two_stage", "TwoStageClassifier"),
    "DecisionFusion": ("decision_fusion", "DecisionFusion"),
    "PredictionResult": ("schema", "PredictionResult"),
    "generate_markdown_report": ("generate_report", "generate_markdown_report"),
    "build_llm_payload": ("llm_report", "build_llm_payload"),
    "preprocess_mri": ("preprocessing", "preprocess_mri"),
    "segment_tumor": ("segmentation", "segment_tumor"),
}

__all__ = ["config", "__version__", *_LAZY.keys()]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        module = importlib.import_module(f".{mod_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
