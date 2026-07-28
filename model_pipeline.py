"""
model_pipeline.py
------------------
Loads the trained MindScan tumor-classification models (via the
`braintumor` package) and exposes a simple predict_tumor() function.

Requires the `braintumor/` package and the `models/` folder (with the
trained .keras models + class_indices.json) to be present under REPO_DIR
(see config.py). These are provided as part of the MindScan project
install -- this script does not train or download them.
"""

import os

import keras
from keras.layers import BatchNormalization

from config import MODELS_DIR, REPO_DIR


def _patch_batchnorm_from_config() -> None:
    """
    Older-format .keras models can include BatchNormalization config keys
    (renorm, renorm_clipping, renorm_momentum) that newer Keras versions no
    longer accept. This patches from_config() to silently drop them so the
    pretrained models still load. Safe to call multiple times.
    """
    def _patched_from_config(cls, config):
        config = dict(config)
        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
            config.pop(key, None)
        return cls(**config)

    BatchNormalization.from_config = classmethod(_patched_from_config)


_patch_batchnorm_from_config()


def resave_fixed_models(models_dir=MODELS_DIR) -> None:
    """
    One-time maintenance utility: re-saves the shipped .keras models after
    applying the BatchNormalization compatibility patch above, so future
    loads don't need the patch at all. Not required for normal use --
    predict_tumor() already applies the patch at import time.
    """
    four_class = os.path.join(models_dir, "brain_tumor_efficientnet.keras")
    three_class = os.path.join(models_dir, "brain_tumor_efficientnet_3class.keras")

    if os.path.exists(four_class):
        model = keras.models.load_model(four_class)
        model.save(os.path.join(models_dir, "brain_tumor_efficientnet_fixed.keras"))
        print("Re-saved:", four_class)

    if os.path.exists(three_class):
        model3 = keras.models.load_model(three_class)
        model3.save(os.path.join(models_dir, "brain_tumor_efficientnet_3class_fixed.keras"))
        print("Re-saved:", three_class)


_pipeline = None


def load_pipeline(verbose: bool = True):
    """
    Loads the BrainTumorPipeline once and caches it. Raises a clear error
    if the `braintumor` package can't be found under REPO_DIR (see
    config.py -- set MINDSCAN_REPO_DIR if it lives somewhere else).
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not os.path.exists(os.path.join(REPO_DIR, "braintumor")):
        raise RuntimeError(
            f"braintumor package not found under {REPO_DIR}. "
            "Set the MINDSCAN_REPO_DIR environment variable to the folder "
            "that contains the `braintumor/` package and `models/` folder."
        )

    from braintumor.pipeline import BrainTumorPipeline

    # verbose=True prints which models were auto-discovered (single / fusion
    # / two-stage) and which segmentation tier is active.
    _pipeline = BrainTumorPipeline(verbose=verbose)
    print("MindScan pipeline loaded. Decision-fusion sources:",
          [s.name for s in _pipeline.sources])
    return _pipeline


def predict_tumor(image_path: str, run_xai: bool = False, run_segmentation: bool = False,
                   run_tumor_analysis: bool = False) -> dict:
    """
    Adapter over BrainTumorPipeline.analyze().

    Returns:
        {"predicted_class": str, "confidence": float, "all_probabilities": dict}

    XAI / segmentation / tumor-analysis are OFF by default since the RAG
    report only needs the classification result -- flip them on if you want
    to enrich the report prompt with tumor location/size/attention-focus.
    """
    pipe = load_pipeline()

    out = pipe.analyze(
        image_path,
        run_xai=run_xai,
        run_segmentation=run_segmentation,
        run_tumor_analysis=run_tumor_analysis,
        make_figures=False,
    )
    r = out["result"]

    result = {
        "predicted_class": r.prediction,
        "confidence": r.confidence,
        "all_probabilities": r.per_class_probabilities,
    }
    # Kept on the side so callers can optionally pull in location/size/XAI
    # without another pipeline call.
    result["_full_result"] = r
    return result
