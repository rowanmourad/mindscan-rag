"""Global reproducibility seeding (H1).

The original training notebook set no seeds, so every run produced a
different val/test split shuffle order and different weight init, causing
+/-1-2% run-to-run accuracy swings. That makes it impossible to attribute a
change to a real improvement vs. noise.

Call :func:`set_global_seed` ONCE, before building any generators or models.
"""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs.

    Note: full bit-exact determinism on GPU also needs
    ``TF_DETERMINISTIC_OPS=1`` (set here) and can slow training; the dominant
    sources of variance (data shuffle order + weight init) are covered
    regardless of device.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # keras.utils.set_random_seed seeds python+numpy+tf together on TF>=2.9
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
    except ImportError:
        pass
