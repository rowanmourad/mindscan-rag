"""braintumor/preprocessing.py - Unified MRI preprocessing.

Consolidates the resize/scaling logic that was duplicated (and sometimes
inconsistent) across the notebook, GUI, and inference modules into one place.

Design principle (important): **inference preprocessing must match training
preprocessing.** Every recent model in this project (EfficientNet family,
fusion) was trained on raw [0,255] RGB resized to the model's native size, with
the backbone doing its own normalization internally. The original Xception
model was trained with a /255 rescale. So the *default* inference path here does
only: load -> RGB -> resize -> (optional /255). It deliberately does NOT apply
CLAHE / denoising by default, because the models never saw those at train time
and adding them at inference shifts the input distribution and hurts accuracy.

The enhancement utilities (CLAHE, denoise, intensity normalization) are provided
because they are useful for (a) EDA visualization and (b) *retraining* a model
with enhanced inputs. If you retrain with ``enhance=True``, run inference with
``enhance=True`` too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rgb(img_path: str | Path) -> np.ndarray:
    """Load any image as an HxWx3 uint8 RGB array (handles grayscale / RGBA)."""
    from PIL import Image
    return np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Enhancement utilities (off by default at inference; useful for EDA/retraining)
# ---------------------------------------------------------------------------
def apply_clahe(rgb_uint8: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the luminance channel.

    Improves local contrast of subtle lesions without blowing out global
    brightness. Applied on the L channel of LAB so colour (here, grayscale
    replicated to RGB) is preserved.
    """
    import cv2
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def denoise(rgb_uint8: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Edge-preserving denoise. 'bilateral' keeps tumor borders crisp; 'median'
    is cheaper and good for salt-and-pepper speckle."""
    import cv2
    if method == "median":
        return cv2.medianBlur(rgb_uint8, 3)
    return cv2.bilateralFilter(rgb_uint8, d=5, sigmaColor=50, sigmaSpace=50)


def normalize_intensity(rgb_uint8: np.ndarray) -> np.ndarray:
    """Min-max stretch to full 0-255 range (per image). Useful for EDA display."""
    import cv2
    return cv2.normalize(rgb_uint8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def brain_mask(rgb_uint8: np.ndarray) -> Tuple[np.ndarray, int]:
    """Approximate intracranial mask via Otsu + largest component.

    Returns (mask uint8 {0,255}, brain_area_px). Used by tumor-coverage % and as
    an optional crop/skull-context reference. This is NOT true skull-stripping;
    on many sequences it includes skull. Document it as an approximation.
    """
    import cv2
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th)
    area = 0
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        area = int(cv2.contourArea(big))
        cv2.drawContours(mask, [big], -1, 255, thickness=cv2.FILLED)
    return mask, area


def enhance(rgb_uint8: np.ndarray, *, clahe: bool = True,
            do_denoise: bool = True) -> np.ndarray:
    """Standard enhancement bundle for EDA / enhanced-retraining experiments."""
    out = rgb_uint8
    if do_denoise:
        out = denoise(out)
    if clahe:
        out = apply_clahe(out)
    return out


def skull_strip(rgb_uint8: np.ndarray, *, kernel_size: int = 11,
                min_frac: float = 0.05, max_frac: float = 0.98
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate skull-strip / neck-artifact removal by intracranial masking.

    Pipeline (matches the requested blueprint):
      1. Otsu binarization of the grayscale image (tissue vs true-black background).
      2. Morphological CLOSE then OPEN with an elliptical kernel (default 11x11) to
         drop thin disconnected artifacts, eyes, and stray neck/spine regions.
      3. Keep only the LARGEST external contour (the intracranial cavity), filled.
      4. Hard element-wise multiply the {0,1} mask into the RGB image, so skull
         margins, eyes and neck muscles become pure black (0).

    Safety guard: if the resulting mask is implausible (covers < ``min_frac`` or
    > ``max_frac`` of the frame - i.e. Otsu failed and the "brain" is a speck or
    the whole image), the ORIGINAL image is returned unchanged with an all-ones
    mask, so a bad mask never destroys a valid scan.

    Returns ``(stripped_rgb_uint8, mask_uint8{0,1})``.
    """
    import cv2
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [big], -1, 1, thickness=cv2.FILLED)

    frac = float(mask.sum()) / float(mask.size + 1e-9)
    if not (min_frac <= frac <= max_frac):
        # Otsu/morphology failed for this slice - do not destroy the image.
        return rgb_uint8, np.ones(gray.shape, dtype=np.uint8)

    stripped = (rgb_uint8 * mask[..., None]).astype(np.uint8)
    return stripped, mask


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------
def preprocess_mri(
    img_path: str | Path,
    img_size: int,
    *,
    rescale_to_unit: bool = False,
    enhance_input: bool = False,
    skull_strip_input: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Produce (original_rgb_uint8, model_input_batch) for one image.

    Parameters
    ----------
    img_size : the resolution the model expects (e.g. 260 for B2).
    rescale_to_unit : True only for the original /255 Xception model.
    enhance_input : apply CLAHE+denoise. Use ONLY if the model was trained with
        enhanced inputs (default False to match how the shipped models trained).
    skull_strip_input : zero out skull/eyes/neck before the model (see
        :func:`skull_strip`). NOTE: the shipped classifiers were trained on raw
        images, so enabling this changes the input distribution; verify accuracy
        or retrain with stripping. ``original_rgb`` is always the un-stripped image
        (for display/overlays); only the model tensor is stripped.

    Returns
    -------
    original_rgb : full-resolution uint8 RGB (for overlays/preview)
    x : float32 array shaped (1, img_size, img_size, 3), scaled per the model.
    """
    import cv2
    original_rgb = load_rgb(img_path)
    work = enhance(original_rgb) if enhance_input else original_rgb
    if skull_strip_input:
        work, _ = skull_strip(work)
    resized = cv2.resize(work, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32)
    if rescale_to_unit:
        x = x / 255.0
    return original_rgb, x[None, ...]


def preprocess_array(
    rgb_uint8: np.ndarray, img_size: int, *,
    rescale_to_unit: bool = False, enhance_input: bool = False,
    skull_strip_input: bool = False,
) -> np.ndarray:
    """In-memory variant of :func:`preprocess_mri` (returns the model batch only)."""
    import cv2
    work = enhance(rgb_uint8) if enhance_input else rgb_uint8
    if skull_strip_input:
        work, _ = skull_strip(work)
    resized = cv2.resize(work, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32)
    if rescale_to_unit:
        x = x / 255.0
    return x[None, ...]
