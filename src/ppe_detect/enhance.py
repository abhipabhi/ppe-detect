"""Low-light preprocessing (CLAHE on the L channel in LAB space)."""

from __future__ import annotations

import cv2
import numpy as np


def mean_brightness(image_bgr: np.ndarray) -> float:
    """Mean grayscale intensity in [0, 255]."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_low_light(image_bgr: np.ndarray, threshold: float = 80.0) -> bool:
    return mean_brightness(image_bgr) < threshold


def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on lightness only.

    Operating on the L channel of LAB leaves chroma untouched, so class-relevant
    colors (helmet / clothing) are preserved while dark regions gain contrast.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def maybe_enhance(
    image_bgr: np.ndarray, mode: str, threshold: float = 80.0
) -> tuple[np.ndarray, bool]:
    """Apply CLAHE according to mode ("on", "off", or brightness-gated "auto").

    Returns the (possibly enhanced) image and whether enhancement was applied.
    """
    if mode == "on" or (mode == "auto" and is_low_light(image_bgr, threshold)):
        return apply_clahe(image_bgr), True
    if mode in ("off", "auto"):
        return image_bgr, False
    raise ValueError(f"unknown enhance mode: {mode!r}")
