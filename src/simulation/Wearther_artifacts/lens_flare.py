# lens_flare.py
from typing import Tuple

import numpy as np


def apply_flare(img: np.ndarray, intensity: float, radius_frac: float, seed: int = 24):
    """
    Add radial Gaussian glare near upper region.
    Returns (out_bgr_uint8, mask_float32 [0,1]).
    """
    import cv2

    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    cx = rng.uniform(w * 0.3, w * 0.7)
    cy = rng.uniform(0, h * 0.3)
    diag = np.sqrt(h * h + w * w)
    radius = diag * radius_frac

    ys, xs = np.indices((h, w), dtype=np.float32)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = np.exp(-0.5 * (dist / radius) ** 2)
    mask = np.clip(mask, 0, 1).astype(np.float32)
    out = img.astype(np.float32) + intensity * 255.0 * mask[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, mask
