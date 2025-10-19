# hetero_fog.py
from typing import List

import cv2
import numpy as np
import noise

from uniform_fog import K_MEAN, simulate_depth, apply_koschmieder


# multi-scale recipe (can be overridden by caller)
SCALES: List[int] = [4, 8, 16, 32, 64, 128]
WEIGHTS: List[float] = [0.3, 0.22, 0.15, 0.11, 0.08, 0.07]


def perlin_map(h: int, w: int, scale: float) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    for yy in range(h):
        for xx in range(w):
            m[yy, xx] = noise.pnoise2(xx / scale, yy / scale, octaves=1, repeatx=w, repeaty=h)
    mmin, mmax = m.min(), m.max()
    return (m - mmin) / (mmax - mmin + 1e-12)


def apply_heterogeneous_fog(img: np.ndarray, strength: float):
    """
    Combine multi-scale Perlin to form k(x,y) with mean K_MEAN*strength.
    Returns (fogged_bgr_uint8, k_map_float32).
    """
    h, w = img.shape[:2]
    depth = simulate_depth(h, w)
    fields = [perlin_map(h, w, s) for s in SCALES]
    combined = sum(wt * f for wt, f in zip(WEIGHTS, fields)) / sum(WEIGHTS)
    base_k = combined * (K_MEAN / (combined.mean() + 1e-12))
    k_map = (base_k * strength).astype(np.float32)
    fogged = apply_koschmieder(img, depth, k_map)
    return fogged, k_map
