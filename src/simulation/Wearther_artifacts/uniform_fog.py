# uniform_fog.py
from typing import Tuple

import numpy as np

MAX_DEPTH = 160.0         # meters
VIS_DIST = 100.0          # distance where t ≈ 0.05
ATMOS_RGB = np.array([220, 220, 235], dtype=np.float32)
K_MEAN = -np.log(0.05) / VIS_DIST  # ≈ 0.0375


def simulate_depth(h: int, w: int) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)
    depth_norm = (h - ys) / h
    return depth_norm[:, None] * MAX_DEPTH


def apply_koschmieder(img: np.ndarray, depth: np.ndarray, k_map: np.ndarray) -> np.ndarray:
    t = np.exp(-k_map * depth)[..., None]
    out = img.astype(np.float32) * t + ATMOS_RGB * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_uniform_fog(img: np.ndarray, strength: float):
    """
    k(x,y) = const = K_MEAN * strength.
    Returns (fogged_bgr_uint8, k_map_float32).
    """
    h, w = img.shape[:2]
    depth = simulate_depth(h, w)
    k_map = np.full((h, w), K_MEAN * strength, dtype=np.float32)
    fogged = apply_koschmieder(img, depth, k_map)
    return fogged, k_map
