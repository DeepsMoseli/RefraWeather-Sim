# perlin.py
from typing import Tuple

import cv2
import gstools as gs
import numpy as np


def apply_perlin_distortion(
    img: np.ndarray,
    amp: float = 2.0,
    len_scale: float = 15.0,
    var: float = 1.0,
    seed_x: int = 42,
    seed_y: int = 43,
    dtype=np.float16,
    downscale: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp with Exponential SRF sampled on a coarse grid then upsampled.
    Returns (warped_bgr_uint8, uv_map_float16).
    """
    h, w = img.shape[:2]
    h_small = max(1, h // downscale)
    w_small = max(1, w // downscale)

    model_small = gs.Exponential(dim=2, var=var, len_scale=len_scale / downscale)
    srf_x = gs.SRF(model_small, mean=0.0)
    srf_y = gs.SRF(model_small, mean=0.0)

    dx_small = srf_x.structured([np.arange(h_small), np.arange(w_small)], seed=seed_x)
    dy_small = srf_y.structured([np.arange(h_small), np.arange(w_small)], seed=seed_y)

    dx = cv2.resize(dx_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    dx = (dx - dx.mean()) / (dx.std() + 1e-8) * amp
    dy = (dy - dy.mean()) / (dy.std() + 1e-8) * amp

    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    uv_map = np.stack([map_x, map_y], axis=2).astype(dtype)
    return warped, uv_map
