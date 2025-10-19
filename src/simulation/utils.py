# utils.py
import json
import os
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_selection_json(basenames, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(list(basenames), f, indent=2)


def safe_imread(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def make_checkerboard(shape: Tuple[int, int], tile_size: int = 32) -> np.ndarray:
    """
    Create a grayscale checkerboard of size (H, W) with squares of tile_size.
    Returns uint8 in {0,255}.
    """
    H, W = shape[:2]
    rows = (np.arange(H) // tile_size)[:, None]
    cols = (np.arange(W) // tile_size)[None, :]
    board = ((rows + cols) % 2).astype(np.uint8) * 255
    return board


def read_calib_parameters(json_path: str):
    """
    Expected JSON layout:
    {
      "K": [[...],[...],[...]],      # or list of Ks
      "D": [[k1,k2,p1,p2,k3,...]],   # or list of Ds
    }
    Returns (success, Ks, Ds, Rs, Ts) for compatibility with your older code.
    """
    if not os.path.exists(json_path):
        return False, None, None, None, None

    with open(json_path, "r") as f:
        data = json.load(f)

    Ks = data.get("K")
    Ds = data.get("D")
    if Ks is None or Ds is None:
        return False, None, None, None, None

    # normalize to list-of-arrays
    if isinstance(Ks[0][0], (int, float)):
        Ks = [Ks]
    if isinstance(Ds[0], (int, float)):
        Ds = [Ds]

    Ks = [np.asarray(k, dtype=np.float32) for k in Ks]
    Ds = [np.asarray(d, dtype=np.float32) for d in Ds]
    return True, Ks, Ds, None, None


def jitter_distortion(D: np.ndarray, alpha: float, abs_min: float = 1e-6) -> np.ndarray:
    """
    Jitter each distortion coefficient by ±max(alpha*|v|, abs_min).
    """
    import random

    Dp = []
    for v in D.flatten():
        delta = max(abs(v) * alpha, abs_min)
        Dp.append(v + random.uniform(-delta, delta))
    return np.array(Dp, dtype=D.dtype)


def visualize_kmap_on_white(k_map: np.ndarray) -> np.ndarray:
    """Normalize float32 map to [0,255] uint8 (single-channel)."""
    k_min, k_max = float(k_map.min()), float(k_map.max())
    if k_max - k_min < 1e-6:
        return np.zeros_like(k_map, dtype=np.uint8)
    return ((k_map - k_min) / (k_max - k_min) * 255).astype(np.uint8)
