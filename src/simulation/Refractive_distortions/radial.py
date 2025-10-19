# radial.py
from typing import Tuple

import cv2
import numpy as np

from ..utils import read_calib_parameters


def apply_redistortion(
    undistorted: np.ndarray,
    K_orig: np.ndarray,
    D_orig: np.ndarray,
    image_size: Tuple[int, int],
    alpha: float = 0.0,
):
    """
    Re-apply your camera distortion to an undistorted image.
    Returns (redistorted_bgr, uv_map_float16, warped_checker_bgr).
    """
    w, h = image_size
    K_new, _ = cv2.getOptimalNewCameraMatrix(K_orig, D_orig, (w, h), alpha, (w, h))

    map_x, map_y = cv2.initInverseRectificationMap(
        cameraMatrix=K_orig,
        distCoeffs=D_orig,
        R=np.eye(3),
        newCameraMatrix=K_new,
        size=(w, h),
        m1type=cv2.CV_32FC1,
    )

    redistorted = cv2.remap(
        undistorted, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    uv_map = np.dstack((map_x, map_y)).astype(np.float16)

    return redistorted, uv_map


def load_first_calib(json_path: str = "dashcam_calib.json"):
    ok, Ks, Ds, _, _ = read_calib_parameters(json_path)
    if not ok:
        raise RuntimeError("Could not read calibration data.")
    return Ks[0], Ds[0].ravel()
