"""Pixel -> arm coordinate mapping.

The saved homography is the ONLY thing that maps pixels to arm (x, y) mm.
Z is not part of it; it is a measured constant in config.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np

import config


class CalibrationMissing(FileNotFoundError):
    pass


def save_calibration(H: np.ndarray, pixel_pts, arm_pts, path: str | None = None) -> None:
    """Persist the homography plus the point pairs it came from (for auditing)."""
    path = path or config.CALIBRATION_PATH
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {H.shape}")
    payload = {
        "H": H,
        "pixel_pts": np.asarray(pixel_pts, dtype=np.float64),
        "arm_pts": np.asarray(arm_pts, dtype=np.float64),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_size": (config.FRAME_WIDTH, config.FRAME_HEIGHT),
    }
    np.save(path, payload, allow_pickle=True)


def load_calibration(path: str | None = None) -> dict:
    path = path or config.CALIBRATION_PATH
    if not os.path.exists(path):
        raise CalibrationMissing(
            f"{path} not found. The owner must run `python calibrate.py` on the physical rig first."
        )
    data = np.load(path, allow_pickle=True).item()
    H = np.asarray(data["H"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"{path}: homography must be 3x3, got {H.shape}")
    return data


def load_homography(path: str | None = None) -> np.ndarray:
    return load_calibration(path)["H"]


def pixel_to_arm(px: float, py: float, H: np.ndarray | None = None) -> tuple[float, float]:
    """Map a pixel centre to arm (x, y) in mm via cv2.perspectiveTransform."""
    if H is None:
        H = load_homography()
    pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def fit_homography(pixel_pts, arm_pts) -> tuple[np.ndarray, np.ndarray]:
    """cv2.findHomography over the clicked/jogged point pairs.

    Returns (H, residuals_mm) where residuals_mm[i] is the reprojection error of pair i.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 1, 2)
    arm_pts = np.asarray(arm_pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(pixel_pts) < 4:
        raise ValueError("need at least 4 point pairs for a homography")
    H, _mask = cv2.findHomography(pixel_pts, arm_pts, 0)
    if H is None:
        raise ValueError("cv2.findHomography failed (points collinear / degenerate?)")
    proj = cv2.perspectiveTransform(pixel_pts, H)
    residuals = np.linalg.norm(proj.reshape(-1, 2) - arm_pts.reshape(-1, 2), axis=1)
    return H, residuals
