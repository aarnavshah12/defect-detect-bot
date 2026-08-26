"""Pixel -> arm coordinate mapping.

The saved homography is the ONLY thing that maps pixels to arm (x, y) mm.
Z is not part of it; it is a measured constant in config.

The pixel side of the calibration is the detector's bbox centre of a block sitting on each tape
mark (not the mark itself): the camera sees block tops, so calibrating on the same quantity that
pick.py maps folds the block-height parallax into H (block tops lie on a plane parallel to the table,
so pixel -> table (x, y) is still exactly a homography).
"""

from __future__ import annotations

import itertools
import os
import time

import cv2
import numpy as np

import config

MIN_PIXEL_SEPARATION = 20.0  # px: two marks closer than this = duplicate click
MIN_ARM_SEPARATION = 5.0     # mm
MIN_PAIRS = 4
RECOMMENDED_PAIRS = 5        # at exactly 4 the fit is exact and residuals cannot reveal a typo


class CalibrationMissing(FileNotFoundError):
    pass


def save_calibration(H: np.ndarray, pixel_pts, arm_pts, path: str | None = None,
                     frame_size: tuple[int, int] | None = None, **extra) -> None:
    """Persist the homography plus the point pairs it came from (for auditing). `extra` fields are stored too."""
    path = path or config.CALIBRATION_PATH
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {H.shape}")
    payload = {
        "H": H,
        "pixel_pts": np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2),
        "arm_pts": np.asarray(arm_pts, dtype=np.float64).reshape(-1, 2),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_size": tuple(int(v) for v in (frame_size or (config.FRAME_WIDTH, config.FRAME_HEIGHT))),
        **extra,
    }
    with open(path, "wb") as f:  # file object: numpy never appends ".npy" to the chosen path
        np.save(f, payload, allow_pickle=True)


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


def check_frame_size(data: dict, frame: np.ndarray) -> None:
    """Refuse to use a calibration made at a different camera resolution."""
    saved = tuple(int(v) for v in data.get("frame_size", ()))
    actual = (int(frame.shape[1]), int(frame.shape[0]))
    if saved and saved != actual:
        raise ValueError(f"calibration was made at {saved[0]}x{saved[1]} but the camera is delivering "
                         f"{actual[0]}x{actual[1]}; re-run calibrate.py")


def pixel_to_arm(px: float, py: float, H: np.ndarray | None = None) -> tuple[float, float]:
    """Map a pixel centre to arm (x, y) in mm via cv2.perspectiveTransform."""
    if H is None:
        H = load_homography()
    pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def translate_homography(H: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """H' = T(dx, dy) . H : the same mapping shifted by a constant (dx, dy) mm in arm coordinates.

    Used to fold a rig-measured constant offset (e.g. from the single hand placement that seeds a carry
    calibration) into the homography itself, so it stays the ONLY pixel -> arm mapping.
    """
    T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return T @ np.asarray(H, dtype=np.float64)


def _line_dist(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    d = b - a
    return abs(d[0] * (c[1] - a[1]) - d[1] * (c[0] - a[0])) / np.linalg.norm(d)


def check_general_position(pts, min_sep: float, label: str) -> None:
    """Raise ValueError unless no two points coincide and some 4 points have no 3 (near-)collinear.

    A homography is uniquely determined only then; cv2.findHomography happily returns a garbage
    matrix with zero residuals for a duplicate click or marks in a line.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    for i, j in itertools.combinations(range(len(pts)), 2):
        if np.linalg.norm(pts[i] - pts[j]) < min_sep:
            raise ValueError(f"{label} points {i + 1} and {j + 1} coincide (< {min_sep:g} apart): duplicate mark?")
    for quad in itertools.combinations(range(len(pts)), 4):
        if all(_line_dist(pts[i], pts[j], pts[k]) >= min_sep for i, j, k in itertools.combinations(quad, 3)):
            return
    raise ValueError(f"{label} points are (near-)collinear: no 4 marks in general position; "
                     f"spread the marks over the whole pick area")


def fit_homography(pixel_pts, arm_pts) -> tuple[np.ndarray, np.ndarray]:
    """cv2.findHomography over the point pairs.

    Returns (H, residuals_mm): in-sample reprojection error per pair. NOTE: with exactly 4 pairs the
    fit is exact and residuals are 0 by construction; use leave_one_out_residuals() for a real check.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 1, 2)
    arm_pts = np.asarray(arm_pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(pixel_pts) != len(arm_pts):
        raise ValueError("pixel_pts and arm_pts must have the same length")
    if len(pixel_pts) < MIN_PAIRS:
        raise ValueError(f"need at least {MIN_PAIRS} point pairs for a homography")
    check_general_position(pixel_pts, MIN_PIXEL_SEPARATION, "pixel")
    check_general_position(arm_pts, MIN_ARM_SEPARATION, "arm")
    H, _mask = cv2.findHomography(pixel_pts, arm_pts, 0)
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError("cv2.findHomography failed (points degenerate?)")
    proj = cv2.perspectiveTransform(pixel_pts, H)
    residuals = np.linalg.norm(proj.reshape(-1, 2) - arm_pts.reshape(-1, 2), axis=1)
    return H, residuals


def leave_one_out_residuals(pixel_pts, arm_pts) -> np.ndarray | None:
    """For n >= 5 pairs: error (mm) of each pair when predicted from the other n-1. None if n < 5.

    This is the check that actually reveals a mistyped arm value or a mis-clicked mark.
    A pair whose removal leaves a degenerate set gets NaN.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    arm_pts = np.asarray(arm_pts, dtype=np.float64).reshape(-1, 2)
    n = len(pixel_pts)
    if n < RECOMMENDED_PAIRS:
        return None
    out = np.full(n, np.nan)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        try:
            H, _ = fit_homography(pixel_pts[keep], arm_pts[keep])
        except ValueError:
            continue
        x, y = pixel_to_arm(pixel_pts[i][0], pixel_pts[i][1], H)
        out[i] = float(np.hypot(x - arm_pts[i][0], y - arm_pts[i][1]))
    return out


def suspect_pair(pixel_pts, arm_pts) -> tuple[int, float, float] | None:
    """(index, max_residual_with_all, max_residual_without_it) for the pair whose removal helps most.

    A single mistyped value also inflates the leave-one-out error of its neighbours, so the culprit is
    better identified as the pair without which the remaining pairs fit cleanly. None if n < 5.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    arm_pts = np.asarray(arm_pts, dtype=np.float64).reshape(-1, 2)
    n = len(pixel_pts)
    if n < RECOMMENDED_PAIRS:
        return None
    _, res_all = fit_homography(pixel_pts, arm_pts)
    best = None
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        try:
            _, res = fit_homography(pixel_pts[keep], arm_pts[keep])
        except ValueError:
            continue
        if best is None or res.max() < best[1]:
            best = (i, float(res.max()))
    return None if best is None else (best[0], float(res_all.max()), best[1])


def describe_residuals(pixel_pts, arm_pts) -> list[str]:
    """Human-readable residual report lines used by calibrate.py (collect and --check)."""
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    arm_pts = np.asarray(arm_pts, dtype=np.float64).reshape(-1, 2)
    lines = []
    loo = leave_one_out_residuals(pixel_pts, arm_pts)
    if loo is None:
        lines.append(f"  {len(pixel_pts)} pairs -> exact fit: residuals are 0 by construction and CANNOT reveal a "
                     f"mistyped value or a mis-clicked mark. Add a {RECOMMENDED_PAIRS}th mark, or rely on --verify.")
        return lines
    for i, r in enumerate(loo):
        val = "n/a (degenerate without it)" if np.isnan(r) else f"{r:.2f} mm"
        lines.append(f"  pair {i + 1}: pixel=({pixel_pts[i][0]:.0f},{pixel_pts[i][1]:.0f}) "
                     f"arm=({arm_pts[i][0]:.1f},{arm_pts[i][1]:.1f}) leave-one-out error={val}")
    finite = loo[np.isfinite(loo)]
    if len(finite):
        lines.append(f"  worst {finite.max():.2f} mm, mean {finite.mean():.2f} mm  (>3 mm usually means a mistyped "
                     f"value or a mis-click; everywhere large = camera moved or marks not on one plane)")
    sus = suspect_pair(pixel_pts, arm_pts)
    if sus is not None:
        i, with_all, without = sus
        if with_all > 3.0 and without < with_all / 3:
            lines.append(f"  SUSPECT: pair {i + 1} looks wrong - max residual drops from {with_all:.2f} mm to "
                         f"{without:.2f} mm without it. Undo it (u) and redo that mark.")
    return lines
