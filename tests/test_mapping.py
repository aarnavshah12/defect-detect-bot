import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mapping  # noqa: E402


def _synthetic_pairs():
    # Ground-truth: pixel -> mm is a rotation+scale+translation (an affine, hence a homography).
    s, theta = 0.25, np.deg2rad(7.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([-120.0, 80.0])
    px = np.array([[100, 100], [1800, 120], [1750, 950], [150, 1000], [960, 540], [500, 700]], dtype=float)
    arm = (px @ R.T) * s + t
    return px, arm


def test_fit_and_roundtrip():
    px, arm = _synthetic_pairs()
    H, res = mapping.fit_homography(px, arm)
    assert res.max() < 1e-3  # OpenCV uses float32 internally
    x, y = mapping.pixel_to_arm(960, 540, H)
    assert abs(x - arm[4][0]) < 1e-3 and abs(y - arm[4][1]) < 1e-3


def test_save_load(tmp_path=None):
    px, arm = _synthetic_pairs()
    H, _ = mapping.fit_homography(px, arm)
    d = tmp_path or tempfile.mkdtemp()
    p = os.path.join(str(d), "calibration.npy")
    mapping.save_calibration(H, px, arm, path=p)
    data = mapping.load_calibration(p)
    assert np.allclose(data["H"], H)
    assert data["pixel_pts"].shape == (6, 2)
    x, y = mapping.pixel_to_arm(100, 100, mapping.load_homography(p))
    assert abs(x - arm[0][0]) < 1e-3


def test_missing_raises():
    try:
        mapping.load_homography("/nonexistent/calibration.npy")
    except mapping.CalibrationMissing:
        return
    raise AssertionError("expected CalibrationMissing")


if __name__ == "__main__":
    test_fit_and_roundtrip(); test_save_load(); test_missing_raises(); print("mapping tests OK")
