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


def test_save_load_extensionless_and_frame_size():
    px, arm = _synthetic_pairs()
    H, _ = mapping.fit_homography(px, arm)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mycal")  # no .npy extension: must land at exactly this path
    mapping.save_calibration(H, px, arm, path=p, frame_size=(1280, 720))
    assert os.path.exists(p)
    data = mapping.load_calibration(p)
    assert np.allclose(data["H"], H) and data["pixel_pts"].shape == (6, 2)
    assert tuple(data["frame_size"]) == (1280, 720)
    mapping.check_frame_size(data, np.zeros((720, 1280, 3), np.uint8))
    try:
        mapping.check_frame_size(data, np.zeros((1080, 1920, 3), np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("expected frame-size mismatch")


def test_missing_raises():
    try:
        mapping.load_homography("/nonexistent/calibration.npy")
    except mapping.CalibrationMissing:
        return
    raise AssertionError("expected CalibrationMissing")


def test_degenerate_sets_rejected():
    px, arm = _synthetic_pairs()
    # duplicate click
    px2 = px[:4].copy(); px2[3] = px2[0] + [3, 3]
    for bad_px, bad_arm, what in (
        (px2, arm[:4], "duplicate"),
        (np.array([[100, 100], [600, 100], [1100, 100], [1600, 100]], float), arm[:4], "collinear"),
    ):
        try:
            mapping.fit_homography(bad_px, bad_arm)
        except ValueError:
            continue
        raise AssertionError(f"{what} point set should be rejected")


def test_leave_one_out_flags_a_typo():
    px, arm = _synthetic_pairs()
    assert mapping.leave_one_out_residuals(px[:4], arm[:4]) is None  # exact fit: no check possible
    bad = arm.copy(); bad[2][0] += 30.0  # mistyped x on pair 3
    loo = mapping.leave_one_out_residuals(px, bad)
    assert loo[2] > 20  # the typo'd pair cannot be predicted from the others
    good = mapping.leave_one_out_residuals(px, arm)
    assert np.nanmax(good) < 1e-2
    i, with_all, without = mapping.suspect_pair(px, bad)
    assert i == 2 and with_all > 3 and without < 1e-2  # removing pair 3 makes the rest fit exactly
    assert mapping.suspect_pair(px, arm)[1] < 1e-2
    lines = mapping.describe_residuals(px, bad)
    assert any("SUSPECT: pair 3" in ln for ln in lines)
    assert not any("SUSPECT" in ln for ln in mapping.describe_residuals(px, arm))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("mapping tests OK")
