"""Offline tests of calibrate.auto_collect (manual and carry modes) with fake arm / detector / owner."""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm as arm_mod  # noqa: E402
import calibrate  # noqa: E402
import config  # noqa: E402
import mapping  # noqa: E402
from detect import Detection  # noqa: E402


class FakeArm:
    """Perfect arm; refuses one grid point. The block follows the cup while suction is on."""

    def __init__(self, refuse=()):
        self.pos = [0.0, -160.0, 200.0]
        self.refuse = set(refuse)
        self.calls = []
        self.suction_on = False
        self.block = [0.0, 0.0]  # arm x, y of the block on the table

    def move_to(self, x, y, z, ms=None):
        if (x, y) in self.refuse:
            raise arm_mod.MoveRefused("nope")
        self.pos = [x, y, z]
        self.calls.append(("move", x, y, z))
        if self.suction_on:
            self.block = [x, y]

    def suction(self, on):
        self.suction_on = on
        self.calls.append(("suction", on))

    def home(self):
        self.calls.append(("home",))

    def read_xyz(self):
        return tuple(int(round(v)) for v in self.pos)

    def wait(self, s):
        pass


class FakeDetector:
    """Camera model: pixel = (4*x + 960, -3*y + 100) for the block on the table; nothing while it is carried."""

    def __init__(self, arm):
        self.arm = arm

    def detect(self, frame):
        if self.arm.suction_on:
            return [Detection("green", 0.95, 50, 50, 80, 80)]
        x, y = self.arm.block
        return [Detection("blue", 0.9, 4 * x + 960, -3 * y + 100, 80, 80), Detection("green", 0.95, 50, 50, 80, 80)]


def _frame():
    return np.zeros((1080, 1920, 3), np.uint8)


def _setup():
    config.TARGET_CLASS = "blue"
    config.TABLE_Z_MM, config.BLOCK_HEIGHT_MM, config.CUP_PRESS_MM, config.TRAVEL_Z_MM = 47.0, 40.0, 5.0, 160.0
    config.MIN_RADIUS_MM = 120.0


def test_manual_mode_builds_a_valid_calibration():
    _setup()
    arm = FakeArm(refuse={calibrate.AUTO_GRID[-1]})
    answers = iter([""] * 2 + ["s"] + [""] * 20)  # skip the 3rd point by hand

    def ask(prompt):
        arm.block = [arm.pos[0], arm.pos[1]]  # owner slides the block under the cup
        return next(answers)

    px, ar = calibrate.auto_collect(calibrate.AUTO_GRID, arm, _frame, FakeDetector(arm), ask, "blue", 47.0,
                                    logging.getLogger("t"))
    assert len(px) == len(calibrate.AUTO_GRID) - 2  # one refused, one skipped
    H, _ = mapping.fit_homography(px, ar)
    x, y = mapping.pixel_to_arm(4 * 40 + 960, -3 * (-170) + 100, H)
    assert abs(x - 40) < 0.5 and abs(y + 170) < 0.5
    assert not any(c[0] == "move" and c[3] < 47.0 + calibrate.PRESENT_CLEARANCE_MM for c in arm.calls)


def test_carry_mode_is_hands_off_after_the_first_placement():
    _setup()
    arm = FakeArm(refuse={calibrate.AUTO_GRID[-1]})
    asks = []

    def ask(prompt):
        asks.append(prompt)
        arm.block = [arm.pos[0], arm.pos[1]]  # owner centres the block under the cup once
        return ""

    px, ar = calibrate.auto_collect(calibrate.AUTO_GRID, arm, _frame, FakeDetector(arm), ask, "blue", 47.0,
                                    logging.getLogger("t"), carry=True)
    assert len(asks) == 1
    assert len(px) == len(calibrate.AUTO_GRID) - 1  # one refused
    H, _ = mapping.fit_homography(px, ar)
    x, y = mapping.pixel_to_arm(4 * 40 + 960, -3 * (-170) + 100, H)
    assert abs(x - 40) < 0.5 and abs(y + 170) < 0.5
    assert not arm.suction_on and arm.calls[-1] == ("home",)
    pick_z = 47.0 + 40.0 - 5.0
    assert all(c[3] >= pick_z for c in arm.calls if c[0] == "move")  # never below pick height


def test_carry_mode_detects_a_failed_lift():
    _setup()
    arm = FakeArm()

    class NoSuction(FakeArm):
        def suction(self, on):
            self.calls.append(("suction", on))  # pump "on" but the block never follows

    arm = NoSuction()

    def ask(prompt):
        arm.block = [arm.pos[0], arm.pos[1]]
        return ""

    try:
        calibrate.auto_collect(calibrate.AUTO_GRID, arm, _frame, FakeDetector(arm), ask, "blue", 47.0,
                               logging.getLogger("t"), carry=True)
    except RuntimeError as e:
        assert "did not lift" in str(e)
    else:
        raise AssertionError("expected a failed-lift error")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("calibrate auto tests OK")
