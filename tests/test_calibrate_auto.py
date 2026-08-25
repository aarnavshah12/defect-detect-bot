"""Offline test of calibrate.auto_collect with fake arm / detector / owner."""
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
    """Perfect arm; refuses one grid point. The 'block' is wherever the cup last was."""

    def __init__(self, refuse=()):
        self.pos = [0.0, -160.0, 200.0]
        self.refuse = set(refuse)
        self.calls = []

    def move_to(self, x, y, z, ms=None):
        if (x, y) in self.refuse:
            raise arm_mod.MoveRefused("nope")
        self.pos = [x, y, z]
        self.calls.append(("move", x, y, z))

    def home(self):
        self.calls.append(("home",))

    def read_xyz(self):
        return tuple(int(round(v)) for v in self.pos)

    def wait(self, s):
        pass


class FakeDetector:
    """Camera model: pixel = 4 * arm + (960, 540) (arm y negative -> lower on screen)."""

    def __init__(self, block_at):
        self.block_at = block_at  # callable -> (x, y) arm mm of the block

    def detect(self, frame):
        x, y = self.block_at()
        return [Detection("blue", 0.9, 4 * x + 960, -4 * y + 100, 80, 80), Detection("green", 0.95, 50, 50, 80, 80)]


def test_auto_collect_builds_a_valid_calibration():
    config.TARGET_CLASS = "blue"
    arm = FakeArm(refuse={(150.0, -200.0)})
    block = {"xy": (0.0, 0.0)}
    answers = iter(["", "", "s", "", "", "", "", "", ""])  # skip the 3rd point by hand

    def ask(prompt):
        block["xy"] = (arm.pos[0], arm.pos[1])  # owner slides the block under the cup
        return next(answers)

    det = FakeDetector(lambda: block["xy"])
    log = logging.getLogger("test")
    px, ar = calibrate.auto_collect(calibrate.AUTO_GRID, arm, lambda: np.zeros((10, 10, 3), np.uint8), det, ask,
                                    "blue", 47.0, log)
    assert len(px) == 7  # 9 points - 1 refused - 1 skipped
    H, _ = mapping.fit_homography(px, ar)
    x, y = mapping.pixel_to_arm(4 * 40 + 960, -4 * (-170) + 100, H)
    assert abs(x - 40) < 0.5 and abs(y + 170) < 0.5
    assert arm.calls[-1] == ("home",)
    assert not any(c[0] == "move" and c[3] < 47.0 + calibrate.PRESENT_CLEARANCE_MM for c in arm.calls)  # never below present height


if __name__ == "__main__":
    test_auto_collect_builds_a_valid_calibration()
    print("calibrate auto tests OK")
