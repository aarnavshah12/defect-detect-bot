"""Offline test of the pick loop with a fake detector, a recording fake arm, and a synthetic homography."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import mapping  # noqa: E402
import pick  # noqa: E402
from detect import Detection  # noqa: E402


class FakeDetector:
    def __init__(self, script):
        self.script = list(script)  # one list of detections per detect() call
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.script.pop(0) if self.script else []


class FakeArm:
    def __init__(self):
        self.calls = []

    def move_to(self, x, y, z, ms=1000):
        self.calls.append(("move", round(x), round(y), round(z)))

    def suction(self, on):
        self.calls.append(("suction", on))

    def home(self, ms=1500):
        self.calls.append(("home",))

    def wait(self, s):
        pass


def _setup():
    config.TABLE_Z_MM = 60.0
    config.DROP_XYZ_MM = (150.0, -150.0, 70.0)
    config.HOME_XYZ_MM = (0.0, -160.0, 210.0)
    config.REACH_X_MM = (-200.0, 200.0)
    config.REACH_Y_MM = (-280.0, -80.0)
    config.REACH_Z_MM = (60.0, 250.0)
    # pixel (px, py) -> arm: x = 0.25*px - 240, y = -0.25*py - 100   (pure scale/offset homography)
    H = np.array([[0.25, 0, -240.0], [0, -0.25, -100.0], [0, 0, 1.0]])
    return H


def frame():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_picks_highest_conf_red_then_stops():
    H = _setup()
    red_a = Detection("red", 0.6, 400, 400, 90, 90)   # -> arm (-140, -200)
    red_b = Detection("red", 0.9, 1200, 600, 90, 90)  # -> arm (60, -250)
    green = Detection("green", 0.95, 800, 500, 90, 90)
    det = FakeDetector([
        [red_a, red_b, green],  # cycle 1 -> picks red_b
        [green],                # lift check: pick spot clear
        [red_a, green],         # cycle 2 -> picks red_a
        [green],                # lift check
        [green],                # cycle 3 -> no red -> done
    ])
    arm = FakeArm()
    loop = pick.PickLoop(det, arm, frame, H, dry_run=False, max_cycles=10)
    loop.arm = arm
    assert loop.run() == 2
    moves = [c for c in arm.calls if c[0] == "move"]
    assert moves[0] == ("move", 60, -250, 100)   # hover over red_b first (highest conf)
    assert moves[1] == ("move", 60, -250, 60)    # descend to table Z
    assert ("suction", True) in arm.calls and ("suction", False) in arm.calls
    assert moves[3] == ("move", 150, -150, 110)  # drop hover
    assert det.calls == 5


def test_out_of_reach_is_skipped_and_ignored():
    H = _setup()
    far = Detection("red", 0.9, 1900, 50, 90, 90)  # -> arm (235, -112): x outside reach
    near = Detection("red", 0.5, 800, 500, 90, 90)  # -> arm (-40, -225)
    det = FakeDetector([[far, near], [far, near], [], [far]])  # skip far; pick near; lift ok; far ignored -> done
    arm = FakeArm()
    loop = pick.PickLoop(det, arm, frame, H, max_cycles=10)
    assert loop.run() == 1
    assert loop.ignored == [(1900, 50)]
    assert not any(c == ("move", 235, -112, 100) for c in arm.calls)


def test_lift_failed_releases_and_ignores():
    H = _setup()
    red = Detection("red", 0.9, 800, 500, 90, 90)
    still_there = Detection("red", 0.8, 803, 498, 90, 90)  # within 10 px
    det = FakeDetector([[red], [still_there], [still_there]])
    arm = FakeArm()
    loop = pick.PickLoop(det, arm, frame, H, max_cycles=10)
    assert loop.run() == 0
    assert loop.ignored == [(800, 500)]
    # released after the failed lift, and never descended to the drop zone
    assert ("suction", False) in arm.calls
    assert ("move", 150, -150, 70) not in arm.calls


def test_dry_run_and_once():
    H = _setup()
    red = Detection("red", 0.9, 800, 500, 90, 90)
    det = FakeDetector([[red], [red], [red]])
    arm = FakeArm()
    loop = pick.PickLoop(det, arm, frame, H, dry_run=True, once=True, max_cycles=10)
    assert loop.run() == 1
    assert det.calls == 1  # dry-run skips the lift check; --once stops after one pick


if __name__ == "__main__":
    import runlog
    runlog.start_run("test-pick")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("pick loop tests OK")
