"""Offline test of the pick loop with a fake detector, a recording fake arm, and a synthetic homography."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arm as arm_mod  # noqa: E402
import config  # noqa: E402
import pick  # noqa: E402
from detect import Detection  # noqa: E402


class FakeDetector:
    def __init__(self, script):
        self.script = list(script)  # one list of detections per detect() call
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return self.script.pop(0) if self.script else []


class StaticDetector:
    def __init__(self, dets):
        self.dets, self.calls = dets, 0

    def detect(self, frame):
        self.calls += 1
        return list(self.dets)


class FakeArm:
    def __init__(self, refuse=()):
        self.calls = []
        self.refuse = set(refuse)

    def move_to(self, x, y, z, ms=None):
        t = (round(x), round(y), round(z))
        self.calls.append(("move",) + t)
        if t in self.refuse:
            raise arm_mod.MoveRefused(f"{t} refused")

    def suction(self, on):
        self.calls.append(("suction", on))

    def home(self, ms=None):
        self.calls.append(("home",))

    def rise(self):
        pass

    def wait(self, s):
        pass


def _setup():
    config.TARGET_CLASS = "red"  # tests are written around red blocks regardless of the owner's default
    config.TABLE_Z_MM = 60.0
    config.TRAVEL_Z_MM = 160.0
    config.DROP_XYZ_MM = (150.0, -150.0, 70.0)
    config.HOME_XYZ_MM = (0.0, -160.0, 210.0)
    config.REACH_X_MM = (-200.0, 200.0)
    config.REACH_Y_MM = (-280.0, -80.0)
    config.REACH_Z_MM = (60.0, 250.0)
    # pixel (px, py) -> arm: x = 0.25*px - 240, y = -0.25*py - 100   (pure scale/offset homography)
    return np.array([[0.25, 0, -240.0], [0, -0.25, -100.0], [0, 0, 1.0]])


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
    assert loop.run() == 2
    moves = [c for c in arm.calls if c[0] == "move"]
    assert moves[0] == ("move", 60, -250, 160)   # travel over red_b first (highest conf)
    assert moves[1] == ("move", 60, -250, 100)   # hover
    assert moves[2] == ("move", 60, -250, 60)    # descend to table Z
    assert ("suction", True) in arm.calls and ("suction", False) in arm.calls
    assert moves[3:6] == [("move", 60, -250, 100), ("move", 60, -250, 160), ("move", 150, -150, 160)]  # lift, travel
    assert moves[6] == ("move", 150, -150, 110)  # drop hover
    # every sideways move happens at travel height
    prev = None
    for m in moves:
        if prev is not None and (m[1], m[2]) != (prev[1], prev[2]):
            assert m[3] == 160 and prev[3] == 160, f"sideways move below travel height: {prev} -> {m}"
        prev = m
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
    assert not any(c[0] == "move" and c[1:3] == (235, -112) for c in arm.calls)


def test_lift_failed_releases_and_ignores():
    H = _setup()
    red = Detection("red", 0.9, 800, 500, 90, 90)
    still_there = Detection("red", 0.8, 803, 498, 90, 90)  # within 10 px
    det = FakeDetector([[red], [still_there], [still_there]])
    arm = FakeArm()
    loop = pick.PickLoop(det, arm, frame, H, max_cycles=10)
    assert loop.run() == 0
    assert loop.ignored == [(800, 500)]
    assert ("suction", False) in arm.calls
    assert ("move", 150, -150, 70) not in arm.calls  # never descended to the drop zone


def test_dry_run_plans_each_block_once_then_done():
    H = _setup()
    red_a = Detection("red", 0.6, 400, 400, 90, 90)
    red_b = Detection("red", 0.9, 1200, 600, 90, 90)
    det = StaticDetector([red_a, red_b])  # static table: nothing ever moves in dry-run
    arm = FakeArm()
    loop = pick.PickLoop(det, arm, frame, H, dry_run=True, max_cycles=10)
    assert loop.run() == 2
    assert det.calls == 3  # two planned picks (no lift check in dry-run), then "no red" -> done
    hovers = []
    for c in arm.calls:  # distinct pick-hover targets, in order (each pick hovers twice)
        if c[0] == "move" and c[3] == 100 and c[1:3] != (150, -150) and c not in hovers:
            hovers.append(c)
    assert hovers == [("move", 60, -250, 100), ("move", -140, -200, 100)]
    assert loop.planned == [(1200, 600), (400, 400)] and loop.ignored == []


def test_once_stops_after_one_pick():
    H = _setup()
    red = Detection("red", 0.9, 800, 500, 90, 90)
    det = StaticDetector([red])
    loop = pick.PickLoop(det, FakeArm(), frame, H, dry_run=True, once=True, max_cycles=10)
    assert loop.run() == 1 and det.calls == 1


def test_move_refused_at_pick_releases_ignores_and_continues():
    H = _setup()
    red = Detection("red", 0.9, 800, 500, 90, 90)  # -> (-40, -225)
    det = FakeDetector([[red], [red]])
    arm = FakeArm(refuse={(-40, -225, 60)})  # descent to table Z silently refused
    loop = pick.PickLoop(det, arm, frame, H, max_cycles=10)
    assert loop.run() == 0
    assert loop.ignored == [(800, 500)]
    assert ("suction", False) in arm.calls and arm.calls[-1] == ("home",)
    assert not any(c[0] == "move" and c[1:3] == (150, -150) for c in arm.calls)  # never carried on to the drop zone


def test_move_refused_at_drop_aborts_run():
    H = _setup()
    red = Detection("red", 0.9, 800, 500, 90, 90)
    det = FakeDetector([[red]])
    arm = FakeArm(refuse={(150, -150, 160)})  # drop travel pose unreachable for the arm's IK
    loop = pick.PickLoop(det, arm, frame, H, max_cycles=10)
    try:
        loop.run()
    except arm_mod.ArmError:
        pass
    else:
        raise AssertionError("expected ArmError for an unreachable drop zone")
    assert loop.picks == 0 and ("suction", False) in arm.calls and loop.ignored == [(800, 500)]


def test_any_class_picks_block_sized_detection_of_any_colour():
    H = _setup()
    yellow = Detection("yellow", 0.4, 800, 500, 90, 90)  # the blue block, mislabelled
    det = FakeDetector([[yellow], [], []])
    arm = FakeArm()
    loop = pick.PickLoop(det, arm, frame, H, target_class="any", max_cycles=10)
    assert loop.run() == 1


def test_fixed_targets_validated_before_any_motion():
    _setup()
    pick.check_fixed_targets()  # fine
    config.DROP_XYZ_MM = (150.0, -150.0, 240.0)  # drop hover 280 > REACH_Z max
    try:
        pick.check_fixed_targets()
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit for an out-of-envelope drop hover")
    finally:
        config.DROP_XYZ_MM = (150.0, -150.0, 70.0)


if __name__ == "__main__":
    import tempfile

    import runlog
    config.LOG_DIR = tempfile.mkdtemp()  # keep test logs out of logs/
    runlog.start_run("test-pick")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("pick loop tests OK")
