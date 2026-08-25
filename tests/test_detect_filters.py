import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import detect  # noqa: E402
from detect import Detection  # noqa: E402


def test_block_sized_drops_hands_and_laptops():
    hand = Detection("blue", 0.87, 603, 228, 636, 451)
    block = Detection("yellow", 0.39, 1242, 50, 136, 101)
    key = Detection("blue", 0.5, 300, 300, 20, 20)
    assert detect.block_sized([hand, block, key]) == [block]


def test_is_target_any_vs_named():
    d = Detection("yellow", 0.5, 0, 0, 100, 100)
    assert detect.is_target(d, "any") and not detect.is_target(d, "blue") and detect.is_target(d, "yellow")


def test_dedupe_keeps_top_confidence():
    a = Detection("blue", 0.55, 445, 384, 178, 191)
    b = Detection("green", 0.50, 445, 384, 178, 191)
    c = Detection("red", 0.9, 900, 300, 100, 100)
    assert detect.dedupe([b, a, c]) == [c, a]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("detect filter tests OK")
