"""Demo overlay drawing for pick.py: Roboflow-style boxes with label tabs, a status bar, a DONE banner.

Pure drawing helpers on BGR frames (OpenCV). Colours are the Roboflow palette in BGR.
"""

from __future__ import annotations

import cv2
import numpy as np

VIOLET = (237, 58, 124)      # #7C3AED
VIOLET_DARK = (182, 33, 91)  # #5B21B6
RED = (68, 68, 239)          # #EF4444
GREEN = (129, 185, 16)       # #10B981
AMBER = (11, 158, 245)       # #F59E0B
WHITE = (250, 250, 249)
GRAY = (128, 114, 107)       # #6B7280
INK = (39, 24, 17)           # #111827

FONT = cv2.FONT_HERSHEY_DUPLEX


def _text(img, s, org, scale, colour, thick=1):
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _text_w(s, scale, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0]


def box(img, x1, y1, x2, y2, label, colour, thick=2, tab=True):
    """Rectangle with a filled label tab on the top-left (Roboflow bounding-box motif)."""
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick)
    if tab and label:
        (tw, th) = _text_w(label, 0.6, 1)
        y_tab = y1 - th - 10 if y1 - th - 10 > 0 else y1
        cv2.rectangle(img, (x1 - thick // 2, y_tab), (x1 + tw + 12, y_tab + th + 10), colour, -1)
        _text(img, label, (x1 + 6, y_tab + th + 3), 0.6, WHITE, 1)


def brackets(img, x1, y1, x2, y2, colour, length=26, thick=4):
    """Corner brackets around a box (the "target" look)."""
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), colour, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), colour, thick, cv2.LINE_AA)


def translucent(img, x1, y1, x2, y2, colour=INK, alpha=0.72):
    roi = img[y1:y2, x1:x2]
    if roi.size:
        cv2.addWeighted(np.full_like(roi, colour, dtype=np.uint8), alpha, roi, 1 - alpha, 0, roi)


def status_bar(img, *, target_class: str, left: int, picked: int, skipped: int, others: int,
               state: str, cycle: int, seconds: int, event: str, dry_run: bool = False):
    """Top status bar: big numbers, current action, last event."""
    h, w = img.shape[:2]
    bar_h = 118
    translucent(img, 0, 0, w, bar_h)
    cv2.line(img, (0, bar_h), (w, bar_h), VIOLET, 3)
    # big numbers
    x = 24
    for label, value, colour in ((f"{target_class.upper()} LEFT", left, RED if left else GREEN),
                                 ("PICKED", picked, WHITE), ("SKIPPED", skipped, GRAY), ("OTHER", others, GRAY)):
        _text(img, label, (x, 34), 0.55, GRAY, 1)
        _text(img, str(value), (x, 88), 1.7, colour, 2)
        x += max(_text_w(label, 0.55)[0], _text_w(str(value), 1.7, 2)[0]) + 44
    # state (right side)
    state_txt = ("DRY RUN  " if dry_run else "") + state.upper()
    (sw, sh) = _text_w(state_txt, 0.95, 2)
    cv2.rectangle(img, (w - sw - 48, 22), (w - 16, 22 + sh + 22), VIOLET, -1)
    _text(img, state_txt, (w - sw - 32, 22 + sh + 6), 0.95, WHITE, 2)
    meta = f"cycle {cycle}   {seconds}s   {event}"
    (mw, _) = _text_w(meta, 0.55)
    _text(img, meta, (w - mw - 20, 104), 0.55, GRAY, 1)


def banner(img, text, colour=GREEN):
    h, w = img.shape[:2]
    (tw, th) = _text_w(text, 1.6, 3)
    x1, y1 = (w - tw) // 2 - 40, h // 2 - th
    translucent(img, x1, y1, x1 + tw + 80, y1 + th * 2 + 20, alpha=0.8)
    cv2.rectangle(img, (x1, y1), (x1 + tw + 80, y1 + th * 2 + 20), colour, 3)
    _text(img, text, (x1 + 40, y1 + th + th // 2 + 4), 1.6, colour, 3)


def countdown(img, seconds_left: int):
    h, w = img.shape[:2]
    txt = f"starting in {seconds_left}"
    (tw, th) = _text_w(txt, 2.2, 4)
    _text(img, txt, ((w - tw) // 2, h // 2 + th // 2), 2.2, INK, 10)
    _text(img, txt, ((w - tw) // 2, h // 2 + th // 2), 2.2, WHITE, 4)
