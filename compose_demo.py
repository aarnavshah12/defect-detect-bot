"""Compose a demo video: a phone clip of the arm + an overlay panel rebuilt from the run's log.

    python compose_demo.py Demo/IMG_7199.MOV logs/20260826-224300.log --offset 25 --out Demo/composite.mp4

--offset  seconds between the log's first line and the start of the phone clip (phone_time = log_time - offset).
The right-hand panel is a schematic of what the model saw at each moment (boxes from the log, the target framed),
plus the status bar; it is exact to the log's timestamps, so it stays in sync with the arm for the whole clip.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import cv2
import numpy as np

import config
import hud

W, H = 1920, 1080
LEFT_W = 1280  # phone clip 1280x720 top-left; schematic 640x720 top-right; status strip below
TOP_H = 720


def parse_log(path: str):
    DET = re.compile(r"det (cycle\d+|lift-check) (\S+) conf=([\d.]+) px=\((\d+),(\d+)\) size=\((\d+)x(\d+)\) -> arm=\(([-\d.]+), ([-\d.]+)\)")
    TGT = re.compile(r"target (\S+) conf=([\d.]+) px=\((\d+),(\d+)\)")
    MOV = re.compile(r"arm: move_to\(([-\d.]+), ([-\d.]+), ([-\d.]+)\) (\d+) ms")
    KINDS = [("suction_on", "suction ON"), ("vent", "arm: vent"), ("home", "arm: home"), ("pick_done", r"pick \d+ complete"),
             ("done", "-> done"), ("skip", "WARNING skip"), ("lift_failed", "lift failed"), ("scan", r"INFO cycle\d+: \d+ detections"),
             ("lift_check", "lift-check: ")]

    def ts(line):
        h, m, s = line[:12].split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    ev, t0 = [], None
    for line in open(path):
        if not re.match(r"\d\d:\d\d:\d\d", line):
            continue
        t = ts(line)
        t0 = t if t0 is None else t0
        t -= t0
        m = DET.search(line)
        if m:
            ev.append((t, "det", dict(tag=m.group(1), cls=m.group(2), conf=float(m.group(3)), cx=int(m.group(4)), cy=int(m.group(5)),
                                      w=int(m.group(6)), h=int(m.group(7)))))
            continue
        m = TGT.search(line)
        if m:
            ev.append((t, "target", dict(cls=m.group(1), conf=float(m.group(2)), cx=int(m.group(3)), cy=int(m.group(4)))))
            continue
        m = MOV.search(line)
        if m:
            ev.append((t, "move", dict(x=float(m.group(1)), y=float(m.group(2)), z=float(m.group(3)), ms=int(m.group(4)))))
            continue
        for kind, pat in KINDS:
            if re.search(pat, line):
                ev.append((t, kind, {}))
                break
    return ev


class Replay:
    """Walks the events and exposes the overlay state at any log time."""

    def __init__(self, events, target_class: str):
        self.ev = events
        self.target_class = target_class
        self.reset()

    def reset(self):
        self.i = 0
        self.dets: list[dict] = []
        self.pending: list[dict] = []
        self.target = None
        self.state = "starting"
        self.picks = 0
        self.skipped = 0
        self.cycle = 0
        self.event = ""
        self.done = False
        self.carrying = False
        self.remaining = 0

    def _finish_scan(self):
        if self.pending:
            self.dets = self.pending
            self.pending = []
            self.remaining = sum(1 for d in self.dets if d["cls"] == self.target_class or self.target_class == "any")

    def advance(self, t: float):
        while self.i < len(self.ev) and self.ev[self.i][0] <= t:
            _, k, d = self.ev[self.i]
            self.i += 1
            if k == "det":
                if d["tag"] != "lift-check":
                    self.pending.append(d)
            elif k == "scan":
                self._finish_scan()
                self.cycle += 1
                self.state = "scanning"
            elif k == "target":
                self._finish_scan()
                self.target = d
                self.state = "moving to block"
                self.event = f"target {d['cls']} {d['conf']:.2f}"
            elif k == "move":
                if self.target is not None and not self.carrying:
                    self.state = "descending" if d["z"] < config.TRAVEL_Z_MM else "moving to block"
                elif self.carrying:
                    self.state = "carrying to bin" if abs(d["x"] - config.DROP_XYZ_MM[0]) < 30 else "lifting"
            elif k == "suction_on":
                self.state = "suction on"
                self.carrying = True
            elif k == "lift_check":
                self.state = "checking the lift"
            elif k == "vent":
                self.state = "dropping"
                self.carrying = False
            elif k == "home":
                self.state = "homing"
            elif k == "pick_done":
                self.picks += 1
                self.remaining = max(0, self.remaining - 1)
                self.event = f"pick {self.picks} done"
                self.target = None
                self.state = "rescanning"
            elif k == "skip":
                self.skipped += 1
                self.event = "block skipped"
                self.target = None
            elif k == "lift_failed":
                self.event = "lift failed - skipped"
            elif k == "done":
                self._finish_scan()
                self.done = True
                self.state = "done"


def schematic(rep: Replay, size=(640, 720), view=None):
    """Dark panel with the logged boxes drawn at their pixel positions, zoomed to `view` (x0, y0, x1, y1 in frame px)."""
    w, h = size
    img = np.full((h, w, 3), hud.INK, np.uint8)
    hud._text(img, "what the model sees", (18, 34), 0.62, hud.GRAY, 1)
    x0, y0, x1, y1 = view if view else (0, 0, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    s = min((w - 24) / (x1 - x0), (h - 70) / (y1 - y0))
    ox = int((w - (x1 - x0) * s) / 2)
    oy = 50 + int((h - 70 - (y1 - y0) * s) / 2)

    def P(px, py):
        return int((px - x0) * s) + ox, int((py - y0) * s) + oy

    for d in rep.dets:
        a = P(d["cx"] - d["w"] / 2, d["cy"] - d["h"] / 2)
        b = P(d["cx"] + d["w"] / 2, d["cy"] + d["h"] / 2)
        is_t = d["cls"] == rep.target_class
        is_target = rep.target is not None and abs(d["cx"] - rep.target["cx"]) < 20 and abs(d["cy"] - rep.target["cy"]) < 20
        colour = hud.VIOLET if is_target else (hud.RED if is_t else hud.GREEN)
        hud.box(img, a[0], a[1], b[0], b[1], f"{d['cls']} {d['conf']:.2f}", colour, 3 if is_target else (2 if is_t else 1))
        if is_target:
            hud.brackets(img, a[0] - 6, a[1] - 6, b[0] + 6, b[1] + 6, hud.VIOLET, length=22, thick=3)
    return img


def detection_extent(events, margin=90):
    """Bounding box (frame px) of every logged detection, so the schematic can zoom to the action."""
    xs, ys = [], []
    for _, k, d in events:
        if k == "det":
            xs += [d["cx"] - d["w"] / 2, d["cx"] + d["w"] / 2]
            ys += [d["cy"] - d["h"] / 2, d["cy"] + d["h"] / 2]
    if not xs:
        return None
    return (max(0, min(xs) - margin), max(0, min(ys) - margin),
            min(config.FRAME_WIDTH, max(xs) + margin), min(config.FRAME_HEIGHT, max(ys) + margin))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phone")
    ap.add_argument("log")
    ap.add_argument("--offset", type=float, required=True, help="phone clip start, seconds after the log's first line")
    ap.add_argument("--out", default="Demo/composite.mp4")
    ap.add_argument("--title", default="Autonomous defect sorting  -  RF-DETR on Roboflow + Hiwonder MaxArm")
    ap.add_argument("--target", default=config.TARGET_CLASS)
    args = ap.parse_args()

    ev = parse_log(args.log)
    rep = Replay(ev, args.target)
    view = detection_extent(ev)
    cap = cv2.VideoCapture(args.phone)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    t_start = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_log = i / fps + args.offset
        rep.advance(t_log)
        canvas = np.full((H, W, 3), hud.INK, np.uint8)
        canvas[0:TOP_H, 0:LEFT_W] = cv2.resize(frame, (LEFT_W, TOP_H))
        canvas[0:TOP_H, LEFT_W:W] = schematic(rep, view=view)
        # status strip
        strip = canvas[TOP_H:H, :]
        cv2.line(canvas, (0, TOP_H), (W, TOP_H), hud.VIOLET, 3)
        hud._text(canvas, args.title, (24, TOP_H + 48), 0.9, hud.WHITE, 1)
        x = 24
        for label, value, colour in ((f"{args.target.upper()} LEFT", rep.remaining, hud.RED if rep.remaining else hud.GREEN),
                                     ("PICKED", rep.picks, hud.WHITE), ("SKIPPED", rep.skipped, hud.GRAY), ("CYCLE", rep.cycle, hud.GRAY)):
            hud._text(canvas, label, (x, TOP_H + 100), 0.6, hud.GRAY, 1)
            hud._text(canvas, str(value), (x, TOP_H + 175), 2.4, colour, 3)
            x += max(hud._text_w(label, 0.6)[0], hud._text_w(str(value), 2.4, 3)[0]) + 70
        state_txt = rep.state.upper()
        (sw, sh) = hud._text_w(state_txt, 1.3, 2)
        cv2.rectangle(canvas, (W - sw - 72, TOP_H + 92), (W - 24, TOP_H + 92 + sh + 34), hud.VIOLET, -1)
        hud._text(canvas, state_txt, (W - sw - 48, TOP_H + 92 + sh + 10), 1.3, hud.WHITE, 2)
        hud._text(canvas, f"{int(max(0, t_log)):3d} s   {rep.event}", (W - 520, TOP_H + 200), 0.7, hud.GRAY, 1)
        hud._text(canvas, "every detection and arm command replayed from the run log", (24, H - 24), 0.55, hud.GRAY, 1)
        if rep.done:
            hud.banner(canvas[0:TOP_H, 0:LEFT_W], f"DONE  -  {rep.picks} {args.target.upper()} REMOVED")
        out.write(canvas)
        i += 1
        if i % 300 == 0:
            print(f"  {i}/{n} frames ({i / fps:.0f}s)  state={rep.state} picked={rep.picks}")
    cap.release()
    out.release()
    print(f"wrote {args.out}: {i} frames, {i / fps:.1f}s, in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
