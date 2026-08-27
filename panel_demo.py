"""Render a stand-alone metrics panel video (black background) timed to a run log, for compositing in an editor.

    python panel_demo.py logs/20260826-224300.log --offset 23 --duration 55 --out Demo/panel.mp4

Place it over the empty area of your composite, aligned to the phone clip's start; --offset is the same sync
value used by compose_demo.py / dashboard_demo.py (seconds between the log's first line and the clip start).
Also writes a still title card (<out>.title.png).
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

import compose_demo as cd
import config
import hud

REMOVED_FLASH_S = 1.6


def arm_path_plot(img, x, y, w, h, moves, t_log):
    """Top view in arm coordinates: bin, home, pick area, commanded path (last moves) and the current target."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (44, 34, 28), -1)
    hud._text(img, "arm path (top view, mm)", (x + 14, y + 26), 0.5, hud.GRAY, 1)
    xs = [config.DROP_XYZ_MM[0], config.HOME_XYZ_MM[0], config.PICK_AREA_X_MM[0], config.PICK_AREA_X_MM[1]]
    ys = [config.DROP_XYZ_MM[1], config.HOME_XYZ_MM[1], config.PICK_AREA_Y_MM[0], config.PICK_AREA_Y_MM[1]]
    x0, x1, y0, y1 = min(xs) - 30, max(xs) + 30, min(ys) - 30, max(ys) + 30
    s = min((w - 30) / (x1 - x0), (h - 50) / (y1 - y0))

    def P(ax, ay):  # arm +x -> right, arm -y (forward, away from base) -> up
        return int(x + 15 + (ax - x0) * s), int(y + h - 12 - (ay - y0) * s)

    a, b = P(config.PICK_AREA_X_MM[0], config.PICK_AREA_Y_MM[0]), P(config.PICK_AREA_X_MM[1], config.PICK_AREA_Y_MM[1])
    cv2.rectangle(img, a, b, (80, 64, 56), 1)
    cv2.circle(img, P(0, 0), 7, hud.GRAY, 1)
    hud._text(img, "base", (P(0, 0)[0] + 10, P(0, 0)[1] + 5), 0.42, hud.GRAY, 1)
    cv2.rectangle(img, (P(config.DROP_XYZ_MM[0], config.DROP_XYZ_MM[1])[0] - 12, P(config.DROP_XYZ_MM[0], config.DROP_XYZ_MM[1])[1] - 12),
                  (P(config.DROP_XYZ_MM[0], config.DROP_XYZ_MM[1])[0] + 12, P(config.DROP_XYZ_MM[0], config.DROP_XYZ_MM[1])[1] + 12), hud.GREEN, 2)
    hud._text(img, "bin", (P(config.DROP_XYZ_MM[0], config.DROP_XYZ_MM[1])[0] + 16, P(config.DROP_XYZ_MM[0], config.DROP_XYZ_MM[1])[1] + 5), 0.42, hud.GREEN, 1)
    hp = P(config.HOME_XYZ_MM[0], config.HOME_XYZ_MM[1])
    cv2.drawMarker(img, hp, hud.GRAY, cv2.MARKER_DIAMOND, 14, 1)
    hud._text(img, "home", (hp[0] + 10, hp[1] + 5), 0.42, hud.GRAY, 1)
    recent = [(tt, d) for tt, d in moves if tt <= t_log][-8:]
    pts = [P(d["x"], d["y"]) for tt, d in recent]
    for i, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
        cv2.line(img, p1, p2, hud.AMBER, 2, cv2.LINE_AA)
    if pts:
        cv2.circle(img, pts[-1], 7, hud.AMBER, -1, cv2.LINE_AA)


def card(img, x, y, w, h, label, value, colour=hud.WHITE, sub="", big=1.9):
    cv2.rectangle(img, (x, y), (x + w, y + h), (44, 34, 28), -1)
    cv2.line(img, (x, y), (x, y + h), hud.VIOLET, 3)
    hud._text(img, label, (x + 20, y + 30), 0.55, hud.GRAY, 1)
    (vw, vh) = hud._text_w(str(value), big, 2)
    hud._text(img, str(value), (x + 20, y + 38 + vh), big, colour, 2)  # value sits just under the label
    if sub:
        hud._text(img, sub, (x + 20, y + h - 12), 0.46, hud.GRAY, 1)


def title_card(w, h, title, subtitle):
    img = np.zeros((h, w, 3), np.uint8)
    hud._text(img, title, (40, 90), 1.6, hud.WHITE, 3)
    cv2.line(img, (40, 110), (40 + hud._text_w(title, 1.6, 3)[0], 110), hud.VIOLET, 4)
    hud._text(img, subtitle, (40, 160), 0.7, hud.GRAY, 1)
    steps = ["webcam", "RF-DETR (Roboflow)", "homography", "MaxArm", "suction", "bin"]
    x, y = 40, 210
    for i, s in enumerate(steps):
        (tw, th) = hud._text_w(s, 0.62, 1)
        if x + tw + 60 > w:  # wrap
            x, y = 40, y + th + 40
        cv2.rectangle(img, (x, y), (x + tw + 24, y + th + 20), hud.VIOLET if i == 1 else (60, 48, 40), -1)
        hud._text(img, s, (x + 12, y + th + 10), 0.62, hud.WHITE, 1)
        x += tw + 24
        if i < len(steps) - 1:
            hud._text(img, ">", (x + 8, y + th + 10), 0.62, hud.GRAY, 1)
            x += 30
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--duration", type=float, required=True, help="seconds (match the phone clip)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="960x540", help="panel size WxH")
    ap.add_argument("--out", default="Demo/panel.mp4")
    ap.add_argument("--target", default=config.TARGET_CLASS)
    ap.add_argument("--title", default="AUTONOMOUS DEFECT SORTING")
    ap.add_argument("--subtitle", default="RF-DETR on Roboflow  -  Hiwonder MaxArm  -  every number from the run log")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.split("x"))

    ev = cd.parse_log(args.log)
    rep = cd.Replay(ev, args.target)
    moves = [(tt, d) for tt, k, d in ev if k == "move"]
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    cv2.imwrite(args.out + ".title.png", title_card(W, H, args.title, args.subtitle))
    n = int(args.duration * args.fps)
    total = None
    last_pick_t, picks_seen = -99.0, 0
    for i in range(n):
        t_clip = i / args.fps
        t_log = t_clip + args.offset
        rep.advance(t_log)
        if total is None and rep.dets:
            total = rep.remaining
        if rep.picks > picks_seen:
            picks_seen, last_pick_t = rep.picks, t_clip
        img = np.zeros((H, W, 3), np.uint8)
        hud._text(img, args.title, (20, 36), 0.75, hud.WHITE, 2)
        cv2.line(img, (20, 48), (20 + hud._text_w(args.title, 0.75, 2)[0], 48), hud.VIOLET, 3)
        # cards: 2 columns
        cw, ch, gap = (W - 60) // 2, 120, 12
        left = rep.remaining
        card(img, 20, 66, cw, ch, f"{args.target.upper()}S LEFT", left, hud.RED if left else hud.GREEN,
             sub=f"of {total if total is not None else '-'} detected")
        card(img, 20 + cw + gap, 66, cw, ch, "PICKED", rep.picks, hud.WHITE, sub=f"skipped {rep.skipped}   cycle {rep.cycle}")
        last_move = next((d for tt, d in reversed(moves) if tt <= t_log), None)
        arm_txt = f"({last_move['x']:.0f}, {last_move['y']:.0f}, {last_move['z']:.0f})" if last_move else "-"
        card(img, 20, 66 + ch + gap, cw, ch, "ARM TARGET  mm", arm_txt, hud.WHITE, sub="end-effector x, y, z", big=1.2)
        removed = 0 <= t_clip - last_pick_t < REMOVED_FLASH_S
        card(img, 20 + cw + gap, 66 + ch + gap, cw, ch, "REMOVED?", "YES!!" if removed else ("ALL CLEAR" if rep.done else "NO"),
             hud.GREEN if (removed or rep.done) else hud.GRAY, sub="verified by re-detection", big=1.9)
        # status chip + path plot
        y2 = 66 + 2 * (ch + gap)
        state_txt = rep.state.upper()
        (sw, sh) = hud._text_w(state_txt, 0.9, 2)
        cv2.rectangle(img, (20, y2), (20 + sw + 40, y2 + sh + 26), hud.VIOLET, -1)
        hud._text(img, state_txt, (40, y2 + sh + 10), 0.9, hud.WHITE, 2)
        hud._text(img, f"{rep.event}", (20, y2 + sh + 58), 0.55, hud.GRAY, 1)
        hud._text(img, f"{int(max(0, t_log)):3d} s   frame {i:04d}", (20, H - 16), 0.5, hud.GRAY, 1)
        arm_path_plot(img, 20 + cw + gap, y2, cw, H - y2 - 20, moves, t_log)
        out.write(img)
    out.release()
    print(f"wrote {args.out} ({W}x{H}, {args.duration:.0f}s @ {args.fps:.0f} fps) and {args.out}.title.png")


if __name__ == "__main__":
    main()
