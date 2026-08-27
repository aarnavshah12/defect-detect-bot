"""Render a vertical "rail" video (default 640x1080, black background) that stacks everything for the side of a
phone clip: title, what the overhead model saw (boxes from the log), metric cards, status, and the arm path.

    python rail_demo.py logs/20260826-224300.log --offset 23 --duration 55 --out Demo/rail.mp4

Drop it on the right third of a 1920x1080 canvas next to the phone clip (cropped to 1280x1080), aligned to the
clip's start. --offset is the same sync value as the other demo scripts.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import compose_demo as cd
import config
import hud
from panel_demo import arm_path_plot, card

REMOVED_FLASH_S = 1.6


def wrap(text, scale, thick, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if hud._text_w(trial, scale, thick)[0] > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="640x1080")
    ap.add_argument("--out", default="Demo/rail.mp4")
    ap.add_argument("--target", default=config.TARGET_CLASS)
    ap.add_argument("--title", default="AUTONOMOUS DEFECT DETECTION WITH PHYSICAL AI")
    ap.add_argument("--subtitle", default="RF-DETR on Roboflow  -  Hiwonder MaxArm")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.split("x"))
    M = 20  # margin

    ev = cd.parse_log(args.log)
    rep = cd.Replay(ev, args.target)
    view = cd.detection_extent(ev)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    n = int(args.duration * args.fps)
    total = None
    last_pick_t, picks_seen = -99.0, 0
    title_lines = wrap(args.title, 0.95, 2, W - 2 * M)
    for i in range(n):
        t_clip = i / args.fps
        t_log = t_clip + args.offset
        rep.advance(t_log)
        if total is None and rep.dets:
            total = rep.remaining
        if rep.picks > picks_seen:
            picks_seen, last_pick_t = rep.picks, t_clip
        img = np.zeros((H, W, 3), np.uint8)
        # 1. title
        y = 44
        for ln in title_lines:
            hud._text(img, ln, (M, y), 0.95, hud.WHITE, 2)
            y += 40
        cv2.line(img, (M, y - 26), (M + hud._text_w(title_lines[-1], 0.95, 2)[0], y - 26), hud.VIOLET, 3)
        hud._text(img, args.subtitle, (M, y + 2), 0.55, hud.GRAY, 1)
        y += 22
        # 2. what the model sees (16:9)
        sh = int((W - 2 * M) * 9 / 16) + 50
        sch = cd.schematic(rep, size=(W - 2 * M, sh), view=view)
        img[y:y + sh, M:W - M] = sch
        cv2.rectangle(img, (M, y), (W - M, y + sh), (60, 48, 40), 1)
        y += sh + 14
        # 3. cards 2x2
        cw, ch, gap = (W - 2 * M - 12) // 2, 104, 12
        left = rep.remaining
        card(img, M, y, cw, ch, f"{args.target.upper()}S LEFT", left, hud.RED if left else hud.GREEN,
             sub=f"of {total if total is not None else '-'} detected", big=1.6)
        card(img, M + cw + gap, y, cw, ch, "PICKED", rep.picks, hud.WHITE, sub=f"skipped {rep.skipped}  cycle {rep.cycle}", big=1.6)
        y += ch + gap
        last_move = next((d for tt, k, d in reversed(ev) if k == "move" and tt <= t_log), None)
        arm_txt = f"({last_move['x']:.0f}, {last_move['y']:.0f}, {last_move['z']:.0f})" if last_move else "-"
        card(img, M, y, cw, ch, "ARM TARGET  mm", arm_txt, hud.WHITE, sub="end-effector x, y, z", big=0.95)
        removed = 0 <= t_clip - last_pick_t < REMOVED_FLASH_S
        card(img, M + cw + gap, y, cw, ch, "REMOVED?", "YES!!" if removed else ("ALL CLEAR" if rep.done else "NO"),
             hud.GREEN if (removed or rep.done) else hud.GRAY, sub="verified by re-detection", big=1.5 if not rep.done else 1.0)
        y += ch + 14
        # 4. status chip + event
        state_txt = rep.state.upper()
        (sw, sth) = hud._text_w(state_txt, 0.8, 2)
        cv2.rectangle(img, (M, y), (M + sw + 36, y + sth + 22), hud.VIOLET, -1)
        hud._text(img, state_txt, (M + 18, y + sth + 8), 0.8, hud.WHITE, 2)
        hud._text(img, rep.event, (M + sw + 50, y + sth + 8), 0.55, hud.GRAY, 1)
        y += sth + 36
        # 5. arm path (rest of the space)
        ph = H - y - 40
        if ph > 120:
            arm_path_plot(img, M, y, W - 2 * M, ph, ev, t_log)
        hud._text(img, f"{int(max(0, t_log)):3d} s   frame {i:04d}", (M, H - 14), 0.5, hud.GRAY, 1)
        out.write(img)
    out.release()
    print(f"wrote {args.out} ({W}x{H}, {args.duration:.0f}s @ {args.fps:.0f} fps)")


if __name__ == "__main__":
    main()
