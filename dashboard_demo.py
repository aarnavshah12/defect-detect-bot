"""Dashboard-style demo video (one continuous shot + live analytics), in the style of the basketball demo.

    python dashboard_demo.py Demo/IMG_7199.MOV logs/20260826-224300.log --offset 23 --out Demo/dashboard.mp4

Layout (1920x1080): the phone clip fills the top-left with the defect model's boxes drawn ON it (the model
runs on the phone frames); a right rail shows live metrics from the run log (defects left, picked, the arm's
commanded target, status, and a REMOVED? card that flips green at each pick); the bottom strip shows what the
overhead camera saw (boxes from the log, the arm's commanded path), a table of the current detections, and a
frame counter. Detections on the phone frames are cached next to the clip so re-renders are fast.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

import compose_demo as cd
import config
import hud
import mapping

W, H = 1920, 1080
VID_W, VID_H = 1320, 742
RAIL_X = 1340
STRIP_Y = 762
PHONE_CONF = 0.5
REMOVED_FLASH_S = 1.6


def phone_detections(path: str, every: int, conf: float):
    """Run the model on every `every`-th frame of the phone clip; cache to <path>.dets.npz."""
    import detect

    cache = path + ".dets.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["frames"].tolist(), z["dets"].tolist()
    saved = (config.MIN_BOX_PX, config.MAX_BOX_PX)
    config.MIN_BOX_PX, config.MAX_BOX_PX = 40, 600  # side view: smaller / partly hidden blocks
    det = detect.Detector(roi=None, confidence=conf)
    cap = cv2.VideoCapture(path)
    frames, dets, i, t0 = [], [], 0, time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(i)
            dets.append([(d.cls, d.conf, d.cx, d.cy, d.w, d.h) for d in det.detect(f)])
            if len(frames) % 100 == 0:
                print(f"  detections: frame {i} ({time.time() - t0:.0f}s)")
        i += 1
    cap.release()
    config.MIN_BOX_PX, config.MAX_BOX_PX = saved
    np.savez(cache, frames=np.array(frames), dets=np.array(dets, dtype=object))
    return frames, dets


def card(img, x, y, w, h, label, value, colour=hud.WHITE, sub="", big=1.9):
    cv2.rectangle(img, (x, y), (x + w, y + h), (44, 34, 28), -1)
    cv2.line(img, (x, y), (x, y + h), hud.VIOLET, 3)
    hud._text(img, label, (x + 22, y + 34), 0.55, hud.GRAY, 1)
    hud._text(img, str(value), (x + 22, y + 34 + int(46 * big)), big, colour, 2)
    if sub:
        hud._text(img, sub, (x + 22, y + h - 16), 0.5, hud.GRAY, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phone")
    ap.add_argument("log")
    ap.add_argument("--offset", type=float, required=True, help="phone clip start, seconds after the log's first line")
    ap.add_argument("--out", default="Demo/dashboard.mp4")
    ap.add_argument("--every", type=int, default=2, help="run the model on every Nth phone frame")
    ap.add_argument("--target", default=config.TARGET_CLASS)
    ap.add_argument("--title", default="AUTONOMOUS DEFECT DETECTION WITH PHYSICAL AI")
    args = ap.parse_args()

    ev = cd.parse_log(args.log)
    rep = cd.Replay(ev, args.target)
    view = cd.detection_extent(ev)
    try:
        Hinv = np.linalg.inv(mapping.load_homography())
    except Exception:  # noqa: BLE001
        Hinv = None
    frames_idx, frames_dets = phone_detections(args.phone, args.every, PHONE_CONF)
    total_defects = None

    cap = cv2.VideoCapture(args.phone)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    sx, sy = VID_W / 1920, VID_H / 1080
    last_pick_t = -99.0
    picks_seen = 0
    path_pts: list[tuple[float, float]] = []
    i = 0
    j = 0
    t_render = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_log = i / fps + args.offset
        rep.advance(t_log)
        if total_defects is None and rep.dets:
            total_defects = rep.remaining
        if rep.picks > picks_seen:
            picks_seen = rep.picks
            last_pick_t = i / fps
        # commanded arm path (last few moves) for the schematic
        path_pts = [(d["x"], d["y"]) for tt, k, d in ev if k == "move" and tt <= t_log][-6:]

        canvas = np.full((H, W, 3), hud.INK, np.uint8)
        # ---- main video with the model's boxes on it
        vid = cv2.resize(frame, (VID_W, VID_H))
        while j + 1 < len(frames_idx) and frames_idx[j + 1] <= i:
            j += 1
        dets = frames_dets[j] if frames_dets else []
        target_box = None
        if rep.target is not None:
            cands = [d for d in dets if d[0] == args.target]
            if cands:
                target_box = max(cands, key=lambda d: d[1])
        for d in dets:
            cls, conf, cx, cy, w, h = d
            x1, y1, x2, y2 = int((cx - w / 2) * sx), int((cy - h / 2) * sy), int((cx + w / 2) * sx), int((cy + h / 2) * sy)
            is_t = cls == args.target
            if d is target_box:
                hud.box(vid, x1, y1, x2, y2, f"{cls} {conf:.2f}", hud.VIOLET, 3)
                hud.brackets(vid, x1 - 8, y1 - 8, x2 + 8, y2 + 8, hud.VIOLET)
            else:
                hud.box(vid, x1, y1, x2, y2, f"{cls} {conf:.2f}", hud.RED if is_t else hud.GREEN, 2 if is_t else 1)
        if 0 <= i / fps - last_pick_t < REMOVED_FLASH_S:
            hud.banner(vid, "DEFECT REMOVED", hud.GREEN)
        elif rep.done:
            hud.banner(vid, f"ALL CLEAR  -  {rep.picks} DEFECTS REMOVED", hud.GREEN)
        canvas[0:VID_H, 0:VID_W] = vid
        hud._text(canvas, args.title, (18, VID_H + 6 + 22), 0.62, hud.GRAY, 1)

        # ---- right rail
        x, w = RAIL_X, W - RAIL_X - 16
        left = rep.remaining
        card(canvas, x, 16, w, 118, f"{args.target.upper()}S LEFT", left, hud.RED if left else hud.GREEN,
             sub=f"of {total_defects if total_defects is not None else '-'} detected")
        card(canvas, x, 150, w, 118, "PICKED", rep.picks, hud.WHITE, sub=f"skipped {rep.skipped}")
        last_move = next((d for tt, k, d in reversed(ev) if k == "move" and tt <= t_log), None)
        arm_txt = f"({last_move['x']:.0f}, {last_move['y']:.0f}, {last_move['z']:.0f})" if last_move else "-"
        card(canvas, x, 284, w, 118, "ARM TARGET  mm", arm_txt, hud.WHITE, sub="end-effector x, y, z", big=1.25)
        card(canvas, x, 418, w, 118, "STATUS", rep.state.upper(), hud.VIOLET, sub=f"cycle {rep.cycle}   {rep.event}", big=1.1)
        removed = 0 <= i / fps - last_pick_t < REMOVED_FLASH_S
        card(canvas, x, 552, w, 174, "REMOVED?", "YES!!" if removed else ("ALL CLEAR" if rep.done else "NO"),
             hud.GREEN if (removed or rep.done) else hud.GRAY, sub="suction pick verified by re-detection", big=2.4)

        # ---- bottom strip
        cv2.line(canvas, (0, STRIP_Y - 8), (W, STRIP_Y - 8), hud.VIOLET, 2)
        strip_h = H - STRIP_Y
        sch = cd.schematic(rep, size=(560, strip_h), view=view)
        pos, cur_path, prev_paths = cd.arm_trajectory(ev, t_log)
        if Hinv is not None and pos is not None and view:
            x0, y0, x1, y1 = view
            s = min((560 - 24) / (x1 - x0), (strip_h - 70) / (y1 - y0))
            ox = int((560 - (x1 - x0) * s) / 2)
            oy = 50 + int((strip_h - 70 - (y1 - y0) * s) / 2)

            def to_px(path):
                if not path:
                    return []
                pts = cv2.perspectiveTransform(np.array([[[px, py]] for px, py in path], dtype=np.float64), Hinv).reshape(-1, 2)
                return [(int((px - x0) * s) + ox, int((py - y0) * s) + oy) for px, py in pts]

            for path in prev_paths:
                pp = to_px(path)
                for a, b in zip(pp[:-1], pp[1:]):
                    cv2.line(sch, a, b, (90, 74, 60), 1, cv2.LINE_AA)
            pp = to_px(cur_path)
            for a, b in zip(pp[:-1], pp[1:]):
                cv2.line(sch, a, b, hud.AMBER, 2, cv2.LINE_AA)
            cp = to_px([(pos[0], pos[1])])[0]
            if pp:
                cv2.line(sch, pp[-2] if len(pp) > 1 else pp[-1], cp, hud.AMBER, 2, cv2.LINE_AA)
            cv2.circle(sch, cp, 8, hud.WHITE, -1, cv2.LINE_AA)
            cv2.circle(sch, cp, 8, hud.AMBER, 2, cv2.LINE_AA)
            hud._text(sch, f"arm z{pos[2]:.0f}", (cp[0] + 12, cp[1] + 6), 0.5, hud.AMBER, 1)
        canvas[STRIP_Y:H, 0:560] = sch
        # table
        tx = 590
        hud._text(canvas, "overhead detections", (tx, STRIP_Y + 34), 0.62, hud.GRAY, 1)
        hud._text(canvas, "class        conf     arm x, y (mm)      status", (tx, STRIP_Y + 64), 0.5, hud.GRAY, 1)
        rows = sorted(rep.dets, key=lambda d: (d["cls"] != args.target, -d["conf"]))[:6]
        for r, d in enumerate(rows):
            arm_xy = "-"
            if Hinv is not None:
                # map the logged pixel through the current calibration for display
                try:
                    ax, ay = mapping.pixel_to_arm(d["cx"], d["cy"], np.linalg.inv(Hinv))
                    arm_xy = f"{ax:6.0f}, {ay:5.0f}"
                except Exception:  # noqa: BLE001
                    pass
            is_tgt = rep.target is not None and abs(d["cx"] - rep.target["cx"]) < 20 and abs(d["cy"] - rep.target["cy"]) < 20
            status = "PICKING" if is_tgt else ("queued" if d["cls"] == args.target else "leave")
            colour = hud.VIOLET if is_tgt else (hud.RED if d["cls"] == args.target else hud.GREEN)
            hud._text(canvas, f"{d['cls']:<10s}  {d['conf']:.2f}     {arm_xy:<16s}  {status}", (tx, STRIP_Y + 96 + r * 30), 0.55, colour, 1)
        # frame counter / footer
        hud._text(canvas, f"FRAME {i:04d}", (W - 210, STRIP_Y + 34), 0.7, hud.GRAY, 1)
        hud._text(canvas, f"{int(max(0, t_log)):3d} s", (W - 210, STRIP_Y + 64), 0.55, hud.GRAY, 1)
        hud._text(canvas, "RF-DETR  ·  Roboflow  ·  Hiwonder MaxArm", (W - 400, H - 20), 0.5, hud.GRAY, 1)

        out.write(canvas)
        i += 1
        if i % 300 == 0:
            print(f"  render {i}/{n} ({time.time() - t_render:.0f}s)")
    cap.release()
    out.release()
    print(f"wrote {args.out}: {i} frames, {i / fps:.1f}s")


if __name__ == "__main__":
    main()
