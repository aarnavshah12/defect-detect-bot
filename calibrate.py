"""Click-to-calibrate the pixel -> arm homography, plus verify and check modes.

  python calibrate.py            collect: put a BLOCK on each tape mark, click it on the live feed (the
                                 click snaps to the detected block's bbox centre - the same quantity
                                 pick.py maps), jog the arm's suction cup onto the mark, type the arm
                                 (x, y) mm in the terminal. Writes calibration.npy.
  python calibrate.py --auto --carry  MOST ACCURATE: you centre the block under the cup once; the arm then
                                 carries it to every grid spot itself, sets it down, lets the camera look,
                                 and picks it back up. Hands-off.
  python calibrate.py --auto     EASIER: the arm drives itself to a grid of spots; at each one you slide
                                 the block under the cup and press Enter; the arm moves away and the
                                 camera records the block. No jogging, no typing. Writes calibration.npy.
  python calibrate.py --verify   verify: click anywhere -> the arm hovers above that point at
                                 TABLE_Z + HOVER_OFFSET. Press b -> hover above the best-detected block's
                                 bbox centre. Moves the arm -> asks "Workspace clear?".
  python calibrate.py --check    offline: matrix, leave-one-out residuals, sample mappings.

Collect-mode keys (feed window): u = undo, d = done (fit + save), q = quit without saving.
Terminal input after each click:  "x,y"  (arm mm)  |  u = discard this click  |  d = done  |  q = quit
Use 5-6 marks spread over the whole pick area (not in a line, one near the edge). With only 4 the fit is
exact and a typo cannot be detected.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading

import cv2
import numpy as np

import config
import mapping
import runlog

WINDOW = "calibrate"
SNAP_RADIUS_PX = 80.0  # a click within this distance of a detected block snaps to its bbox centre


def _stdin_reader(q: "queue.Queue[str]") -> None:
    for line in sys.stdin:
        q.put(line.strip())


def _nearest_detection(dets, x: float, y: float):
    best, best_d = None, SNAP_RADIUS_PX
    for d in dets:
        dist = math.hypot(d.cx - x, d.cy - y)
        if dist <= best_d:
            best, best_d = d, dist
    return best


def _draw_pairs(frame, pixel_pts, arm_pts, pending, dets):
    import detect

    vis = detect.draw(frame, dets, roi=None) if dets else frame.copy()
    for i, (p, a) in enumerate(zip(pixel_pts, arm_pts)):
        cv2.drawMarker(vis, (int(p[0]), int(p[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, f"{i + 1}: ({a[0]:.0f},{a[1]:.0f})", (int(p[0]) + 10, int(p[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if pending is not None:
        cv2.drawMarker(vis, (int(pending[0]), int(pending[1])), (0, 200, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, "type arm x,y in terminal", (int(pending[0]) + 10, int(pending[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(vis, f"pairs: {len(pixel_pts)}  (need >={mapping.MIN_PAIRS}, use {mapping.RECOMMENDED_PAIRS}+)"
                     f"   u=undo d=done q=quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def collect(camera: int, out_path: str, snap: bool = True) -> None:
    import detect

    log = runlog.start_run("calibrate")
    det = detect.Detector(roi=None) if snap else None
    cap = detect.open_camera(camera)
    pixel_pts: list[tuple[float, float]] = []
    arm_pts: list[tuple[float, float]] = []
    pending: list[tuple[float, float] | None] = [None]
    latest: dict = {"dets": [], "frame_wh": (config.FRAME_WIDTH, config.FRAME_HEIGHT)}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or pending[0] is not None:
            return
        px, py = float(x), float(y)
        snapped = _nearest_detection(latest["dets"], px, py) if snap else None
        if snapped is not None:
            px, py = snapped.cx, snapped.cy
            log.info("click (%d,%d) snapped to %s bbox centre (%.0f,%.0f)", x, y, snapped.cls, px, py)
            how = f"snapped to the detected {snapped.cls} block's centre ({px:.0f}, {py:.0f})"
        else:
            log.info("click pixel=(%d,%d) (no detection nearby; using the raw click)", x, y)
            how = "no block detected near the click - using the raw click (put a block ON the mark for best accuracy)"
        pending[0] = (px, py)
        print(f"\nMark {len(pixel_pts) + 1}: {how}. Jog the suction cup onto this mark, read the arm's (x, y) mm, "
              f"type it as `x,y` and press Enter (u = discard, d = done, q = quit): ", end="", flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(q,), daemon=True).start()
    print(f"Place a block on a tape mark and click it on the feed. Use {mapping.RECOMMENDED_PAIRS}-6 marks spread over "
          f"the whole pick area, not in a line, one near the edge.")

    def finish() -> bool:
        if len(pixel_pts) < mapping.MIN_PAIRS:
            print(f"\nNeed at least {mapping.MIN_PAIRS} pairs, have {len(pixel_pts)}.")
            return False
        try:
            H, _ = mapping.fit_homography(pixel_pts, arm_pts)
        except ValueError as e:
            print(f"\nCannot fit: {e}. Undo the bad pair (u) or add marks.")
            log.warning("fit refused: %s", e)
            return False
        mapping.save_calibration(H, pixel_pts, arm_pts, path=out_path, frame_size=latest["frame_wh"])
        log.info("saved %s pairs=%d frame=%s", out_path, len(pixel_pts), latest["frame_wh"])
        print(f"\nSaved {out_path}\nH =\n{np.array2string(H, precision=5)}")
        for line in mapping.describe_residuals(pixel_pts, arm_pts):
            print(line)
        return True

    try:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("camera read failed")
                break
            n += 1
            latest["frame_wh"] = (frame.shape[1], frame.shape[0])
            if det is not None and (n % 3 == 1) and pending[0] is None:
                latest["dets"] = det.detect(frame)
            cv2.imshow(WINDOW, _draw_pairs(frame, pixel_pts, arm_pts, pending[0], latest["dets"]))
            key = cv2.waitKey(30) & 0xFF
            cmd = None
            try:
                cmd = q.get_nowait()
            except queue.Empty:
                pass
            if key == ord("q") or cmd == "q":
                print("\nQuit without saving.")
                break
            if key == ord("d") or cmd == "d":
                if finish():
                    break
                continue
            if key == ord("u") or cmd == "u":
                if pending[0] is not None:
                    pending[0] = None
                    print("\nDiscarded pending click.")
                elif pixel_pts:
                    pixel_pts.pop(); arm_pts.pop()
                    print("\nRemoved last pair.")
                continue
            if cmd and pending[0] is not None:
                try:
                    xs, ys = cmd.replace(" ", "").split(",")
                    ax, ay = float(xs), float(ys)
                except ValueError:
                    print(f"Could not parse {cmd!r}; type it as x,y (mm): ", end="", flush=True)
                    continue
                pixel_pts.append(pending[0]); arm_pts.append((ax, ay))
                log.info("pair %d pixel=%s arm=(%.1f,%.1f)", len(pixel_pts), pending[0], ax, ay)
                print(f"Pair {len(pixel_pts)} recorded. Click the next mark, or d when done.")
                pending[0] = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


def verify(camera: int, path: str) -> None:
    """Click on the feed (or press b for the best-detected block); the arm hovers above that point.

    Gate: within ~5 mm on 3+ spots including one near the edge of the pick area.

    Offset correction: if the cup is consistently off-centre, nudge it with w/a/s/d (2 mm) until it is dead
    centre over the block, press c to record the offset, repeat on 2-3 blocks, then x to save a calibration
    with the mean offset folded into the homography.
    """
    import arm as arm_mod
    import detect

    log = runlog.start_run("calibrate-verify")
    cal = mapping.load_calibration(path)
    H = cal["H"]
    table_z = config.require("TABLE_Z_MM")
    hover_z = _pick_z() + config.HOVER_OFFSET_MM
    arm_mod.check_target(*config.require("HOME_XYZ_MM"))
    det = detect.Detector(roi=None)  # any block-sized detection counts in verify (draw() marks targets thicker)
    cap = detect.open_camera(camera)
    mapping.check_frame_size(cal, detect.grab(cap))
    a = arm_mod.Arm()
    a.connect()
    a.confirm_workspace_clear()
    a.home()
    clicks: "queue.Queue[tuple[float, float]]" = queue.Queue()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, f, p: clicks.put((float(x), float(y))) if e == cv2.EVENT_LBUTTONDOWN else None)
    print("click = hover there | b = hover over best block | w/a/s/d = nudge cup 2 mm | c = record offset "
          "| x = save corrected calibration | h = home | q = quit")
    last = None            # (px, py) of the current hover target
    hover_xy = None        # arm (x, y) the calibration mapped it to
    cup_xy = None          # arm (x, y) where the cup is now (after nudges)
    offsets: list[tuple[float, float]] = []
    dets = []
    n = 0
    step = 2.0

    def hud(vis):
        line = "click=hover  b=best block  wasd=nudge  c=record offset  x=save  h=home  q=quit"
        cv2.putText(vis, line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if cup_xy is not None and hover_xy is not None:
            dx, dy = cup_xy[0] - hover_xy[0], cup_xy[1] - hover_xy[1]
            cv2.putText(vis, f"nudge so far: ({dx:+.0f}, {dy:+.0f}) mm   recorded offsets: {len(offsets)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return vis

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % 3 == 1:
                dets = det.detect(frame)
            vis = detect.draw(frame, dets, roi=None)
            if last is not None:
                cv2.drawMarker(vis, (int(last[0]), int(last[1])), (0, 200, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.imshow(WINDOW, hud(vis))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                a.home()
                last = hover_xy = cup_xy = None
                continue
            if key in (ord("w"), ord("a"), ord("s"), ord("d")) and cup_xy is not None:
                dx = {ord("a"): -step, ord("d"): step}.get(key, 0.0)
                dy = {ord("w"): -step, ord("s"): step}.get(key, 0.0)  # forward (away from the base) is -Y
                try:
                    a.move_to(cup_xy[0] + dx, cup_xy[1] + dy, hover_z, 400)
                    cup_xy = (cup_xy[0] + dx, cup_xy[1] + dy)
                except (arm_mod.UnsafeTarget, arm_mod.MoveRefused) as e:
                    print(f"refused: {e}")
                continue
            if key == ord("c") and cup_xy is not None and hover_xy is not None:
                off = (cup_xy[0] - hover_xy[0], cup_xy[1] - hover_xy[1])
                offsets.append(off)
                mean = (sum(o[0] for o in offsets) / len(offsets), sum(o[1] for o in offsets) / len(offsets))
                log.info("offset recorded %s; mean of %d = (%.1f, %.1f) mm", off, len(offsets), *mean)
                print(f"recorded offset ({off[0]:+.0f}, {off[1]:+.0f}) mm; mean of {len(offsets)}: ({mean[0]:+.1f}, {mean[1]:+.1f}). "
                      f"Do another block (b / click) or x to save.")
                continue
            if key == ord("x"):
                if not offsets:
                    print("no offsets recorded yet (nudge with w/a/s/d, then c)")
                    continue
                mean = (sum(o[0] for o in offsets) / len(offsets), sum(o[1] for o in offsets) / len(offsets))
                H = mapping.translate_homography(H, *mean)
                prev = tuple(cal.get("offset_correction", (0.0, 0.0)))
                total = (prev[0] + mean[0], prev[1] + mean[1])
                mapping.save_calibration(H, cal["pixel_pts"], cal["arm_pts"], path=path,
                                         frame_size=tuple(cal["frame_size"]), offset_correction=total,
                                         offset_samples=len(offsets))
                cal = mapping.load_calibration(path)
                log.info("saved %s with offset correction (%.1f, %.1f) mm from %d blocks", path, *mean, len(offsets))
                print(f"Saved {path}: mapping shifted by ({mean[0]:+.1f}, {mean[1]:+.1f}) mm (total correction "
                      f"({total[0]:+.1f}, {total[1]:+.1f})). Hover a block again to check; it should now be centred.")
                offsets = []
                continue
            if key == ord("b"):
                if dets:
                    best = max(dets, key=lambda d: d.conf)
                    clicks.put((best.cx, best.cy))
                    print(f"best detection: {best}")
                else:
                    print("no block detected")
            try:
                px, py = clicks.get_nowait()
            except queue.Empty:
                continue
            x, y = mapping.pixel_to_arm(px, py, H)
            log.info("verify pixel=(%.0f,%.0f) -> arm=(%.1f,%.1f) hover z=%.1f", px, py, x, y, hover_z)
            try:
                a.rise()  # straight up from wherever the arm actually is, then sideways at travel height
                a.move_to(x, y, config.TRAVEL_Z_MM)
                a.move_to(x, y, hover_z)
                last, hover_xy, cup_xy = (px, py), (x, y), (x, y)
            except (arm_mod.UnsafeTarget, arm_mod.MoveRefused) as e:
                log.warning("refused: %s", e)
                print(f"refused: {e}")
                continue
            print(f"pixel ({px:.0f},{py:.0f}) -> arm ({x:.1f}, {y:.1f}) mm. Is the cup centred over the block? "
                  f"If not: w/a/s/d to nudge it there, then c.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            a.home()
        finally:
            a.close()


def check(path: str) -> None:
    data = mapping.load_calibration(path)
    H = data["H"]
    print(f"{path}: created {data.get('created')} frame_size={data.get('frame_size')} "
          f"offset_correction={tuple(data.get('offset_correction', (0.0, 0.0)))} mm")
    print("H =\n" + np.array2string(H, precision=5))
    px, ar = data["pixel_pts"], data["arm_pts"]
    for line in mapping.describe_residuals(px, ar):
        print(line)
    w, h = data.get("frame_size", (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    for name, (x, y) in {"centre": (w / 2, h / 2), "top-left": (0, 0), "bottom-right": (w, h)}.items():
        ax, ay = mapping.pixel_to_arm(x, y, H)
        print(f"  {name:12s} pixel=({x:.0f},{y:.0f}) -> arm=({ax:.1f},{ay:.1f}) mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto", action="store_true", help="arm-driven calibration (easiest)")
    ap.add_argument("--carry", action="store_true", help="with --auto: the arm places the block itself (most accurate)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-snap", action="store_true", help="collect: use raw clicks, do not snap to detected blocks")
    ap.add_argument("--camera", type=int, default=None, help="default config.WEBCAM_INDEX")
    ap.add_argument("--out", default=None, help="default config.CALIBRATION_PATH")
    args = ap.parse_args()
    args.out = args.out or config.CALIBRATION_PATH
    camera = config.WEBCAM_INDEX if args.camera is None else args.camera
    if args.check:
        check(args.out)
    elif args.auto:
        auto(camera, args.out, carry=args.carry)
    elif args.verify:
        verify(camera, args.out)
    else:
        collect(camera, args.out, snap=not args.no_snap)


if __name__ == "__main__":
    main()
