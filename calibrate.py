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
            import pick as pick_mod
            if not pick_mod.inside_pick_area(x, y):
                print(f"NOTE: ({x:.0f}, {y:.0f}) is outside config.PICK_AREA x{config.PICK_AREA_X_MM} "
                      f"y{config.PICK_AREA_Y_MM} - pick.py would SKIP this block. Widen the area if the cup lands well here.")
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


# --auto grid in arm mm (in front of the base, comfortably inside reach). Points the arm refuses or the
# camera cannot see are skipped; at least mapping.MIN_PAIRS (ideally RECOMMENDED_PAIRS) must succeed.
def _auto_grid():
    """config.CALIB_GRID minus spots near the bin, ordered farthest-from-bin first."""
    import arm as arm_mod

    dx, dy = (config.DROP_XYZ_MM or (0.0, 0.0, 0.0))[:2]
    pz = (config.TABLE_Z_MM + config.BLOCK_HEIGHT_MM - config.CUP_PRESS_MM) if config.TABLE_Z_MM is not None else 90.0
    pts = [(x, y) for x, y in config.CALIB_GRID if math.hypot(x - dx, y - dy) >= config.DROP_KEEPOUT_MM]
    # comfortably reachable spots first (the first one seeds the whole run), farthest from the bin first
    return sorted(pts, key=lambda p: (arm_mod.extension_ratio(p[0], p[1], pz) > 0.8, -math.hypot(p[0] - dx, p[1] - dy)))


AUTO_GRID = _auto_grid()
PRESENT_CLEARANCE_MM = config.BLOCK_HEIGHT_MM + 30.0  # cup height above table Z while you slide the block under it
EDGE_MARGIN_PX = 8  # a box touching the frame edge is clipped: its centre is wrong


def _pick_z() -> float:
    return config.TABLE_Z_MM + config.BLOCK_HEIGHT_MM - config.CUP_PRESS_MM


def _clipped(d, frame) -> bool:
    x1, y1, x2, y2 = d.bbox
    h, w = frame.shape[:2]
    return x1 <= EDGE_MARGIN_PX or y1 <= EDGE_MARGIN_PX or x2 >= w - EDGE_MARGIN_PX or y2 >= h - EDGE_MARGIN_PX


CALIB_CONFIDENCE = 0.25  # only one block is on the table during calibration, so a low cut-off is safe


def _steady_detection(grab, detector, target_class, arm, log, k, quiet=False):
    """Three looks; any two within 25 px agree (rejects a hand in the shot). Returns (best, frame, dets) or None.

    On failure logs what the camera did see (raw boxes, and which were dropped as clipped) and saves the frame.
    """
    import detect as detect_mod
    looks = []
    for attempt in range(3):
        frame = grab()
        raw = detector.detect(frame)
        ok = [d for d in raw if detect_mod.is_target(d, target_class) and not _clipped(d, frame)]
        looks.append((frame, raw, ok))
        for (f1, _, d1) in looks[:-1]:
            if d1 and ok:
                b1, b2 = max(d1, key=lambda d: d.conf), max(ok, key=lambda d: d.conf)
                if math.hypot(b1.cx - b2.cx, b1.cy - b2.cy) <= 25:
                    return b2, frame, ok
        arm.wait(0.5)
    if not quiet:
        for i, (frame, raw, ok) in enumerate(looks):
            desc = ", ".join(f"{d.cls} {d.conf:.2f} ({d.cx:.0f},{d.cy:.0f}) {d.w:.0f}x{d.h:.0f}"
                             f"{' CLIPPED' if _clipped(d, frame) else ''}" for d in raw) or "nothing"
            log.warning("point %d look %d saw: %s", k, i + 1, desc)
        log.warning("point %d: saved %s", k, runlog.save_frame(looks[-1][0], f"calib-point{k:02d}-miss"))
    return None


def auto_collect(points, arm, grab, detector, ask, target_class: str, table_z: float, log, show=None,
                 carry: bool = False):
    """Core of --auto, with injectable I/O so it can be tested offline.

    Manual (carry=False): for each (x, y) the cup stops above the spot, the owner slides the block under it and
    presses Enter, the arm goes home, the camera records the block's bbox centre.

    Carry (carry=True): the owner centres the block under the cup ONCE; the arm picks it up and, for each
    (x, y), sets it down there, goes home, lets the camera record it, then picks it back up. Placement is then
    the arm's repeatability, not a human's eye. A block that failed to lift is caught because the camera then
    sees it at the previous spot.
    """
    pixel_pts, arm_pts = [], []
    present_z = table_z + PRESENT_CLEARANCE_MM
    travel_z = max(config.TRAVEL_Z_MM, present_z + 30.0)
    pick_z = _pick_z()
    what = "block" if target_class == "any" else f"{target_class} block"
    unsafe, refused = arm_mod_unsafe(), arm_mod_refused()
    last_pixel = None

    def go(x, y, z, ms=None):
        arm.move_to(x, y, z) if ms is None else arm.move_to(x, y, z, ms)

    if carry:
        x0, y0 = points[0]
        go(x0, y0, travel_z)
        go(x0, y0, pick_z + 20.0)
        answer = ask(f"Centre the {what} under the cup - look straight down from above, take your time - "
                     f"then press Enter (q = stop): ").strip().lower()
        if answer == "q":
            arm.home()
            return pixel_pts, arm_pts
        # Where is the block before the pick? (cup is 20 mm above it; usually still visible from the side)
        pre = _steady_detection(grab, detector, target_class, arm, log, 0, quiet=True)
        pre_px = (pre[0].cx, pre[0].cy) if pre else None
        log.info("block before the pick: %s", f"pixel ({pre_px[0]:.0f},{pre_px[1]:.0f})" if pre_px else "not visible (cup in the way)")
        go(x0, y0, pick_z, 700)
        arm.suction(True)
        arm.wait(config.SUCTION_ON_PAUSE_S + 0.3)
        go(x0, y0, travel_z)
        arm.home()
        arm.wait(1.0)
        after = _steady_detection(grab, detector, target_class, arm, log, 0, quiet=True)
        if after is not None:
            a_px = (after[0].cx, after[0].cy)
            still_there = pre_px is not None and math.hypot(a_px[0] - pre_px[0], a_px[1] - pre_px[1]) <= 60
            if still_there:
                path = runlog.save_frame(after[1], "carry-lift-failed")
                arm.suction(False)
                arm.home()
                raise RuntimeError(f"the cup did not lift the block: after the pick the camera still sees it at "
                                   f"pixel ({a_px[0]:.0f},{a_px[1]:.0f}), where it was before. Pick height was "
                                   f"{pick_z:.0f} mm (TABLE_Z {table_z:.0f} + BLOCK_HEIGHT {config.BLOCK_HEIGHT_MM:.0f} "
                                   f"- CUP_PRESS {config.CUP_PRESS_MM:.0f}): lower BLOCK_HEIGHT_MM if the cup stopped "
                                   f"above the block. Frame: {path}")
            log.info("a block-sized detection at (%.0f,%.0f) after the pick is not at the pick spot (probably the "
                     "block on the cup, seen at home) - continuing", a_px[0], a_px[1])
        log.info("block lifted; starting the %d-point carry calibration", len(points))

    for k, (x, y) in enumerate(points, 1):
        try:
            if carry:
                go(x, y, travel_z)
                go(x, y, pick_z + 20.0)
                go(x, y, pick_z + config.RELEASE_LIFT_MM, 700)  # block just above the table, cup NOT pressed
                real = arm.read_xyz()  # where the arm ACTUALLY is (servo sag / IK error at long reach)
                placed_xy = (float(real[0]), float(real[1])) if real else (float(x), float(y))
                if real and math.hypot(real[0] - x, real[1] - y) > 4:
                    log.info("point %d: arm at (%.0f, %.0f) vs commanded (%.0f, %.0f) - using the real position",
                             k, real[0], real[1], x, y)
                _release(arm)
                go(x, y, travel_z)        # straight up with the valve still OPEN: no vacuum can form even if
                _valve_close(arm)         # the cup touches the block again on the way; close only when clear
            else:
                go(x, y, travel_z)
                go(x, y, present_z)
        except (unsafe, refused) as e:
            log.warning("point %d (%.0f, %.0f) not reachable (%s) - skipped", k, x, y, e)
            arm.home()
            continue
        if not carry:
            answer = ask(f"Point {k}/{len(points)}: slide the {what} under the cup (centred under it), "
                         f"then press Enter (s = skip, q = stop): ").strip().lower()
            if answer == "q":
                arm.home()
                break
            if answer == "s":
                arm.home()
                continue
            real = arm.read_xyz()
            ax, ay = (float(real[0]), float(real[1])) if real else (x, y)
            go(x, y, travel_z)
        else:
            ax, ay = placed_xy
        arm.home()
        arm.wait(1.5)  # let the arm clear the view, the owner's hand leave, and the camera settle
        seen = _steady_detection(grab, detector, target_class, arm, log, k)
        if seen is None:
            log.warning("point %d: no %s seen steadily by the camera after the arm moved away - skipped "
                        "(is this spot in the camera's view? hands out?)", k, what)
            if show:
                show(grab(), [], None)
            if carry:
                # the block should be at (x, y); try to pick it back up anyway
                try:
                    go(x, y, travel_z); go(x, y, pick_z + 20.0); go(x, y, pick_z, 700)
                    arm.suction(True); arm.wait(config.SUCTION_ON_PAUSE_S + 0.3); go(x, y, travel_z)
                except (unsafe, refused) as e:
                    log.error("could not re-pick the block at (%.0f, %.0f): %s", x, y, e)
                    arm.home()
                    break
            continue
        best, frame, dets = seen
        if carry and last_pixel is not None and math.hypot(best.cx - last_pixel[0], best.cy - last_pixel[1]) <= 25:
            log.error("point %d: the block is still where it was at the previous point - it was not picked up. "
                      "Stopping; check suction / pick height.", k)
            arm.home()
            break
        last_pixel = (best.cx, best.cy)
        pixel_pts.append((best.cx, best.cy))
        arm_pts.append((ax, ay))
        log.info("pair %d: pixel=(%.0f,%.0f) arm=(%.1f,%.1f) conf=%.2f", len(pixel_pts), best.cx, best.cy, ax, ay, best.conf)
        if show:
            show(frame, dets, best)
        if carry and k < len(points):
            try:
                go(x, y, travel_z); go(x, y, pick_z + 20.0); go(x, y, pick_z, 700)
                arm.suction(True); arm.wait(config.SUCTION_ON_PAUSE_S + 0.3); go(x, y, travel_z)
            except (unsafe, refused) as e:
                log.error("could not re-pick the block at (%.0f, %.0f): %s", x, y, e)
                arm.home()
                break
    if carry:
        arm.suction(False)
    arm.home()
    return pixel_pts, arm_pts


def _release(arm):
    """Pump off + valve open, wait for the vacuum to vent (valve left open)."""
    if hasattr(arm, "vent"):
        arm.vent()
    else:
        arm.suction(False)
    arm.wait(config.VENT_S)


def _valve_close(arm):
    if hasattr(arm, "valve_close"):
        arm.valve_close()


def arm_mod_unsafe():
    import arm as arm_mod
    return arm_mod.UnsafeTarget


def arm_mod_refused():
    import arm as arm_mod
    return arm_mod.MoveRefused


def auto(camera: int, out_path: str, carry: bool = False) -> None:
    import arm as arm_mod
    import detect

    log = runlog.start_run("calibrate-auto")
    table_z = config.require("TABLE_Z_MM")
    arm_mod.check_target(*config.require("HOME_XYZ_MM"))
    det = detect.Detector(roi=None, confidence=CALIB_CONFIDENCE)
    cap = detect.open_camera(camera)
    frame_wh = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    def show(frame, dets, best):
        vis = detect.draw(frame, dets, roi=None)
        if best is not None:
            cv2.drawMarker(vis, (int(best.cx), int(best.cy)), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.imshow(WINDOW, vis)
        cv2.waitKey(1)

    def grab():
        nonlocal frame_wh
        f = detect.grab(cap)
        frame_wh = (f.shape[1], f.shape[0])
        return f

    show(grab(), [], None)
    what = "block" if config.TARGET_CLASS == "any" else f"{config.TARGET_CLASS} block"
    what = "block"
    if carry:
        print(f"Carry calibration: {len(AUTO_GRID)} spots. ONE {what} on the table, nothing else block-sized in view. "
              f"You centre the block under the cup once; the arm does the rest (about {len(AUTO_GRID) * 15} s).")
    else:
        print(f"Auto calibration: {len(AUTO_GRID)} spots. Keep ONLY ONE {what} on the table (and nothing else "
              f"block-sized in view). The arm will stop above each spot; slide the block under the cup and press Enter.")
    a = arm_mod.Arm().connect()
    try:
        a.confirm_workspace_clear()
        a.home()
        # Calibration is geometry: ANY block works, whatever class the model gives it.
        pixel_pts, arm_pts = auto_collect(AUTO_GRID, a, grab, det, input, "any", table_z, log, show, carry=carry)
    finally:
        try:
            a.home()
        finally:
            a.close()
            cap.release()
            cv2.destroyAllWindows()
    print(f"\n{len(pixel_pts)} usable points.")
    if len(pixel_pts) < mapping.MIN_PAIRS:
        raise SystemExit(f"need at least {mapping.MIN_PAIRS}; move the camera so it sees more of the arm's "
                         f"workspace, or run again.")
    try:
        H, _ = mapping.fit_homography(pixel_pts, arm_pts)
    except ValueError as e:
        raise SystemExit(f"cannot fit: {e}")
    mapping.save_calibration(H, pixel_pts, arm_pts, path=out_path, frame_size=frame_wh)
    log.info("saved %s pairs=%d frame=%s", out_path, len(pixel_pts), frame_wh)
    print(f"Saved {out_path}")
    _, res = mapping.fit_homography(pixel_pts, arm_pts)
    print(f"  FIT ERROR = expected pick-time error: rms {float((res ** 2).mean() ** 0.5):.1f} mm, max {res.max():.1f} mm "
          f"over {len(pixel_pts)} points  (a suction cup on a 40 mm block tolerates ~10 mm)")
    print("  per-point leave-one-out check (includes that point's own placement noise; only the SUSPECT line matters):")
    for line in mapping.describe_residuals(pixel_pts, arm_pts):
        print(line)
    if len(pixel_pts) < mapping.RECOMMENDED_PAIRS:
        print(f"Only {len(pixel_pts)} points: no typo check possible. Fine if --verify looks good.")


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
