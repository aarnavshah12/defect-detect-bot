"""Click-to-calibrate the pixel -> arm homography, plus verify and check modes.

  python calibrate.py            collect: put a BLOCK on each tape mark, click it on the live feed (the
                                 click snaps to the detected block's bbox centre - the same quantity
                                 pick.py maps), jog the arm's suction cup onto the mark, type the arm
                                 (x, y) mm in the terminal. Writes calibration.npy.
  python calibrate.py --auto     EASIEST: the arm drives itself to a grid of spots; at each one you slide
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
    """
    import arm as arm_mod
    import detect

    log = runlog.start_run("calibrate-verify")
    cal = mapping.load_calibration(path)
    H = cal["H"]
    table_z = config.require("TABLE_Z_MM")
    hover_z = table_z + config.HOVER_OFFSET_MM
    arm_mod.check_target(*config.require("HOME_XYZ_MM"))
    det = detect.Detector(roi=None)
    cap = detect.open_camera(camera)
    mapping.check_frame_size(cal, detect.grab(cap))
    a = arm_mod.Arm()
    a.connect()
    a.confirm_workspace_clear()
    a.home()
    clicks: "queue.Queue[tuple[float, float]]" = queue.Queue()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, f, p: clicks.put((float(x), float(y))) if e == cv2.EVENT_LBUTTONDOWN else None)
    print("Click a point -> the arm hovers above it. b = hover above the best-detected block. h = home. q = quit.")
    last = None
    dets = []
    n = 0
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
            cv2.putText(vis, "click = hover there   b = hover over best block   h = home   q = quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                a.home()
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
            last = (px, py)
            x, y = mapping.pixel_to_arm(px, py, H)
            log.info("verify pixel=(%.0f,%.0f) -> arm=(%.1f,%.1f) hover z=%.1f", px, py, x, y, hover_z)
            try:
                a.move_to(x, y, hover_z)
            except (arm_mod.UnsafeTarget, arm_mod.MoveRefused) as e:
                log.warning("refused: %s", e)
                print(f"refused: {e}")
                continue
            print(f"pixel ({px:.0f},{py:.0f}) -> arm ({x:.1f}, {y:.1f}) mm. Measure the cup-to-target offset; "
                  f"gate is ~5 mm on 3+ spots including one near the edge.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            a.home()
        finally:
            a.close()


# --auto grid in arm mm (in front of the base, comfortably inside reach). Points the arm refuses or the
# camera cannot see are skipped; at least mapping.MIN_PAIRS (ideally RECOMMENDED_PAIRS) must succeed.
AUTO_GRID = config.CALIB_GRID
PRESENT_CLEARANCE_MM = 50.0  # cup height above table Z while you slide the (40 mm) block under it


def auto_collect(points, arm, grab, detector, ask, target_class: str, table_z: float, log, show=None):
    """Core of --auto, with injectable I/O so it can be tested offline.

    For each (x, y): move the cup to (x, y, table_z + PRESENT_CLEARANCE_MM), ask the owner to slide the block
    under it and press Enter (or 's' to skip), read the arm's real x,y, send the arm home (out of the camera's
    view), grab a fresh frame, take the best `target_class` detection as the pixel. Returns (pixel_pts, arm_pts).
    """
    pixel_pts, arm_pts = [], []
    present_z = table_z + PRESENT_CLEARANCE_MM
    travel_z = present_z + 50.0
    for k, (x, y) in enumerate(points, 1):
        try:
            arm.move_to(x, y, travel_z)
            arm.move_to(x, y, present_z)
        except (arm_mod_unsafe(), arm_mod_refused()) as e:
            log.warning("point %d (%.0f, %.0f) not reachable (%s) - skipped", k, x, y, e)
            arm.home()
            continue
        what = "block" if target_class == "any" else f"{target_class} block"
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
        arm.move_to(x, y, travel_z)
        arm.home()
        arm.wait(1.5)  # let the arm clear the view, the owner's hand leave, and the camera settle
        import detect as detect_mod
        best = None
        for attempt in range(4):
            # Two looks 0.5 s apart must agree: a hand still in the shot (or moving block) is rejected.
            frame = grab()
            dets = [d for d in detector.detect(frame) if detect_mod.is_target(d, target_class)]
            arm.wait(0.5)
            frame2 = grab()
            dets2 = [d for d in detector.detect(frame2) if detect_mod.is_target(d, target_class)]
            if dets and dets2:
                b1, b2 = max(dets, key=lambda d: d.conf), max(dets2, key=lambda d: d.conf)
                if math.hypot(b1.cx - b2.cx, b1.cy - b2.cy) <= 15:
                    best = b2
                    dets = dets2
                    frame = frame2
                    break
            log.warning("point %d: block not seen steadily (attempt %d) - keep hands out of the picture", k, attempt + 1)
            arm.wait(1.0)
        if best is None:
            log.warning("point %d: no %s seen steadily by the camera after the arm moved away - skipped "
                        "(is this spot in the camera's view? hands out?)", k, what)
            if show:
                show(frame, [], None)
            continue
        pixel_pts.append((best.cx, best.cy))
        arm_pts.append((ax, ay))
        log.info("pair %d: pixel=(%.0f,%.0f) arm=(%.1f,%.1f) conf=%.2f", len(pixel_pts), best.cx, best.cy, ax, ay, best.conf)
        if show:
            show(frame, dets, best)
    return pixel_pts, arm_pts


def arm_mod_unsafe():
    import arm as arm_mod
    return arm_mod.UnsafeTarget


def arm_mod_refused():
    import arm as arm_mod
    return arm_mod.MoveRefused


def auto(camera: int, out_path: str) -> None:
    import arm as arm_mod
    import detect

    log = runlog.start_run("calibrate-auto")
    table_z = config.require("TABLE_Z_MM")
    arm_mod.check_target(*config.require("HOME_XYZ_MM"))
    det = detect.Detector(roi=None)
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
    print(f"Auto calibration: {len(AUTO_GRID)} spots. Keep ONLY ONE {what} on the table (and nothing else "
          f"block-sized in view). "
          f"The arm will stop above each spot; slide the block under the cup and press Enter.")
    a = arm_mod.Arm().connect()
    try:
        a.confirm_workspace_clear()
        a.home()
        pixel_pts, arm_pts = auto_collect(AUTO_GRID, a, grab, det, input, config.TARGET_CLASS, table_z, log, show)
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
    for line in mapping.describe_residuals(pixel_pts, arm_pts):
        print(line)
    if len(pixel_pts) < mapping.RECOMMENDED_PAIRS:
        print(f"Only {len(pixel_pts)} points: no typo check possible. Fine if --verify looks good.")


def check(path: str) -> None:
    data = mapping.load_calibration(path)
    H = data["H"]
    print(f"{path}: created {data.get('created')} frame_size={data.get('frame_size')}")
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
        auto(camera, args.out)
    elif args.verify:
        verify(camera, args.out)
    else:
        collect(camera, args.out, snap=not args.no_snap)


if __name__ == "__main__":
    main()
