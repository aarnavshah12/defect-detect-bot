"""Click-to-calibrate the pixel -> arm homography, plus verify and check modes.

  python calibrate.py            collect: click each tape mark on the live feed, jog the arm to it,
                                 type the arm (x, y) mm in the terminal; writes calibration.npy
  python calibrate.py --verify   verify: click anywhere; the arm hovers over that point at
                                 TABLE_Z + HOVER_OFFSET (moves the arm -> asks "Workspace clear?")
  python calibrate.py --check    offline: print the saved matrix, residuals and a few sample mappings

Collect-mode keys (feed window): u = undo last pair, d = done (fit + save), q = quit without saving.
Terminal input after each click:  "x,y"  (arm mm)  |  u = discard this click  |  d = done  |  q = quit
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import cv2
import numpy as np

import config
import mapping
import runlog

WINDOW = "calibrate"


def _stdin_reader(q: "queue.Queue[str]") -> None:
    for line in sys.stdin:
        q.put(line.strip())


def _draw_pairs(frame, pixel_pts, arm_pts, pending):
    vis = frame.copy()
    for i, (p, a) in enumerate(zip(pixel_pts, arm_pts)):
        cv2.drawMarker(vis, (int(p[0]), int(p[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, f"{i + 1}: ({a[0]:.0f},{a[1]:.0f})", (int(p[0]) + 10, int(p[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if pending is not None:
        cv2.drawMarker(vis, (int(pending[0]), int(pending[1])), (0, 200, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, "type arm x,y in terminal", (int(pending[0]) + 10, int(pending[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(vis, f"pairs: {len(pixel_pts)}  (need >=4)   u=undo d=done q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def collect(camera: int, out_path: str) -> None:
    from detect import open_camera

    log = runlog.start_run("calibrate")
    cap = open_camera(camera)
    pixel_pts: list[tuple[float, float]] = []
    arm_pts: list[tuple[float, float]] = []
    pending: list[tuple[float, float] | None] = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and pending[0] is None:
            pending[0] = (float(x), float(y))
            log.info("click pixel=(%d,%d)", x, y)
            print(f"\nMark {len(pixel_pts) + 1} at pixel ({x}, {y}). Jog the arm so the suction cup touches "
                  f"this mark, read the arm's (x, y) in mm, type it as `x,y` and press Enter "
                  f"(u = discard click, d = done, q = quit): ", end="", flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(q,), daemon=True).start()
    print("Click a tape mark on the feed. Spread 4-6 marks over the whole pick area, not in a line.")

    def finish() -> bool:
        if len(pixel_pts) < 4:
            print(f"\nNeed at least 4 pairs, have {len(pixel_pts)}.")
            return False
        H, res = mapping.fit_homography(pixel_pts, arm_pts)
        mapping.save_calibration(H, pixel_pts, arm_pts, path=out_path)
        log.info("saved %s", out_path)
        print(f"\nSaved {out_path}\nH =\n{np.array2string(H, precision=5)}")
        for i, r in enumerate(res):
            print(f"  pair {i + 1}: pixel={pixel_pts[i]} arm={arm_pts[i]} residual={r:.2f} mm")
        print(f"  max residual {res.max():.2f} mm, mean {res.mean():.2f} mm "
              f"(>3 mm on a point usually means a mistyped value or a moved camera)")
        return True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("camera read failed")
                break
            cv2.imshow(WINDOW, _draw_pairs(frame, pixel_pts, arm_pts, pending[0]))
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
    """Click on the feed; the arm hovers above that point. Gate: ~5 mm on 3+ spots incl. one near the edge."""
    import arm as arm_mod
    from detect import open_camera

    log = runlog.start_run("calibrate-verify")
    H = mapping.load_homography(path)
    table_z = config.require("TABLE_Z_MM")
    hover_z = table_z + config.HOVER_OFFSET_MM
    a = arm_mod.Arm()
    a.connect()
    a.confirm_workspace_clear()
    a.home()
    cap = open_camera(camera)
    clicks: "queue.Queue[tuple[int, int]]" = queue.Queue()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, f, p: clicks.put((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    print("Click a point on the feed; the arm will hover above it. h = home, q = quit.")
    last = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis = frame.copy()
            if last is not None:
                cv2.drawMarker(vis, last, (0, 200, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(vis, "click = hover there   h = home   q = quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                a.home()
            try:
                px, py = clicks.get_nowait()
            except queue.Empty:
                continue
            last = (px, py)
            x, y = mapping.pixel_to_arm(px, py, H)
            log.info("verify click pixel=(%d,%d) -> arm=(%.1f,%.1f) hover z=%.1f", px, py, x, y, hover_z)
            try:
                a.move_to(x, y, hover_z)
            except arm_mod.UnsafeTarget as e:
                log.warning("refused: %s", e)
            print(f"pixel ({px},{py}) -> arm ({x:.1f}, {y:.1f}) mm. Measure the cup-to-mark offset; "
                  f"gate is ~5 mm on 3+ spots including one near the edge.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        a.home()
        a.close()


def check(path: str) -> None:
    data = mapping.load_calibration(path)
    H = data["H"]
    print(f"{path}: created {data.get('created')} frame_size={data.get('frame_size')}")
    print("H =\n" + np.array2string(H, precision=5))
    px, ar = data["pixel_pts"], data["arm_pts"]
    _, res = mapping.fit_homography(px, ar)
    for i in range(len(px)):
        print(f"  pair {i + 1}: pixel=({px[i][0]:.0f},{px[i][1]:.0f}) arm=({ar[i][0]:.1f},{ar[i][1]:.1f}) "
              f"residual={res[i]:.2f} mm")
    w, h = data.get("frame_size", (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    for name, (x, y) in {"centre": (w / 2, h / 2), "top-left": (0, 0), "bottom-right": (w, h)}.items():
        ax, ay = mapping.pixel_to_arm(x, y, H)
        print(f"  {name:12s} pixel=({x:.0f},{y:.0f}) -> arm=({ax:.1f},{ay:.1f}) mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--camera", type=int, default=config.WEBCAM_INDEX)
    ap.add_argument("--out", default=config.CALIBRATION_PATH)
    args = ap.parse_args()
    if args.check:
        check(args.out)
    elif args.verify:
        verify(args.camera, args.out)
    else:
        collect(args.camera, args.out)


if __name__ == "__main__":
    main()
