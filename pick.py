"""Main pick loop: detect -> map -> pick -> drop -> home -> re-detect, until no target blocks remain.

    python pick.py --dry-run          no serial writes; logs planned targets and draws them on the frame
    python pick.py --once             one real pick, then stop
    python pick.py                    loop until no `red` remain (owner must stay present)
    python pick.py --class blue       pick a different class (default config.TARGET_CLASS)

Real motion requires calibration.npy and all owner-provided values in config.py, and asks
"Workspace clear? [y/N]" before the first move. Never run unattended.
"""

from __future__ import annotations

import argparse
import math
import time

import cv2

import arm as arm_mod
import config
import detect
import mapping
import runlog


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PickLoop:
    """The loop logic, separated from I/O so it can be unit-tested with fakes."""

    def __init__(self, detector, arm, grab, H, target_class: str = config.TARGET_CLASS,
                 dry_run: bool = False, once: bool = False, max_cycles: int = 20,
                 show=None):
        self.det = detector
        self.arm = arm
        self.grab = grab          # () -> BGR frame (always a FRESH frame)
        self.H = H
        self.target_class = target_class
        self.dry_run = dry_run
        self.once = once
        self.max_cycles = max_cycles
        self.show = show          # optional callback(frame_with_drawing)
        self.log = runlog.get_logger()
        self.ignored: list[tuple[float, float]] = []  # pixel spots skipped for the rest of the session
        self.picks = 0
        self.table_z = config.require("TABLE_Z_MM")
        self.hover = config.HOVER_OFFSET_MM
        self.drop = config.require("DROP_XYZ_MM")

    # -- helpers ----------------------------------------------------------------
    def _is_ignored(self, d: detect.Detection) -> bool:
        return any(_dist((d.cx, d.cy), spot) <= config.LIFT_FAIL_RADIUS_PX for spot in self.ignored)

    def _log_detections(self, dets, tag: str) -> None:
        for d in dets:
            x, y = mapping.pixel_to_arm(d.cx, d.cy, self.H)
            self.log.info("det %s %s -> arm=(%.1f, %.1f)", tag, d, x, y)
        n = sum(d.cls == self.target_class for d in dets)
        self.log.info("%s: %d detections, %d %s", tag, len(dets), n, self.target_class)

    def _annotate(self, frame, dets, target: detect.Detection | None, x: float | None, y: float | None):
        vis = detect.draw(frame, dets, self.target_class)
        if target is not None:
            cv2.circle(vis, (int(target.cx), int(target.cy)), 14, (255, 255, 255), 2)
            cv2.putText(vis, f"pick -> arm ({x:.0f}, {y:.0f}, {self.table_z:.0f})",
                        (int(target.cx) + 16, int(target.cy) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        for spot in self.ignored:
            cv2.drawMarker(vis, (int(spot[0]), int(spot[1])), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        return vis

    def _lift_failed(self, target: detect.Detection) -> bool:
        """Re-detect; True if a target-class block is still at the pick spot."""
        frame = self.grab()
        dets = self.det.detect(frame)
        self._log_detections(dets, "lift-check")
        for d in dets:
            if d.cls == self.target_class and _dist((d.cx, d.cy), (target.cx, target.cy)) <= config.LIFT_FAIL_RADIUS_PX:
                runlog.save_frame(self._annotate(frame, dets, d, *mapping.pixel_to_arm(d.cx, d.cy, self.H)),
                                  f"pick{self.picks + 1:03d}-liftfail")
                return True
        return False

    # -- one cycle --------------------------------------------------------------
    def cycle(self, n: int) -> str:
        """Returns 'picked' | 'skipped' | 'done'."""
        frame = self.grab()
        dets = self.det.detect(frame)
        self._log_detections(dets, f"cycle{n}")
        candidates = [d for d in dets if d.cls == self.target_class and not self._is_ignored(d)]
        if not candidates:
            self.log.info("no %s detected (%d ignored spots) -> done", self.target_class, len(self.ignored))
            if self.show:
                self.show(self._annotate(frame, dets, None, None, None))
            return "done"
        target = max(candidates, key=lambda d: d.conf)
        x, y = mapping.pixel_to_arm(target.cx, target.cy, self.H)
        vis = self._annotate(frame, dets, target, x, y)
        path = runlog.save_frame(vis, f"cycle{n:03d}-target")
        self.log.info("target %s -> arm=(%.1f, %.1f) z=%.1f frame=%s", target, x, y, self.table_z, path)
        if self.show:
            self.show(vis)

        try:
            arm_mod.check_target(x, y, self.table_z)
        except arm_mod.UnsafeTarget as e:
            self.log.warning("skip: target outside reach (%s); ignoring this spot for the session", e)
            self.ignored.append((target.cx, target.cy))
            return "skipped"

        if self.dry_run:
            self.log.info("[dry-run] would pick at (%.1f, %.1f, %.1f) and drop at %s", x, y, self.table_z, self.drop)

        a = self.arm
        hover_z = self.table_z + self.hover
        dx, dy, dz = self.drop
        a.move_to(x, y, hover_z)
        a.move_to(x, y, self.table_z)
        a.suction(True)
        a.wait(config.SUCTION_ON_PAUSE_S)
        a.move_to(x, y, hover_z)
        a.move_to(dx, dy, dz + self.hover)   # clear of the pick spot before checking the lift
        if not self.dry_run and self._lift_failed(target):
            self.log.warning("lift failed at %s; releasing, ignoring this spot for the session", target)
            a.suction(False)
            self.ignored.append((target.cx, target.cy))
            a.home()
            return "skipped"
        a.move_to(dx, dy, dz)
        a.suction(False)
        a.wait(config.SUCTION_OFF_PAUSE_S)
        a.move_to(dx, dy, dz + self.hover)
        a.home()
        self.picks += 1
        self.log.info("pick %d complete", self.picks)
        return "picked"

    def run(self) -> int:
        self.arm.home()
        for n in range(1, self.max_cycles + 1):
            result = self.cycle(n)
            if result == "done":
                break
            if result == "picked" and self.once:
                self.log.info("--once: stopping after one pick")
                break
        else:
            self.log.warning("reached max cycles (%d); stopping", self.max_cycles)
        self.log.info("finished: %d picks, %d ignored spots", self.picks, len(self.ignored))
        return self.picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="log + draw planned targets; no serial writes")
    ap.add_argument("--once", action="store_true", help="stop after one successful pick")
    ap.add_argument("--class", dest="target_class", default=config.TARGET_CLASS)
    ap.add_argument("--camera", type=int, default=config.WEBCAM_INDEX)
    ap.add_argument("--conf", type=float, default=config.CONFIDENCE)
    ap.add_argument("--max-cycles", type=int, default=20)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    log = runlog.start_run("pick-dry" if args.dry_run else "pick")
    missing = config.missing_owner_values()
    if missing:
        raise SystemExit(f"config.py still has owner-provided values unset: {missing}. "
                         f"Fill them in (block-picker-plan.md, 'Owner-provided values') before running pick.py.")
    H = mapping.load_homography()  # raises CalibrationMissing with instructions if absent
    det = detect.Detector(confidence=args.conf)
    log.info("model=%s classes=%s target=%r dry_run=%s once=%s", det.model_id, det.class_names,
             args.target_class, args.dry_run, args.once)
    if args.target_class not in det.class_names:
        raise SystemExit(f"class {args.target_class!r} is not one of the model's classes {det.class_names}")

    cap = detect.open_camera(args.camera)
    show = None
    if not args.no_window:
        def show(vis):
            cv2.imshow("pick", vis)
            cv2.waitKey(1)
    a = arm_mod.Arm(dry_run=args.dry_run).connect()
    try:
        loop = PickLoop(det, a, lambda: detect.grab(cap), H, target_class=args.target_class,
                        dry_run=args.dry_run, once=args.once, max_cycles=args.max_cycles, show=show)
        loop.run()
        if show and args.dry_run:
            log.info("dry-run finished; press any key in the window to close")
            cv2.waitKey(0)
    finally:
        try:
            if not args.dry_run:
                a.suction(False)
                a.home()
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            log.error("cleanup: %s", e)
        a.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
