"""Main pick loop: detect -> map -> pick -> drop -> home -> re-detect, until no target blocks remain.

    python pick.py --dry-run          no serial writes; logs a planned target for each visible block once
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

    def __init__(self, detector, arm, grab, H, target_class: str | None = None,
                 dry_run: bool = False, once: bool = False, max_cycles: int = 20,
                 show=None):
        self.det = detector
        self.arm = arm
        self.grab = grab          # () -> BGR frame (always a FRESH frame)
        self.H = H
        self.target_class = target_class or config.TARGET_CLASS
        self.dry_run = dry_run
        self.once = once
        self.max_cycles = max_cycles
        self.show = show          # optional callback(frame_with_drawing)
        self.log = runlog.get_logger()
        self.ignored: list[tuple[float, float]] = []  # pixel spots skipped for the rest of the session
        self.planned: list[tuple[float, float]] = []  # dry-run: spots already given a planned target
        self.picks = 0
        # live status for the UI panel
        self.state = "starting"
        self.remaining = 0
        self.others = 0
        self.cycle_no = 0
        self.t0 = time.time()
        self.last_vis = None
        self.last_event = ""
        self.table_z = config.require("TABLE_Z_MM")
        self.pick_z = self.table_z + config.BLOCK_HEIGHT_MM - config.CUP_PRESS_MM  # cup on the block's top
        self.hover = config.HOVER_OFFSET_MM
        self.travel_z = config.TRAVEL_Z_MM
        self.drop = config.require("DROP_XYZ_MM")

    # -- helpers ----------------------------------------------------------------
    def _is_ignored(self, d: detect.Detection) -> bool:
        return any(_dist((d.cx, d.cy), spot) <= config.LIFT_FAIL_RADIUS_PX
                   for spot in self.ignored + self.planned)

    def _log_detections(self, dets, tag: str) -> None:
        for d in dets:
            x, y = mapping.pixel_to_arm(d.cx, d.cy, self.H)
            self.log.info("det %s %s -> arm=(%.1f, %.1f)", tag, d, x, y)
        n = sum(detect.is_target(d, self.target_class) for d in dets)
        self.log.info("%s: %d detections, %d %s", tag, len(dets), n, self.target_class)

    def _panel(self, vis):
        """Draw the status panel (bottom-left) on a copy of vis."""
        vis = vis.copy()
        h, w = vis.shape[:2]
        lines = [
            (f"{self.target_class.upper()} LEFT: {self.remaining}", (80, 80, 255) if self.remaining else (80, 220, 80)),
            (f"other blocks: {self.others}    picked: {self.picks}    skipped: {len(self.ignored)}", (230, 230, 230)),
            (f"{'DRY RUN  ' if self.dry_run else ''}{self.state}", (0, 200, 255)),
            (f"cycle {self.cycle_no}   {int(time.time() - self.t0)}s   {self.last_event}", (180, 180, 180)),
        ]
        ph = 34 * len(lines) + 20
        cv2.rectangle(vis, (0, h - ph), (min(w, 760), h), (20, 20, 20), -1)
        for i, (text, colour) in enumerate(lines):
            cv2.putText(vis, text, (14, h - ph + 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.85 if i == 0 else 0.7,
                        colour, 2)
        return vis

    def _render(self, state: str | None = None) -> None:
        if state is not None:
            self.state = state
        if self.show and self.last_vis is not None:
            self.show(self._panel(self.last_vis))

    def _annotate(self, frame, dets, target: detect.Detection | None, x: float | None, y: float | None):
        vis = detect.draw(frame, dets, self.target_class)
        if target is not None:
            cv2.circle(vis, (int(target.cx), int(target.cy)), 14, (255, 255, 255), 2)
            cv2.putText(vis, f"pick -> arm ({x:.0f}, {y:.0f}, {self.pick_z:.0f})",
                        (int(target.cx) + 16, int(target.cy) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        for spot in self.ignored:
            cv2.drawMarker(vis, (int(spot[0]), int(spot[1])), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        for spot in self.planned:
            cv2.drawMarker(vis, (int(spot[0]), int(spot[1])), (255, 255, 255), cv2.MARKER_SQUARE, 20, 1)
        return vis

    def _lift_failed(self, target: detect.Detection) -> bool:
        """Re-detect; True if a target-class block is still at the pick spot."""
        frame = self.grab()
        dets = self.det.detect(frame)
        self._log_detections(dets, "lift-check")
        for d in dets:
            if detect.is_target(d, self.target_class) and _dist((d.cx, d.cy), (target.cx, target.cy)) <= config.LIFT_FAIL_RADIUS_PX:
                runlog.save_frame(self._annotate(frame, dets, d, *mapping.pixel_to_arm(d.cx, d.cy, self.H)),
                                  f"pick{self.picks + 1:03d}-liftfail")
                return True
        return False

    # -- one cycle --------------------------------------------------------------
    def cycle(self, n: int) -> str:
        """Returns 'picked' | 'skipped' | 'done'."""
        self.cycle_no = n
        self._render("scanning")
        frame = self.grab()
        dets = self.det.detect(frame)
        self._log_detections(dets, f"cycle{n}")
        candidates = [d for d in dets if detect.is_target(d, self.target_class) and not self._is_ignored(d)]
        self.remaining = len(candidates)
        self.others = sum(not detect.is_target(d, self.target_class) for d in dets)
        if not candidates:
            self.log.info("no %s left to pick (%d ignored spots, %d planned) -> done",
                          self.target_class, len(self.ignored), len(self.planned))
            self.last_vis = self._annotate(frame, dets, None, None, None)
            self._render("DONE - table clear of " + self.target_class)
            return "done"
        target = max(candidates, key=lambda d: d.conf)
        x, y = mapping.pixel_to_arm(target.cx, target.cy, self.H)
        x, y = x + config.PICK_OFFSET_MM[0], y + config.PICK_OFFSET_MM[1]  # (0, 0) unless a hole sits under the cup
        vis = self._annotate(frame, dets, target, x, y)
        path = runlog.save_frame(vis, f"cycle{n:03d}-target")
        self.log.info("target %s -> arm=(%.1f, %.1f) pick z=%.1f frame=%s", target, x, y, self.pick_z, path)
        self.last_vis = vis
        self.last_event = f"target {target.cls} {target.conf:.2f} -> ({x:.0f}, {y:.0f})"
        self._render("moving to block")

        try:
            arm_mod.check_target(x, y, self.pick_z)
        except arm_mod.UnsafeTarget as e:
            self.log.warning("skip: target outside reach (%s); ignoring this spot for the session", e)
            self.ignored.append((target.cx, target.cy))
            return "skipped"

        spot = (target.cx, target.cy)
        if self.dry_run:
            self.log.info("[dry-run] would pick at (%.1f, %.1f, %.1f) and drop at %s", x, y, self.pick_z, self.drop)
            self.planned.append(spot)

        a = self.arm
        hover_z = self.pick_z + self.hover
        tz = self.travel_z
        dx, dy, dz = self.drop
        at_drop = False
        carrying = False
        try:
            # Sideways moves only at travel height; vertical moves only above the spot.
            a.move_to(x, y, tz)
            self._render("descending")
            a.move_to(x, y, hover_z)
            a.move_to(x, y, self.pick_z, 700)  # slow final descent onto the block
            self._render("suction on")
            a.suction(True)
            carrying = True
            a.wait(config.SUCTION_ON_PAUSE_S)
            self._render("lifting")
            a.move_to(x, y, hover_z)
            a.move_to(x, y, tz)
            at_drop = True
            self._render("carrying to bin")
            a.move_to(dx, dy, tz)                # clear of the pick spot before checking the lift
            self._render("checking the lift")
            if not self.dry_run and self._lift_failed(target):
                self.log.warning("lift failed at %s; releasing, ignoring this spot for the session", target)
                self.last_event = "lift FAILED - spot skipped"
                self._render("lift failed - releasing")
                a.suction(False)
                self.ignored.append(spot)
                a.home()
                return "skipped"
            self._render("dropping")
            a.move_to(dx, dy, dz + self.hover)
            a.move_to(dx, dy, dz)
            a.vent()                             # valve stays OPEN while we lift away, so the cup lets go
            a.wait(config.VENT_S)
            a.move_to(dx, dy, dz + self.hover)
            a.valve_close()
            a.wait(config.SUCTION_OFF_PAUSE_S)
            a.move_to(dx, dy, tz)
            self._render("homing")
            a.home()
        except arm_mod.MoveRefused as e:
            # The arm did not reach a target (silently refused by its IK, or stalled). Never continue
            # the sequence blind: set the block down if we can, vent, park, and do not retry the spot.
            self.log.error("move refused (%s); releasing, homing, ignoring spot %s", e, spot)
            self.last_event = "move refused - spot skipped"
            self._render("move refused - releasing")
            if carrying and not self.dry_run:
                pos = a.read_xyz()
                if pos is not None and _dist(pos, (x, y)) <= arm_mod.POSITION_TOLERANCE_MM and pos[2] > self.pick_z + 5:
                    try:  # still in the pick column: lower the block back onto its spot instead of dropping it
                        a.move_to(x, y, self.pick_z, 900)
                    except (arm_mod.UnsafeTarget, arm_mod.MoveRefused) as e2:
                        self.log.warning("could not lower the block before venting (%s); venting in place", e2)
            a.suction(False)
            self.ignored.append(spot)
            a.home()
            if at_drop:
                raise arm_mod.ArmError(f"drop-zone pose {self.drop} (+hover) refused by the arm; "
                                       f"fix config.DROP_XYZ_MM") from e
            return "skipped"
        self.picks += 1
        self.remaining = max(0, self.remaining - 1)
        self.last_event = f"pick {self.picks} done"
        self.log.info("%s %d complete", "simulated pick" if self.dry_run else "pick", self.picks)
        self._render("rescanning")
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
        self.log.info("finished: %d %s, %d ignored spots", self.picks,
                      "simulated picks" if self.dry_run else "picks", len(self.ignored))
        print(f"\n=== {'DRY RUN ' if self.dry_run else ''}FINISHED in {int(time.time() - self.t0)}s: "
              f"{self.picks} {self.target_class} block(s) picked, {len(self.ignored)} skipped, "
              f"{self.remaining} left ===")
        return self.picks


def check_fixed_targets() -> None:
    """Validate home, drop, drop hover and pick hover against the reach envelope before anything moves.

    Otherwise a bad DROP_XYZ_MM is only discovered mid-cycle, with a block on the cup.
    """
    hover = config.HOVER_OFFSET_MM
    tz = config.TRAVEL_Z_MM
    dx, dy, dz = config.require("DROP_XYZ_MM")
    hx, hy, hz = config.require("HOME_XYZ_MM")
    table_z = config.require("TABLE_Z_MM")
    log = runlog.get_logger()
    block = config.BLOCK_HEIGHT_MM
    min_travel = table_z + 2 * block + 10.0  # a carried block must clear a block on the table
    if tz < min_travel:
        raise SystemExit(f"config.TRAVEL_Z_MM={tz} is too low: needs >= TABLE_Z + 2 x BLOCK_HEIGHT + 10 = {min_travel:.0f}")
    pick_z = table_z + block - config.CUP_PRESS_MM
    for name, p in (("HOME_XYZ_MM", (hx, hy, hz)), ("DROP_XYZ_MM", (dx, dy, dz)),
                    ("drop hover", (dx, dy, dz + hover)), ("drop travel", (dx, dy, tz))):
        ext = arm_mod.extension_ratio(*p)
        if ext > arm_mod.EXTENSION_WARN:
            log.warning("%s %s needs %.0f%% of the arm's full stretch; the firmware may refuse it - "
                        "bring it closer to the base or lower it", name, p, ext * 100)
    try:
        arm_mod.check_target(hx, hy, hz)
        arm_mod.check_target(dx, dy, dz)
        arm_mod.check_target(dx, dy, dz + hover)
        arm_mod.check_target(dx, dy, tz)
        lo, hi = config.require("REACH_Z_MM")
        if not lo <= tz <= hi:
            raise arm_mod.UnsafeTarget(f"travel Z {tz:.1f} outside reach Z [{lo}, {hi}]")
        if not lo <= pick_z + hover <= hi:
            raise arm_mod.UnsafeTarget(f"pick hover Z {pick_z + hover:.1f} outside reach Z [{lo}, {hi}]")
    except arm_mod.UnsafeTarget as e:
        raise SystemExit(f"config.py HOME_XYZ_MM / DROP_XYZ_MM / TABLE_Z_MM+HOVER_OFFSET_MM outside REACH_*: {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="log + draw planned targets; no serial writes")
    ap.add_argument("--once", action="store_true", help="stop after one successful pick")
    ap.add_argument("--class", dest="target_class", default=config.TARGET_CLASS)
    ap.add_argument("--camera", type=int, default=None, help="default config.WEBCAM_INDEX")
    ap.add_argument("--conf", type=float, default=None, help="default config.CONFIDENCE")
    ap.add_argument("--max-cycles", type=int, default=20)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    log = runlog.start_run("pick-dry" if args.dry_run else "pick")
    missing = config.missing_owner_values()
    if missing:
        raise SystemExit(f"config.py still has owner-provided values unset: {missing}. "
                         f"Fill them in (block-picker-plan.md, 'Owner-provided values') before running pick.py.")
    check_fixed_targets()
    cal = mapping.load_calibration()  # raises CalibrationMissing with instructions if absent
    H = cal["H"]
    det = detect.Detector(confidence=args.conf)
    log.info("model=%s classes=%s target=%r dry_run=%s once=%s", det.model_id, det.class_names,
             args.target_class, args.dry_run, args.once)
    if args.target_class != "any" and args.target_class not in det.class_names:
        raise SystemExit(f"class {args.target_class!r} is not one of the model's classes {det.class_names}")

    camera = config.WEBCAM_INDEX if args.camera is None else args.camera
    cap = detect.open_camera(camera)
    mapping.check_frame_size(cal, detect.grab(cap))
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
        if show:
            log.info("finished; press any key in the window to close")
            cv2.waitKey(0)
    finally:
        try:
            # Park only if motion was already authorised this session; never re-prompt from cleanup
            # (the owner may have just declined or hit Ctrl-C at the prompt).
            if not args.dry_run and a.cleared:
                a.suction(False)
                a.home()
        except BaseException as e:  # noqa: BLE001 - a second Ctrl-C must still release the port
            log.error("cleanup: %r", e)
        finally:
            a.close()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
