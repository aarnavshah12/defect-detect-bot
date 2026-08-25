"""Thin laptop-side wrapper over the Hiwonder MaxArm serial protocol.

Source: Hiwonder MaxArm docs, section 10 "MaxArm Serial Communication"
(https://docs.hiwonder.com/projects/MaxArm/en/latest/docs/10.MaxArm_Serial_Communication_formatted.html).
The ESP32 on the arm runs Hiwonder's communication-routine firmware and listens on the micro-USB
serial port at 9600 baud for frames:

    0xAA 0x55 | func | len | data... | check        check = (~(func + len + sum(data))) & 0xFF

    FUNC_SET_XYZ           0x03  data = int16 x, int16 y, int16 z (mm, little-endian) + uint16 ms
    FUNC_SET_SUCTIONNOZZLE 0x07  data = 1 pump on | 2 pump off + valve open (release) | 3 valve close
    FUNC_READ_XYZ          0x13  reply 0xAA 0x55 0x13 0x06 <hhh> check
    FUNC_READ_ANGLE        0x11  reply 0xAA 0x55 0x11 0x06 <hhh> check

Reply frames are checksummed over the WHOLE frame including the 0xAA 0x55 header (firmware
MaxArm_ctl.py: checksum_crc8(0, 0, send_data)); requests exclude the header. parse_frame()
accepts both conventions.

Hardware facts validated on this arm by the owner's connect-4 project (2026-08-17, see
~/Documents/connect 4 bot/maxarm.py): Hiwonder's MaxArm_micropython_microUSB slave firmware is
installed (it opens UART(1, 9600, tx=1, rx=3), i.e. the USB port); a plain port open does NOT reset
the board, only a DTR/RTS pulse does; after power-on/reset the firmware sleeps 10 s then homes
(~3 s) before it answers, so connect() polls read_xyz for up to READY_TIMEOUT_S; servo read-back
agrees with the commanded target to within ~8 mm. The port number changes with the USB socket
(/dev/cu.usbserial-310 then, -3110 now), so a lone /dev/cu.usbserial-* is used as fallback.

The arm's own inverse kinematics does the servo math; we only ever send end-effector (x, y, z).
Motion commands have NO completion acknowledgement, so move_to() waits the commanded duration
(plus a margin) and then reads the position back.

Safety (enforced here, not in comments): every target must be inside config.REACH_* and Z must be
>= config.TABLE_Z_MM, otherwise UnsafeTarget is raised (logged) and nothing is sent. The first real
motion in a process requires the owner to answer "Workspace clear? [y/N]". connect() refuses to
proceed if the arm never answers. The position read-back after a move is the protocol's only
completion signal: if it is missing or off by more than POSITION_TOLERANCE_MM the move is treated
as refused (the firmware silently ignores unreachable targets) and MoveRefused is raised.

CLI (owner tools):
    python arm.py --probe            open the port, try read_xyz at 9600 and 115200, no motion
    python arm.py --read             print the arm's current (x, y, z) and servo angles
    python arm.py --home             move to config.HOME_XYZ_MM
    python arm.py --xyz X Y Z        move to a point (validated)
    python arm.py --jog              keyboard jog; stamp HOME / TABLE_Z / DROP; prints a config.py snippet
    python arm.py --jog --bootstrap  same, before config.py has reach limits: validates against the
                                     firmware's own envelope (50 <= radius <= 300 mm, 0 <= z <= 255) instead

Jog keys:  w/s = y -/+ (away / toward you)   a/d = x -/+   r/f = z up/down   1/2/3 = step 2/10/25 mm
           p = read position   o = suction toggle   h = stamp HOME   t = stamp TABLE_Z (current z)
           k = stamp DROP   g = go to stamped HOME   x = type an absolute "x y z"   q = quit + print snippet
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import select
import struct
import sys
import time

import config
import runlog

HEADER = b"\xAA\x55"
FUNC_SET_ANGLE = 0x01
FUNC_SET_XYZ = 0x03
FUNC_SET_PWMSERVO = 0x05
FUNC_SET_SUCTIONNOZZLE = 0x07
FUNC_READ_ANGLE = 0x11
FUNC_READ_XYZ = 0x13

NOZZLE_PUMP_ON = 1
NOZZLE_PUMP_OFF_VALVE_OPEN = 2
NOZZLE_VALVE_CLOSE = 3

POSITION_TOLERANCE_MM = 10.0  # read-back within this = arrived (servo feedback validated to ~8 mm)
READY_TIMEOUT_S = 25.0        # firmware sleeps 10 s + homes ~3 s after a reset before answering
MIN_MOVE_MS = 200             # shorter = a full-speed jerk
_READ_DELAY_S = 0.1           # the docs' MaxArm_ctl sleeps 0.1 s before reading a reply
_SETTLE_S = 0.3               # margin added to every commanded move duration


class ArmError(RuntimeError):
    pass


class UnsafeTarget(ValueError):
    """Target outside the configured reach limits or below table Z. Never sent to the arm."""


class MoveRefused(ArmError):
    """The arm did not reach the commanded target (silent IK refusal, stall, or no read-back)."""


# ---------------------------------------------------------------- protocol helpers

def checksum(body: bytes) -> int:
    """(func + len + data) negated, low byte — per docs 10.1.5."""
    return (~sum(body)) & 0xFF


def build_frame(func: int, data: bytes = b"") -> bytes:
    body = bytes([func & 0xFF, len(data) & 0xFF]) + bytes(data)
    return HEADER + body + bytes([checksum(body)])


def parse_frame(buf: bytes, expect_func: int) -> bytes | None:
    """Find the first valid frame with function `expect_func` in buf; return its data bytes.

    Accepts the firmware's reply convention (checksum over header+func+len+data) and the request
    convention (func+len+data); they differ by exactly 1 because 0xAA + 0x55 == 0xFF.
    """
    i = 0
    while True:
        i = buf.find(HEADER, i)
        if i < 0 or i + 4 > len(buf):
            return None
        func, length = buf[i + 2], buf[i + 3]
        end = i + 4 + length
        # A truncated candidate (bogus length byte) is skipped like a bad one; keep scanning.
        if end + 1 <= len(buf) and func == expect_func and buf[end] in (checksum(buf[i:end]), checksum(buf[i + 2:end])):
            return bytes(buf[i + 4:end])
        i += 2


def find_port(preferred: str | None) -> str:
    """The configured port if present, else the sole /dev/cu.usbserial-* device."""
    if preferred and os.path.exists(preferred):
        return preferred
    candidates = sorted(glob.glob("/dev/cu.usbserial-*"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ArmError(f"no MaxArm serial port found (looked for {preferred} and /dev/cu.usbserial-*); "
                       f"is the arm plugged in and switched on?")
    raise ArmError(f"several usbserial ports {candidates}; set config.SERIAL_PORT to the arm's")


def set_xyz_frame(x: float, y: float, z: float, ms: int) -> bytes:
    for name, v in (("x", x), ("y", y), ("z", z)):
        if not -32768 <= int(round(v)) <= 32767:
            raise UnsafeTarget(f"{name}={v} does not fit int16 mm")
    if not 0 <= int(ms) <= 65535:
        raise ValueError(f"duration {ms} ms does not fit uint16")
    return build_frame(FUNC_SET_XYZ, struct.pack("<hhhH", int(round(x)), int(round(y)), int(round(z)), int(ms)))


def nozzle_frame(sub: int) -> bytes:
    if sub not in (NOZZLE_PUMP_ON, NOZZLE_PUMP_OFF_VALVE_OPEN, NOZZLE_VALVE_CLOSE):
        raise ValueError(f"nozzle sub-function must be 1, 2 or 3, got {sub}")
    return build_frame(FUNC_SET_SUCTIONNOZZLE, bytes([sub]))


# ---------------------------------------------------------------- safety

BOOTSTRAP_RADIUS_MM = (50.0, 300.0)  # firmware ignores radius < 50; 300 is well beyond the arm's reach
BOOTSTRAP_Z_MM = (0.0, 255.0)        # firmware clamps z to 255


def check_bootstrap(x: float, y: float, z: float) -> None:
    """Envelope used ONLY by `--jog --bootstrap` before config.py has reach limits.

    Small keyboard steps, owner present, firmware limits as the outer bound. Never used by pick.py.
    """
    r = math.hypot(x, y)
    if not BOOTSTRAP_RADIUS_MM[0] <= r <= BOOTSTRAP_RADIUS_MM[1]:
        raise UnsafeTarget(f"radius {r:.0f} outside bootstrap envelope {BOOTSTRAP_RADIUS_MM}")
    if not BOOTSTRAP_Z_MM[0] <= z <= BOOTSTRAP_Z_MM[1]:
        raise UnsafeTarget(f"z={z:.0f} outside bootstrap envelope {BOOTSTRAP_Z_MM}")


def check_target(x: float, y: float, z: float) -> None:
    """Raise UnsafeTarget unless (x, y, z) is inside reach limits and z >= table Z."""
    table_z = config.require("TABLE_Z_MM")
    if z < table_z:
        raise UnsafeTarget(f"z={z:.1f} is below table Z {table_z:.1f}")
    for axis, v in (("X", x), ("Y", y), ("Z", z)):
        lo, hi = config.require(f"REACH_{axis}_MM")
        if not lo <= v <= hi:
            raise UnsafeTarget(f"{axis}={v:.1f} outside reach [{lo}, {hi}]")


def _cooked_input(prompt: str) -> str:
    """input() that works even if the terminal is in cbreak/raw mode (e.g. inside --jog)."""
    try:
        import termios

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
    except Exception:  # noqa: BLE001 - not a tty
        return input(prompt)
    cooked = list(saved)
    cooked[3] |= termios.ICANON | termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, cooked)
    try:
        return input(prompt)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# ---------------------------------------------------------------- arm

class Arm:
    def __init__(self, port: str | None = None, baudrate: int | None = None, dry_run: bool = False,
                 bootstrap: bool = False):
        self.port = port or config.SERIAL_PORT
        self.baudrate = baudrate or config.BAUDRATE
        self.dry_run = dry_run
        self.bootstrap = bootstrap  # jog-only: validate against firmware limits instead of config
        self._ser = None
        self._cleared = False
        self.log = runlog.get_logger()

    # -- connection -------------------------------------------------------------
    def connect(self) -> "Arm":
        if self.dry_run:
            self.log.info("[dry-run] arm: no serial connection (%s)", self.port)
            return self
        import serial  # pyserial

        self.port = find_port(self.port)
        self.log.info("arm: opening %s @ %d", self.port, self.baudrate)
        try:
            # Plain open with pyserial defaults: validated not to reset the board (only a DTR/RTS
            # pulse does). Do not toggle dtr/rts here.
            ser = serial.Serial(self.port, self.baudrate, timeout=config.SERIAL_TIMEOUT_S,
                                write_timeout=config.SERIAL_TIMEOUT_S)
        except serial.SerialException as e:
            raise ArmError(f"cannot open {self.port}: {e}") from e
        time.sleep(0.1)
        ser.reset_input_buffer()
        self._ser = ser
        pos = self.wait_ready()
        if pos is None:
            self.close()
            raise ArmError(f"arm did not answer read_xyz on {self.port} @ {self.baudrate} within "
                           f"{READY_TIMEOUT_S:.0f}s (arm switched off? wrong baud? firmware not "
                           f"MaxArm_micropython_microUSB? wrong /dev/cu.usbserial-* device?)")
        self.log.info("arm: connected, current xyz=%s", pos)
        return self

    @property
    def cleared(self) -> bool:
        """True once the owner has authorised motion this session (or in dry-run)."""
        return self._cleared or self.dry_run

    def wait_ready(self, timeout: float = READY_TIMEOUT_S) -> tuple[int, int, int] | None:
        """Poll read_xyz until the firmware answers (after a reset it sleeps 10 s, then homes)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            pos = self.read_xyz()
            if pos is not None:
                return pos
        return None

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
            self.log.info("arm: closed %s", self.port)

    def __enter__(self) -> "Arm":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    def confirm_workspace_clear(self) -> None:
        """Ask once per process before any real motion. No default yes."""
        if self.dry_run or self._cleared:
            return
        answer = _cooked_input("Workspace clear? Arm is about to move. [y/N] ").strip().lower()
        if answer != "y":
            raise ArmError("owner did not confirm the workspace is clear; aborting before any motion")
        self._cleared = True
        self.log.info("arm: owner confirmed workspace clear")

    # -- low level ------------------------------------------------------------
    def _send(self, frame: bytes) -> None:
        if self._ser is None:
            raise ArmError("arm not connected (call connect())")
        self._ser.write(frame)
        self._ser.flush()

    def _query(self, func: int, reply_len: int) -> bytes | None:
        if self.dry_run:
            return None
        self._ser.reset_input_buffer()
        self._send(build_frame(func))
        time.sleep(_READ_DELAY_S)
        buf = self._ser.read(reply_len)
        data = parse_frame(buf, func)
        if data is None and self._ser.in_waiting:
            buf += self._ser.read(self._ser.in_waiting)
            data = parse_frame(buf, func)
        return data

    def read_xyz(self) -> tuple[int, int, int] | None:
        data = self._query(FUNC_READ_XYZ, 11)
        return struct.unpack("<hhh", data) if data and len(data) == 6 else None

    def read_angles(self) -> tuple[int, int, int] | None:
        data = self._query(FUNC_READ_ANGLE, 11)
        return struct.unpack("<hhh", data) if data and len(data) == 6 else None

    # -- motion ---------------------------------------------------------------
    def move_to(self, x: float, y: float, z: float, ms: int | None = None) -> tuple[int, int, int] | None:
        """Validate, send, block for the commanded duration, then verify arrival via read-back.

        Raises UnsafeTarget (nothing sent), ValueError for a duration outside
        [MIN_MOVE_MS, MOVE_TIMEOUT_S] (nothing sent), or MoveRefused when the read-back is missing or
        farther than POSITION_TOLERANCE_MM from the target.
        """
        ms = config.MOVE_MS if ms is None else int(ms)
        try:
            (check_bootstrap if self.bootstrap else check_target)(x, y, z)
        except UnsafeTarget as e:
            self.log.error("arm: REFUSED target (%.1f, %.1f, %.1f): %s", x, y, z, e)
            raise
        wait_s = ms / 1000.0 + _SETTLE_S
        if ms < MIN_MOVE_MS or wait_s > config.MOVE_TIMEOUT_S:
            raise ValueError(f"duration {ms} ms outside [{MIN_MOVE_MS}, "
                             f"{int((config.MOVE_TIMEOUT_S - _SETTLE_S) * 1000)}] ms (config.MOVE_TIMEOUT_S)")
        if self.dry_run:
            self.log.info("[dry-run] arm: move_to(%.1f, %.1f, %.1f) %d ms", x, y, z, ms)
            return None
        self.confirm_workspace_clear()
        frame = set_xyz_frame(x, y, z, ms)
        self.log.info("arm: move_to(%.1f, %.1f, %.1f) %d ms frame=%s", x, y, z, ms, frame.hex())
        t0 = time.time()
        self._send(frame)
        self.wait(wait_s)  # never truncated: there is no completion ack to time out on
        pos = self.read_xyz()
        if pos is None:
            pos = self.read_xyz()  # one retry: a single dropped reply is not a refused move
        err = None if pos is None else max(abs(pos[0] - x), abs(pos[1] - y), abs(pos[2] - z))
        if pos is None or err > POSITION_TOLERANCE_MM:
            self.log.error("arm: target (%.1f, %.1f, %.1f) NOT reached: read-back %s (err %s mm) after %.2fs "
                           "-> treating as refused", x, y, z, pos, err, time.time() - t0)
            raise MoveRefused(f"target ({x:.0f}, {y:.0f}, {z:.0f}) not reached; read-back {pos}")
        self.log.info("arm: at %s after %.2fs (max axis error %.1f mm)", pos, time.time() - t0, err)
        return pos

    def home(self, ms: int = 1500) -> tuple[int, int, int] | None:
        hx, hy, hz = config.require("HOME_XYZ_MM")
        self.log.info("arm: home")
        return self.move_to(hx, hy, hz, ms)

    def suction(self, on: bool) -> None:
        if self.dry_run:
            self.log.info("[dry-run] arm: suction(%s)", on)
            return
        self.confirm_workspace_clear()
        if on:
            self.log.info("arm: suction ON (pump on)")
            self._send(nozzle_frame(NOZZLE_PUMP_ON))
        else:
            self.log.info("arm: suction OFF (pump off + valve open, then valve close)")
            self._send(nozzle_frame(NOZZLE_PUMP_OFF_VALVE_OPEN))
            self.wait(0.2)  # docs: venting is very short
            self._send(nozzle_frame(NOZZLE_VALVE_CLOSE))

    @staticmethod
    def wait(seconds: float) -> None:
        time.sleep(max(0.0, seconds))


# ---------------------------------------------------------------- CLI

def _probe(port: str) -> None:
    import serial

    port = find_port(port)
    print(f"Probing {port} (no motion is commanded; a plain open does not reset the board).")
    for baud in (9600, 115200):
        try:
            a = Arm(port, baud).connect()
            pos, ang = a.read_xyz(), a.read_angles()
            a.close()
            print(f"  {baud:6d} baud: read_xyz={pos} read_angles={ang}")
        except (ArmError, serial.SerialException) as e:
            print(f"  {baud:6d} baud: {e}")


def _getch(timeout: float = 0.05) -> str | None:
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def _jog(a: Arm) -> None:
    """Keyboard jog (same keys as the connect-4 jog.py). Prints a config.py snippet on quit."""
    import termios
    import tty

    steps = {"1": 2, "2": 10, "3": 25}
    step = 10
    pos = list(a.read_xyz() or (0, 0, 0))
    a.confirm_workspace_clear()  # ask now, in normal terminal mode, not on the first key press
    stamped: dict[str, object] = {}
    lo = list(pos)
    hi = list(pos)
    pumping = False
    print(f"arm at {pos}. step {step} mm. w/s y  a/d x  r/f z  1/2/3 step  p read  o suction  "
          f"h HOME  t TABLE_Z  k DROP  g go-home  x absolute  q quit")
    if a.bootstrap:
        print("BOOTSTRAP envelope in use (config reach limits not set): keep steps small, watch the arm.")

    def track(p):
        for i in range(3):
            lo[i], hi[i] = min(lo[i], p[i]), max(hi[i], p[i])

    def goto(target, ms=400):
        nonlocal pos
        try:
            real = a.move_to(target[0], target[1], target[2], ms)
        except MoveRefused as e:
            real = a.read_xyz()
            print(f"REFUSED {target} (out of the arm's reach): {e}; arm at {real}")
        except (UnsafeTarget, ValueError, config.ConfigError) as e:
            print(f"refused: {e}")
            return
        if real:
            pos = list(real)
            track(pos)
        print(f"at {pos}")

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            k = _getch()
            if k is None:
                continue
            if k in steps:
                step = steps[k]
                print(f"step {step} mm")
            elif k in "wsadrf":
                dx = {"a": -step, "d": step}.get(k, 0)
                dy = {"w": -step, "s": step}.get(k, 0)  # forward (away from the base) is -Y
                dz = {"f": -step, "r": step}.get(k, 0)
                goto([pos[0] + dx, pos[1] + dy, pos[2] + dz])
            elif k == "p":
                real = a.read_xyz()
                print(f"arm reports {real} angles {a.read_angles()}")
                if real:
                    pos = list(real)
            elif k == "o":
                pumping = not pumping
                a.suction(pumping)
                print("suction ON" if pumping else "released")
            elif k == "h":
                stamped["HOME_XYZ_MM"] = tuple(float(v) for v in pos)
                print(f"HOME_XYZ_MM = {stamped['HOME_XYZ_MM']}")
            elif k == "t":
                stamped["TABLE_Z_MM"] = float(pos[2])
                print(f"TABLE_Z_MM = {stamped['TABLE_Z_MM']}  (cup just touching the table here?)")
            elif k == "k":
                stamped["DROP_XYZ_MM"] = tuple(float(v) for v in pos)
                print(f"DROP_XYZ_MM = {stamped['DROP_XYZ_MM']}")
            elif k == "g":
                if "HOME_XYZ_MM" in stamped:
                    goto(list(stamped["HOME_XYZ_MM"]), 1500)
                elif config.HOME_XYZ_MM:
                    goto(list(config.HOME_XYZ_MM), 1500)
                else:
                    print("no HOME stamped yet (h)")
            elif k == "x":
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
                try:
                    parts = [float(v) for v in input("absolute x y z (mm): ").split()]
                    if len(parts) == 3:
                        goto(parts, 1000)
                    else:
                        print("expected three numbers")
                except ValueError as e:
                    print("bad input:", e)
                finally:
                    tty.setcbreak(sys.stdin.fileno())
            elif k == "q":
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        if pumping:
            a.suction(False)
    table_z = stamped.get("TABLE_Z_MM")
    print("\n# --- paste into config.py (visited envelope this session; widen a little if you need to) ---")
    for key in ("TABLE_Z_MM", "DROP_XYZ_MM", "HOME_XYZ_MM"):
        print(f"{key} = {stamped[key]}" if key in stamped else f"# {key}: not stamped this session")
    zmin = table_z if table_z is not None else lo[2]
    print(f"REACH_X_MM = ({lo[0]:.1f}, {hi[0]:.1f})")
    print(f"REACH_Y_MM = ({lo[1]:.1f}, {hi[1]:.1f})")
    print(f"REACH_Z_MM = ({zmin:.1f}, {hi[2]:.1f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="default config.SERIAL_PORT")
    ap.add_argument("--baud", type=int, default=None, help="default config.BAUDRATE")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ap.add_argument("--jog", action="store_true")
    ap.add_argument("--bootstrap", action="store_true",
                    help="with --jog: use the firmware envelope when config.py reach limits are still None")
    args = ap.parse_args()
    if args.bootstrap and not args.jog:
        ap.error("--bootstrap only applies to --jog")
    if args.jog and not args.bootstrap and config.missing_owner_values():
        ap.error(f"config.py still has {config.missing_owner_values()} unset; use --jog --bootstrap to find them")
    runlog.start_run("arm")
    if args.probe:
        _probe(args.port or config.SERIAL_PORT)
        return
    with Arm(args.port, args.baud, bootstrap=args.bootstrap) as a:
        if args.read:
            print("xyz =", a.read_xyz(), "angles =", a.read_angles())
        if args.home:
            print("at", a.home())
        if args.xyz:
            print("at", a.move_to(*args.xyz))
        if args.jog:
            _jog(a)


if __name__ == "__main__":
    sys.exit(main())
