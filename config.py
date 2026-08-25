"""Block Picker configuration.

Every physical / environment value lives here and nowhere else.
Values that are ``None`` are OWNER-PROVIDED and still TODO: any code path
that needs one calls ``require(...)`` and fails loudly instead of guessing.
See block-picker-plan.md -> "Owner-provided values".
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Roboflow detection
# --------------------------------------------------------------------------
# Trained model (workspace/model-slug). Version 1 confirmed on the dashboard
# via the API on 2026-08-25 (project aarnavs-space/blocks-lllea-p7f2h, rfdetr-small).
# NOTE: `inference.get_model` accepts this slug as-is, or "blocks-lllea-p7f2h/1"; appending "/1"
# to the slug is rejected as an invalid model id (verified 2026-08-25).
MODEL_ID = "aarnavs-space/blocks-lllea-p7f2h-1-rfdetr-small-t1"
API_KEY_ENV = "ROBOFLOW_API_KEY"
# Exact class string as it appears on the dashboard (classes: blue, green, red, yellow).
TARGET_CLASS = "red"
CONFIDENCE = 0.5  # minimum detection confidence
# Optional pick-area region of interest (x, y, w, h) in FULL-FRAME pixels. Detection runs on
# this crop and centres are mapped back to full-frame pixels, so calibration is unaffected.
# Set it once the camera is mounted: the model was trained with blocks large in frame and
# mislabels small blocks when fed the whole 1080p frame (measured 2026-08-25). None = full frame.
PICK_ROI: tuple[int, int, int, int] | None = None

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
# Probed 2026-08-25: index 0 = external "HD Web Camera" (1920x1080), index 1 = built-in.
WEBCAM_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --------------------------------------------------------------------------
# Arm (Hiwonder MaxArm, ESP32 controller over USB serial)
# --------------------------------------------------------------------------
# CH340 USB-serial of the arm. The number changes with the USB socket (-310 in the connect-4
# project, -3110 on 2026-08-25); arm.py falls back to a lone /dev/cu.usbserial-* if this is absent.
SERIAL_PORT = "/dev/cu.usbserial-3110"
BAUDRATE = 9600  # MaxArm_micropython_microUSB firmware: UART(1, 9600, tx=1, rx=3); validated 2026-08-17
SERIAL_TIMEOUT_S = 1.0
MOVE_TIMEOUT_S = 6.0  # maximum allowed move duration (+ settle); longer moves are refused, never truncated

# Default duration for a move (ms). Longer = gentler.
MOVE_MS = 1000

# Owner-provided (jogged 2026-08-25): arm Z (mm) where the suction cup just touches the table.
TABLE_Z_MM: float | None = 47.0  # measured with arm.py --jog, 2026-08-25
# Hover height above table / drop zone (mm).
HOVER_OFFSET_MM = 40.0
# Owner-provided: drop zone end-effector position (mm).
DROP_XYZ_MM: tuple[float, float, float] | None = (-220.0, 13.0, 120.0)  # verified reachable incl. hover, 2026-08-25
# Owner-provided: home pose, out of camera view, clear of the pick area (mm).
HOME_XYZ_MM: tuple[float, float, float] | None = (-210.0, -56.0, 139.0)
# Owner-provided: reach limits the arm may be sent to, (min, max) per axis in mm (envelope visited in --jog).
REACH_X_MM: tuple[float, float] | None = (-269.0, 272.0)
REACH_Y_MM: tuple[float, float] | None = (-256.0, 24.0)
REACH_Z_MM: tuple[float, float] | None = (47.0, 210.0)

# Suction settle times (s) from the plan's pick sequence.
SUCTION_ON_PAUSE_S = 0.5
SUCTION_OFF_PAUSE_S = 0.3

# --------------------------------------------------------------------------
# Calibration / logging
# --------------------------------------------------------------------------
CALIBRATION_PATH = "calibration.npy"
LOG_DIR = "logs"
# "Lift failed" check: same-class detection still within this many px of the pick spot.
LIFT_FAIL_RADIUS_PX = 10.0


class ConfigError(RuntimeError):
    """A required owner-provided value is still TODO."""


def require(name: str):
    """Return config value ``name`` or raise ConfigError if it is still None."""
    value = globals().get(name)
    if value is None:
        raise ConfigError(
            f"config.{name} is not set. This is an owner-provided value "
            f"(see block-picker-plan.md, 'Owner-provided values'). Fill it in config.py."
        )
    return value


def api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise ConfigError(f"Environment variable {API_KEY_ENV} is not set.")
    return key


def missing_owner_values() -> list[str]:
    """Names of owner-provided values still unset (for status messages)."""
    names = ["TABLE_Z_MM", "DROP_XYZ_MM", "HOME_XYZ_MM", "REACH_X_MM", "REACH_Y_MM", "REACH_Z_MM"]
    return [n for n in names if globals().get(n) is None]
