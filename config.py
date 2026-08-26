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
# Owner's defect model (project block-defects-9kz1n, version 2, rfdetr-large, classes Defect / Good),
# switched in 2026-08-26. The earlier placeholder was "aarnavs-space/blocks-lllea-p7f2h-1-rfdetr-small-t1".
# `inference.get_model` accepts the slug as-is; do NOT append "/2".
MODEL_ID = "aarnavs-space/block-defects-9kz1n-2-rfdetr-large-t1"
API_KEY_ENV = "ROBOFLOW_API_KEY"
# Exact class string as the model reports it (this model: "Defect", "Good"). Only this class is picked;
# "any" = any block-sized detection. `pick.py --class X` overrides. Calibration always uses any block.
TARGET_CLASS = "Defect"
# Plausible block size in pixels (blocks are ~200-390 px in the current mounted view, a block on its long
# side is the tallest). Detections outside this range are dropped (laptop, hands, the mat).
MIN_BOX_PX = 100
MAX_BOX_PX = 450
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
# UVC controls applied every time the camera is opened (via tools/uvc-util; macOS ignores OpenCV's exposure
# settings). This webcam has no exposure-time/gain control. What actually lowers its exposure is
# backlight-compensation=4 (auto-exposure meters for the bright blocks instead of the black mat: block-top
# clipping 14% -> 2%); brightness is a post-sensor offset that only darkens. Measured 2026-08-26.
CAMERA_UVC_NAME = "HD Web Camera"
CAMERA_CONTROLS = {"backlight-compensation": 4, "brightness": 112, "contrast": 128, "auto-exposure-mode": 2}

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
# Blocks are 4x4 cm (Hiwonder kit). The cup grabs the TOP of a block sitting on the table, so the pick
# height is TABLE_Z + BLOCK_HEIGHT - CUP_PRESS (the rubber cup squashes a few mm to seal).
BLOCK_HEIGHT_MM = 42.0  # owner: 50 left the cup above the block (2026-08-26); 42 = 5 mm of cup press on a 40 mm block
CUP_PRESS_MM = 5.0
# Pick point offset from the detected block centre, in arm mm (x, y). Use when the defect (a drilled hole)
# sits in the middle of the top face: a hole under the cup leaks the vacuum. (0, 0) = pick at the centre.
PICK_OFFSET_MM = (0.0, 0.0)
# Hover height above the pick / drop point (mm): the slow, vertical final approach.
HOVER_OFFSET_MM = 40.0
# ALL sideways travel happens at this absolute Z (mm). Must clear a 40 mm block on the table while the
# cup is carrying another 40 mm block: table Z + 40 + 40 + margin. Owner can raise it if blocks are taller.
TRAVEL_Z_MM = 160.0
# Owner-provided: drop zone end-effector position (mm).
DROP_XYZ_MM: tuple[float, float, float] | None = (-220.0, 13.0, 165.0)  # max: hover must stay <= REACH_Z  # verified reachable incl. hover, 2026-08-25
# Owner-provided: home pose, out of camera view, clear of the pick area (mm).
HOME_XYZ_MM: tuple[float, float, float] | None = (-210.0, -56.0, 190.0)
# Owner-provided: reach limits the arm may be sent to, (min, max) per axis in mm (envelope visited in --jog).
REACH_X_MM: tuple[float, float] | None = (-269.0, 272.0)
REACH_Y_MM: tuple[float, float] | None = (-256.0, 24.0)
REACH_Z_MM: tuple[float, float] | None = (47.0, 210.0)
# Never send the cup within this horizontal radius of the arm's own base (the base is in the camera's view
# and can be detected as a block). The firmware refuses < 50 mm; the base structure is wider than that.
MIN_RADIUS_MM = 120.0

# Suction settle times (s) from the plan's pick sequence.
SUCTION_ON_PAUSE_S = 1.0  # let the vacuum build before lifting (porous wood, holes under the cup)
SUCTION_OFF_PAUSE_S = 0.3
# How long the vent valve stays open when releasing (s). Taped / smooth tops seal well and need longer
# than the docs' 0.2 s or the block stays on the cup.
VENT_S = 1.0
# When the arm sets a block down (calibration carry mode), release this much ABOVE the pick height so the
# cup is not pressed into the block when it vents (a squashed cup drags the block back up).
RELEASE_LIFT_MM = 25.0

# --------------------------------------------------------------------------
# Calibration / logging
# --------------------------------------------------------------------------
CALIBRATION_PATH = "calibration.npy"
# Arm (x, y) mm spots for `calibrate.py --auto`. Keep them inside the camera's view: on 2026-08-25 the
# view covered arm x from about -150 to +60 (x = +151 sat on the top edge and was clipped) and y down
# to about -240. Spots the arm cannot reach are skipped automatically.
CALIB_GRID = [(x, y) for y in (-120.0, -160.0, -200.0, -235.0) for x in (-150.0, -85.0, -20.0, 45.0)]
# Calibration spots closer than this to DROP_XYZ_MM are skipped (the bin is there), and the run starts
# at the spot farthest from the bin.
DROP_KEEPOUT_MM = 170.0
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
    """ROBOFLOW_API_KEY from the environment, else from .claude/settings.local.json (gitignored)."""
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".claude", "settings.local.json")
    try:
        import json

        with open(local) as f:
            key = json.load(f).get("env", {}).get(API_KEY_ENV)
    except (OSError, ValueError):
        key = None
    if key:
        return key
    raise ConfigError(f"{API_KEY_ENV} is not set: export it, or put it under env.{API_KEY_ENV} in {local}")


def missing_owner_values() -> list[str]:
    """Names of owner-provided values still unset (for status messages)."""
    names = ["TABLE_Z_MM", "DROP_XYZ_MM", "HOME_XYZ_MM", "REACH_X_MM", "REACH_Y_MM", "REACH_Z_MM"]
    return [n for n in names if globals().get(n) is None]
