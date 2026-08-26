"""Apply UVC camera controls (brightness etc.) with the bundled uvc-util (macOS only).

OpenCV's AVFoundation backend ignores exposure/brightness settings, so we talk to the webcam directly over
USB. Works while the stream is open. Missing tool or unsupported control = a logged warning, never a crash.
"""

from __future__ import annotations

import os
import subprocess

import config
import runlog

TOOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "uvc-util")
TOOL = os.path.join(TOOL_DIR, "uvc-util")


def _tool() -> str | None:
    if os.path.exists(TOOL):
        return TOOL
    build = os.path.join(TOOL_DIR, "build.sh")
    if os.path.exists(build):
        r = subprocess.run(["/bin/sh", build], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(TOOL):
            return TOOL
    return None


def _select_args() -> list[str]:
    return ["-N", config.CAMERA_UVC_NAME] if config.CAMERA_UVC_NAME else ["-I", "0"]


def get(control: str) -> str | None:
    tool = _tool()
    if tool is None:
        return None
    r = subprocess.run([tool, *_select_args(), "-o", control], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def apply(controls: dict | None = None) -> bool:
    """Set each control; returns True if every one was accepted."""
    log = runlog.get_logger()
    controls = config.CAMERA_CONTROLS if controls is None else controls
    if not controls:
        return True
    tool = _tool()
    if tool is None:
        log.warning("camera: tools/uvc-util not available; camera controls %s NOT applied", controls)
        return False
    ok = True
    for name, value in controls.items():
        r = subprocess.run([tool, *_select_args(), "-s", f"{name}={value}"], capture_output=True, text=True)
        if r.returncode != 0 or r.stderr.strip():
            ok = False
            log.warning("camera: could not set %s=%s: %s", name, value, (r.stderr or r.stdout).strip()[:120])
    readback = {name: get(name) for name in controls}
    log.info("camera controls: %s", readback)
    return ok
