"""Per-run timestamped logging: logs/YYYYMMDD-HHMMSS.log plus a frames folder next to it."""

from __future__ import annotations

import logging
import os
import sys
import time

import cv2

import config

_run_stamp: str | None = None
_frames_dir: str | None = None


def start_run(name: str = "run", level: int = logging.INFO) -> logging.Logger:
    """Create logs/<stamp>.log (and logs/<stamp>-frames/) and return a configured logger."""
    global _run_stamp, _frames_dir
    os.makedirs(config.LOG_DIR, exist_ok=True)
    _run_stamp = time.strftime("%Y%m%d-%H%M%S")
    _frames_dir = os.path.join(config.LOG_DIR, f"{_run_stamp}-frames")
    log_path = os.path.join(config.LOG_DIR, f"{_run_stamp}.log")

    logger = logging.getLogger("blockpicker")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info("run=%s name=%s log=%s", _run_stamp, name, log_path)
    return logger


def get_logger() -> logging.Logger:
    logger = logging.getLogger("blockpicker")
    if not logger.handlers:
        return start_run()
    return logger


def save_frame(frame, tag: str) -> str:
    """Save a frame as JPEG in this run's frames folder; returns the path."""
    global _frames_dir
    if _frames_dir is None:
        start_run()
    os.makedirs(_frames_dir, exist_ok=True)
    path = os.path.join(_frames_dir, f"{time.strftime('%H%M%S')}-{tag}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path
