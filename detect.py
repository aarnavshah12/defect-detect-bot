"""Detection: load the Roboflow model once; detect(frame) -> list[Detection].

Standalone:  python detect.py                # live feed from config.WEBCAM_INDEX with boxes drawn
             python detect.py --image f.jpg  # run on a saved frame (writes <f>-boxes.jpg)
Keys in the live window:  q = quit,  s = save current frame + detections to logs/.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config
import runlog

# BGR colours per class for drawing; target class is drawn thicker.
_COLOURS = {
    "Defect": (0, 0, 255),
    "Good": (0, 200, 0),
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 100, 0),
    "yellow": (0, 220, 220),
}


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    cx: float  # bbox centre, FULL-FRAME pixels
    cy: float
    w: float
    h: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (int(self.cx - self.w / 2), int(self.cy - self.h / 2),
                int(self.cx + self.w / 2), int(self.cy + self.h / 2))

    def __str__(self) -> str:
        return f"{self.cls} conf={self.conf:.2f} px=({self.cx:.0f},{self.cy:.0f}) size=({self.w:.0f}x{self.h:.0f})"


class Detector:
    """Wraps the locally-run Roboflow model. Loads once; call .detect(frame) per frame."""

    def __init__(self, model_id: str | None = None, confidence: float | None = None,
                 roi: tuple[int, int, int, int] | None = None, api_key: str | None = None):
        from inference import get_model  # heavy import; keep it here so unit tests can mock Detector

        # Defaults resolve at call time so config edits/overrides are always honoured.
        self.model_id = model_id or config.MODEL_ID
        self.confidence = config.CONFIDENCE if confidence is None else confidence
        self.roi = config.PICK_ROI if roi is None else roi
        t0 = time.time()
        self.model = get_model(model_id=self.model_id, api_key=api_key or config.api_key())
        self.class_names: list[str] = list(getattr(self.model, "class_names", None) or [])
        self.load_seconds = time.time() - t0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the model on a BGR frame (optionally on config.PICK_ROI); centres in full-frame px."""
        x0 = y0 = 0
        img = frame
        if self.roi is not None:
            x0, y0, w, h = self.roi
            img = frame[y0:y0 + h, x0:x0 + w]
            if img.size == 0:
                raise ValueError(f"PICK_ROI {self.roi} is outside the frame {frame.shape[1]}x{frame.shape[0]}")
        result = self.model.infer(img, confidence=self.confidence)[0]
        dets = [
            Detection(p.class_name, float(p.confidence), float(p.x) + x0, float(p.y) + y0,
                      float(p.width), float(p.height))
            for p in result.predictions
        ]
        return dedupe(block_sized(dets))


def block_sized(dets: list[Detection]) -> list[Detection]:
    """Drop detections whose box is not plausibly a block (hands, laptop, mat...)."""
    return [d for d in dets
            if config.MIN_BOX_PX <= d.w <= config.MAX_BOX_PX and config.MIN_BOX_PX <= d.h <= config.MAX_BOX_PX]


def is_target(d: Detection, target_class: str | None = None) -> bool:
    """True if d is a pickable target: its class matches, or target_class is "any"."""
    target_class = target_class or config.TARGET_CLASS
    return target_class == "any" or d.cls == target_class


def iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def dedupe(dets: list[Detection], iou_thresh: float = 0.6) -> list[Detection]:
    """Class-agnostic NMS: one physical block = one detection (highest confidence wins).

    The model sometimes emits two classes for the same box (e.g. blue 0.55 + green 0.50).
    Keeping only the top one means a block is never treated as the target class merely
    because a weaker duplicate said so.
    """
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.conf, reverse=True):
        if all(iou(d, k) < iou_thresh for k in kept):
            kept.append(d)
    return kept


def draw(frame: np.ndarray, dets: list[Detection], target_class: str | None = None,
         roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Return a copy of frame with boxes, labels, and the ROI outline."""
    target_class = target_class or config.TARGET_CLASS
    roi = config.PICK_ROI if roi is None else roi
    out = frame.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 1)
    for d in dets:
        colour = _COLOURS.get(d.cls, (200, 200, 200))
        thick = 3 if is_target(d, target_class) else 1
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
        cv2.circle(out, (int(d.cx), int(d.cy)), 4, colour, -1)
        cv2.putText(out, f"{d.cls} {d.conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def open_camera(index: int | None = None) -> cv2.VideoCapture:
    """Open the webcam at the configured resolution and apply config.CAMERA_CONTROLS (brightness etc.)."""
    import camera_controls

    index = config.WEBCAM_INDEX if index is None else index
    cap = cv2.VideoCapture(index)
    camera_controls.apply()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
        logging.getLogger("blockpicker").warning("camera %d negotiated %dx%d, config asks %dx%d",
                                                 index, w, h, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    return cap


def grab(cap: cv2.VideoCapture, flush: int = 3) -> np.ndarray:
    """Read a FRESH frame (drop a few buffered ones first)."""
    frame = None
    for _ in range(flush + 1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
    return frame


def _parse_roi(s: str | None):
    if not s:
        return config.PICK_ROI
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi expects x,y,w,h")
    return tuple(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="run on a saved frame instead of the camera")
    ap.add_argument("--camera", type=int, default=None, help="default config.WEBCAM_INDEX")
    ap.add_argument("--conf", type=float, default=None, help="default config.CONFIDENCE")
    ap.add_argument("--roi", type=_parse_roi, default=None, help="x,y,w,h pick-area crop (default config.PICK_ROI)")
    ap.add_argument("--no-window", action="store_true", help="do not open a GUI window (headless test)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N camera frames (0 = until q)")
    args = ap.parse_args()
    roi = args.roi if args.roi is not None else config.PICK_ROI

    log = runlog.start_run("detect")
    det = Detector(confidence=args.conf, roi=roi)
    log.info("model=%s classes=%s load=%.1fs roi=%s conf=%.2f", det.model_id, det.class_names,
             det.load_seconds, roi, det.confidence)
    if config.TARGET_CLASS != "any" and config.TARGET_CLASS not in det.class_names:
        log.warning("target class %r not in model classes %s", config.TARGET_CLASS, det.class_names)

    def report(dets: list[Detection], frame_tag: str) -> None:
        for d in dets:
            log.info("det %s %s", frame_tag, d)
        n_target = sum(is_target(d) for d in dets)
        log.info("%s: %d detections, %d %s", frame_tag, len(dets), n_target, config.TARGET_CLASS)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"cannot read {args.image}")
        t0 = time.time()
        dets = det.detect(frame)
        log.info("inference %.3fs", time.time() - t0)
        report(dets, args.image)
        out_path = args.image.rsplit(".", 1)[0] + "-boxes.jpg"
        cv2.imwrite(out_path, draw(frame, dets, roi=roi))
        log.info("wrote %s", out_path)
        if not args.no_window:
            cv2.imshow("detect", draw(frame, dets, roi=roi))
            cv2.waitKey(0)
        return

    camera = config.WEBCAM_INDEX if args.camera is None else args.camera
    cap = open_camera(camera)
    log.info("camera %d opened %dx%d", camera, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("camera read failed")
                break
            t0 = time.time()
            dets = det.detect(frame)
            dt = time.time() - t0
            n += 1
            report(dets, f"frame{n}")
            vis = draw(frame, dets, roi=roi)
            cv2.putText(vis, f"{1 / max(dt, 1e-3):.1f} fps  q=quit s=save", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if not args.no_window:
                cv2.imshow("detect", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    log.info("saved %s", runlog.save_frame(frame, f"frame{n}"))
                    log.info("saved %s", runlog.save_frame(vis, f"frame{n}-boxes"))
            if args.frames and n >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
