"""Collect training frames from the mounted camera, with the same camera controls the robot uses.

    python capture.py                  live view; SPACE saves a frame, q quits
    python capture.py --every 2        also auto-save every 2 s
    python capture.py --out dataset/holes

Frames go to dataset/<timestamp>-NNN.jpg at full resolution. Vary block positions, rotations, defects, and
lighting between frames; several blocks per frame is fine.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2

import config
import detect
import runlog


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--every", type=float, default=0.0, help="auto-save interval in seconds (0 = manual only)")
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()
    log = runlog.start_run("capture")
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cap = detect.open_camera(config.WEBCAM_INDEX if args.camera is None else args.camera)
    n, last = 0, time.time()
    print(f"saving to {args.out}/{stamp}-NNN.jpg   SPACE = save   q = quit" + (f"   auto every {args.every}s" if args.every else ""))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vis = frame.copy()
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blown = 100 * (g > 245).mean()
            cv2.putText(vis, f"saved {n}   blown-out {blown:.1f}%   SPACE=save q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if blown > 2 else (255, 255, 255), 2)
            cv2.imshow("capture", vis)
            key = cv2.waitKey(1) & 0xFF
            save = key == ord(" ") or (args.every and time.time() - last >= args.every)
            if key == ord("q"):
                break
            if save:
                n += 1
                path = os.path.join(args.out, f"{stamp}-{n:03d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                log.info("saved %s (blown-out %.1f%%)", path, blown)
                last = time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{n} frames in {args.out}/")


if __name__ == "__main__":
    main()
