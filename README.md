# Defect-detect bot

An overhead webcam, an RF-DETR model trained on [Roboflow](https://roboflow.com) (running locally), and a
Hiwonder MaxArm. It spots defective wood blocks (drilled holes, scratches), suction-picks them, drops them
down a chute, and rescans until the table is clean. Good blocks stay where they are.

Pipeline: `webcam -> RF-DETR (Defect/Good) -> homography (pixels -> arm mm) -> MaxArm -> suction -> bin`.
The arm's own firmware does the inverse kinematics; this code only ever sends end-effector (x, y, z) and
suction on/off over USB serial.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python opencv-python numpy inference pyserial
echo "ROBOFLOW_API_KEY=..." > .env
.venv/bin/python tests/test_mapping.py && .venv/bin/python tests/test_arm_protocol.py && .venv/bin/python tests/test_pick.py
```

Every physical number lives in `config.py`. Anything still `None` hasn't been measured, and the scripts
refuse to move the arm until it is. To measure: `python arm.py --jog --bootstrap` — drive the arm with the
keyboard (`t` = table height, `k` = drop point, `h` = home, `q` prints a ready-to-paste config snippet).

## Bring-up, in order

1. **Arm** — `python arm.py --probe` (no motion; power the arm ≥ 15 s first, it homes itself at boot),
   then `python arm.py --read`. The serial protocol is documented in `docs/maxarm-sdk-recon.md` — the
   important part is that the firmware never acknowledges a move and silently ignores unreachable targets,
   so every move is verified by reading the position back.
2. **Camera** — `python detect.py` shows the live feed with boxes (`s` saves a frame). Camera exposure is
   set automatically on open (see `camera_controls.py` — macOS ignores OpenCV's exposure API, so this talks
   UVC directly via the bundled `tools/uvc-util`). `python capture.py` collects training frames.
3. **Calibration** — one block on the table, then `python calibrate.py --auto --carry`: centre the block
   under the cup once, and the arm carries it to a grid of spots by itself (~4 min, hands off). Then
   `python calibrate.py --verify`: press `b` to hover over the best-detected block; if the cup is
   consistently off-centre, nudge it with `w/a/s/d`, press `c` on 2–3 blocks, `x` to fold the offset into
   the calibration. Move the camera or the arm base -> recalibrate.
4. **Run** — `python pick.py --dry-run` (no motion), `python pick.py --once` (one pick), `python pick.py`
   (until no defects remain). `--zoom` crops the window to the pick area, `--record demo.mp4` records the
   overlay, `--countdown 5` gives you time to arrange windows before motion starts.

## Safety

- Targets outside the reach box, below table height, or within 120 mm of the arm's own base are refused
  before a byte hits the serial port. Home/drop/travel poses are validated at start-up.
- A move whose position read-back is missing or >10 mm off is treated as refused: vent, set the block down
  if mid-carry, home, skip that block.
- First motion of a session asks `Workspace clear? [y/N]`. No default yes. Don't run it unattended.
- Every detection and every serial frame is logged to `logs/<timestamp>.log`, with the frame that triggered
  each pick saved alongside — the demo videos are rendered from these logs.

## Files

| File | What it does |
|---|---|
| `config.py` | Every physical / environment value. |
| `detect.py` | Model loaded once; `Detector.detect(frame) -> [Detection]`; live view; size filter + dedupe. |
| `calibrate.py` | Self-calibration (`--auto --carry`), manual mode, verify + offset nudge, offline check. |
| `mapping.py` | Homography fit/load, degeneracy checks, leave-one-out residuals, suspect-point finder. |
| `arm.py` | MaxArm serial driver + safety envelope; `--jog`, `--probe`, `--max-z` reach probe. |
| `pick.py` | The loop, with a live HUD. |
| `camera_controls.py` | UVC camera controls (exposure) applied on every camera open. |
| `capture.py` | Training-frame capture with the same camera settings the robot uses. |
| `hud.py` + `compose_demo.py` / `dashboard_demo.py` / `panel_demo.py` / `rail_demo.py` | Demo-video tooling: overlays and analytics panels rendered/replayed from run logs. |
| `runlog.py` | Timestamped per-run logging + saved frames. |
| `tests/` | Offline unit tests (fake serial, fake detector, fake arm — no hardware needed). |
