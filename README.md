# Block Picker

Overhead webcam + Roboflow RF-DETR detector (run locally) + Hiwonder MaxArm. The arm picks every
`red` block it can see and drops it in a fixed drop zone. Plan: `block-picker-plan.md`. Status: `PROGRESS.md`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python opencv-python numpy inference pyserial
export ROBOFLOW_API_KEY=...          # never commit it
.venv/bin/python tests/test_mapping.py && .venv/bin/python tests/test_arm_protocol.py && .venv/bin/python tests/test_pick.py
```

All physical values live in `config.py`. Anything still `None` there is owner-provided and the
scripts refuse to move the arm until it is filled in.

## Build gates (do them in this order)

1. **SDK recon** — `PROGRESS.md` (and `docs/maxarm-sdk-recon.md`) record the MaxArm serial protocol,
   function codes, units and boot behaviour, validated on this arm. Owner confirms the ESP32 still runs
   Hiwonder's `MaxArm_micropython_microUSB` firmware (micro-USB, 9600 baud), then: `python arm.py --probe`
   and `python arm.py --read` (no motion; power the arm on ≥ 15 s first — it homes itself at boot).
2. **Detection standalone** — mount the camera, then `python detect.py`. Gate: `red` boxes on real red
   blocks, none on the mat. If small blocks are mislabelled, set `PICK_ROI` in `config.py` to the pick
   area (see `PROGRESS.md`) and re-check. `python detect.py --image frame.jpg` tests a saved frame.
3. **Calibration** (owner runs it) — 4–6 tape marks spread over the pick area. `python calibrate.py`:
   click a mark, jog the arm to it (`python arm.py --jog` or the Hiwonder app), type its arm `x,y`.
   Writes `calibration.npy`. Then `python calibrate.py --verify`: click anywhere, the arm hovers there.
   Gate: within ~5 mm on 3+ spots, one near the edge. `python calibrate.py --check` prints residuals.
   Move the camera → recalibrate.
4. **Dry-run pick loop** — `python pick.py --dry-run`. Prints and draws a target for each visible red
   block; reports "no red" on a clear table. Gate: owner eyeballs the drawn targets against the blocks.
5. **Real pick loop** — `python pick.py --once`, then `python pick.py`. Gate: three consecutive
   successful picks at different positions, no non-red block touched.

## Safety

- `arm.py` refuses any target outside `REACH_*_MM` or below `TABLE_Z_MM` (raises, never sends).
- The first real motion of a process asks `Workspace clear? [y/N]`. No default yes.
- **Never leave `pick.py` running unattended**, even with `--once`. Stay within reach of the power switch.
- `--dry-run` never writes to the serial port.
- Every detection and every arm command is logged to `logs/<timestamp>.log`; the frame that triggered
  each pick is saved next to it.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Every physical / environment value. |
| `detect.py` | Model loaded once; `Detector.detect(frame) -> list[Detection]`; standalone live view. |
| `calibrate.py` | Click-to-calibrate, verify mode, offline check. Writes `calibration.npy`. |
| `mapping.py` | `load_homography()`, `pixel_to_arm(px, py)` via `cv2.perspectiveTransform`. |
| `arm.py` | MaxArm serial protocol wrapper: `connect`, `home`, `move_to`, `suction`, `wait`; safety envelope. |
| `pick.py` | Main loop: `--dry-run`, `--once`, `--class`. |
| `runlog.py` | Timestamped per-run log + saved frames. |
| `tests/` | Offline unit tests (no hardware). |
