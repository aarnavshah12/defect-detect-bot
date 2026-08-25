import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import arm  # noqa: E402


def test_checksum_matches_doc_examples():
    # Docs 10.2: read_angles command is AA 55 11 00 EE, read_xyz is AA 55 13 00 EC.
    assert arm.build_frame(arm.FUNC_READ_ANGLE) == bytes.fromhex("AA551100EE")
    assert arm.build_frame(arm.FUNC_READ_XYZ) == bytes.fromhex("AA551300EC")


def test_set_xyz_frame_matches_doc_example():
    # Hiwonder docs / master main.py: xyz (120, -180, 85) in 1000 ms
    assert arm.set_xyz_frame(120, -180, 85, 1000) == bytes.fromhex("aa550308" "7800" "4cff" "5500" "e803" "f1")


def test_set_xyz_frame_layout():
    f = arm.set_xyz_frame(120, -30, 85, 1000)
    assert f[:2] == b"\xAA\x55" and f[2] == 0x03 and f[3] == 0x08
    assert f[4:10] == (120).to_bytes(2, "little", signed=True) + (-30).to_bytes(2, "little", signed=True) + (85).to_bytes(2, "little", signed=True)
    assert f[10:12] == (1000).to_bytes(2, "little")
    assert f[12] == arm.checksum(f[2:12]) and len(f) == 13


def test_nozzle_frames():
    assert arm.nozzle_frame(1) == bytes.fromhex("AA550701" + "01" + format((~(0x07 + 0x01 + 0x01)) & 0xFF, "02X"))
    for bad in (0, 4):
        try:
            arm.nozzle_frame(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_parse_reply_firmware_checksum_and_noise():
    body = bytes([0x13, 0x06]) + (100).to_bytes(2, "little", signed=True) + (-20).to_bytes(2, "little", signed=True) + (60).to_bytes(2, "little", signed=True)
    frame = b"\xAA\x55" + body
    fw_style = frame + bytes([arm.checksum(frame)])   # firmware: checksum over the whole frame
    req_style = frame + bytes([arm.checksum(body)])   # request convention (also accepted)
    bad = frame + bytes([(arm.checksum(frame) + 7) & 0xFF])
    assert arm.checksum(frame) == (arm.checksum(body) + 1) & 0xFF
    assert arm.parse_frame(b"\x00\xAA" + bad + fw_style, 0x13) == body[2:]
    assert arm.parse_frame(req_style, 0x13) == body[2:]
    assert arm.parse_frame(bad, 0x13) is None
    assert arm.parse_frame(fw_style, 0x11) is None


def test_safety_rejects_bad_targets(monkeypatch=None):
    saved = {k: getattr(config, k) for k in ("TABLE_Z_MM", "REACH_X_MM", "REACH_Y_MM", "REACH_Z_MM")}
    config.TABLE_Z_MM = 50.0
    config.REACH_X_MM = (-150.0, 150.0)
    config.REACH_Y_MM = (100.0, 280.0)
    config.REACH_Z_MM = (50.0, 200.0)
    try:
        arm.check_target(0, 200, 90)  # fine
        for bad in ((0, 200, 49.9), (200, 200, 90), (0, 50, 90), (0, 200, 250)):
            try:
                arm.check_target(*bad)
            except arm.UnsafeTarget:
                continue
            raise AssertionError(f"expected UnsafeTarget for {bad}")
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def test_unset_config_raises():
    saved = config.TABLE_Z_MM
    config.TABLE_Z_MM = None
    try:
        arm.check_target(0, 200, 90)
    except config.ConfigError:
        return
    finally:
        config.TABLE_Z_MM = saved
    raise AssertionError("expected ConfigError")


def test_dry_run_never_touches_serial():
    saved = {k: getattr(config, k) for k in ("TABLE_Z_MM", "REACH_X_MM", "REACH_Y_MM", "REACH_Z_MM", "HOME_XYZ_MM")}
    config.TABLE_Z_MM, config.REACH_X_MM, config.REACH_Y_MM, config.REACH_Z_MM = 50.0, (-150, 150), (100, 280), (50, 200)
    config.HOME_XYZ_MM = (0, 120, 150)
    try:
        a = arm.Arm(port="/dev/does-not-exist", dry_run=True).connect()
        assert a.move_to(0, 200, 90) is None
        assert a.home() is None
        a.suction(True); a.suction(False)
        assert a.read_xyz() is None
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


if __name__ == "__main__":
    import tempfile
    config.LOG_DIR = tempfile.mkdtemp()  # keep test logs out of logs/
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("arm protocol tests OK")
