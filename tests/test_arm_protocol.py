import os
import struct
import sys
import tempfile

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


def _reply(func, xyz):
    frame = b"\xAA\x55" + bytes([func, 6]) + struct.pack("<hhh", *xyz)
    return frame + bytes([arm.checksum(frame)])  # firmware: checksum over the whole frame


def test_parse_reply_firmware_checksum_and_noise():
    body = bytes([0x13, 0x06]) + struct.pack("<hhh", 100, -20, 60)
    frame = b"\xAA\x55" + body
    fw_style = frame + bytes([arm.checksum(frame)])
    req_style = frame + bytes([arm.checksum(body)])   # request convention (also accepted)
    bad = frame + bytes([(arm.checksum(frame) + 7) & 0xFF])
    assert arm.checksum(frame) == (arm.checksum(body) + 1) & 0xFF
    assert arm.parse_frame(b"\x00\xAA" + bad + fw_style, 0x13) == body[2:]
    assert arm.parse_frame(req_style, 0x13) == body[2:]
    assert arm.parse_frame(bad, 0x13) is None
    assert arm.parse_frame(fw_style, 0x11) is None
    # a bogus header with an overrunning length byte must not hide the valid reply behind it
    assert arm.parse_frame(b"\xAA\x55\x13\x20" + fw_style, 0x13) == body[2:]


def _set_envelope():
    saved = {k: getattr(config, k) for k in ("TABLE_Z_MM", "REACH_X_MM", "REACH_Y_MM", "REACH_Z_MM", "HOME_XYZ_MM")}
    config.TABLE_Z_MM, config.REACH_X_MM, config.REACH_Y_MM, config.REACH_Z_MM = 50.0, (-150.0, 150.0), (-280.0, -100.0), (50.0, 200.0)
    config.HOME_XYZ_MM = (0.0, -160.0, 150.0)
    return saved


def _restore(saved):
    for k, v in saved.items():
        setattr(config, k, v)


def test_safety_rejects_bad_targets():
    saved = _set_envelope()
    try:
        arm.check_target(0, -200, 90)  # fine
        for bad in ((0, -200, 49.9), (200, -200, 90), (0, -50, 90), (0, -200, 250)):
            try:
                arm.check_target(*bad)
            except arm.UnsafeTarget:
                continue
            raise AssertionError(f"expected UnsafeTarget for {bad}")
    finally:
        _restore(saved)


def test_extension_ratio_matches_rig_observations():
    # Home pose from the SDK is exactly the fully stretched horizontal forearm: L2 up, L3+L4 out.
    home = (0, -(arm.L1 + arm.L3 + arm.L4), arm.L0 + arm.L2)
    assert 0.7 < arm.extension_ratio(*home) < 0.75  # = L3 / (L2 + L3)
    # Observed 2026-08-25: (-266, 13, 153) reached in the jog, (-266, 13, 193) refused by the firmware.
    assert arm.extension_ratio(-266, 13, 153) < 0.95
    assert arm.extension_ratio(-266, 13, 193) > 0.99
    assert arm.extension_ratio(-210, -56, 139) < 0.8


def test_bootstrap_envelope():
    arm.check_bootstrap(0, -160, 150)
    for bad in ((0, -20, 100), (0, -320, 100), (0, -160, 260), (0, -160, -1)):
        try:
            arm.check_bootstrap(*bad)
        except arm.UnsafeTarget:
            continue
        raise AssertionError(f"expected UnsafeTarget for {bad}")
    saved = config.TABLE_Z_MM
    config.TABLE_Z_MM = None  # bootstrap arm must not need config values
    try:
        a = arm.Arm(port="/dev/fake", bootstrap=True)
        a._ser = FakeSerial(); a._cleared = True; a.wait = lambda s: None
        assert a.move_to(0, -160, 150) == (0, -160, 150)
    finally:
        config.TABLE_Z_MM = saved


def test_unset_config_raises():
    saved = config.TABLE_Z_MM
    config.TABLE_Z_MM = None
    try:
        arm.check_target(0, -200, 90)
    except config.ConfigError:
        return
    finally:
        config.TABLE_Z_MM = saved
    raise AssertionError("expected ConfigError")


def test_dry_run_never_touches_serial():
    saved = _set_envelope()
    try:
        a = arm.Arm(port="/dev/does-not-exist", dry_run=True).connect()
        assert a.move_to(0, -200, 90) is None
        assert a.home() is None
        a.suction(True); a.suction(False)
        assert a.read_xyz() is None and a.cleared
    finally:
        _restore(saved)


class FakeSerial:
    """Fake port: SET_XYZ moves the 'arm' unless the target is in `refuse`; READ_XYZ echoes position."""

    def __init__(self, refuse=(), silent=False):
        self.pos = (0, -160, 150)
        self.refuse = set(refuse)
        self.silent = silent
        self.written = []
        self.reply = b""
        self.in_waiting = 0

    def write(self, data):
        self.written.append(bytes(data))
        if data[2] == arm.FUNC_SET_XYZ:
            x, y, z, _ = struct.unpack("<hhhH", data[4:12])
            if (x, y, z) not in self.refuse:
                self.pos = (x, y, z)
        elif data[2] == arm.FUNC_READ_XYZ and not self.silent:
            self.reply += _reply(arm.FUNC_READ_XYZ, self.pos)

    def flush(self):
        pass

    def read(self, n):
        out, self.reply = self.reply[:n], self.reply[n:]
        return out

    def reset_input_buffer(self):
        self.reply = b""

    def close(self):
        pass


def _fake_arm(**kw):
    a = arm.Arm(port="/dev/fake")
    a._ser = FakeSerial(**kw)
    a._cleared = True
    a.wait = lambda s: None
    return a


def test_move_to_verifies_arrival_and_raises_on_refusal():
    saved = _set_envelope()
    try:
        a = _fake_arm(refuse={(100, -200, 90)})
        assert a.move_to(0, -200, 90) == (0, -200, 90)
        try:
            a.move_to(100, -200, 90)  # firmware silently ignores this one
        except arm.MoveRefused:
            pass
        else:
            raise AssertionError("expected MoveRefused")
        silent = _fake_arm(silent=True)
        try:
            silent.move_to(0, -200, 90)
        except arm.MoveRefused:
            pass
        else:
            raise AssertionError("expected MoveRefused when nothing answers")
    finally:
        _restore(saved)


def test_move_duration_bounds_refused_before_sending():
    saved = _set_envelope()
    try:
        a = _fake_arm()
        for ms in (0, 100, int(config.MOVE_TIMEOUT_S * 1000) + 1, 65535):
            try:
                a.move_to(0, -200, 90, ms)
            except ValueError:
                pass
            else:
                raise AssertionError(f"ms={ms} should be refused")
        assert a._ser.written == []  # nothing transmitted
    finally:
        _restore(saved)


if __name__ == "__main__":
    config.LOG_DIR = tempfile.mkdtemp()  # keep test logs out of logs/
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("arm protocol tests OK")
