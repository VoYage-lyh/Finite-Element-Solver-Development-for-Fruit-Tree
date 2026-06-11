"""Tests for the migrated DS5L1 driver (framing, helpers, rig linkage).

Modbus framing is exercised against a fake serial port; the simulation→rig
linkage against a fake driver — no pyserial/hardware needed.
"""
from __future__ import annotations

import json

import pytest

from orchard_fem.actuator.ds5l1 import (
    DS5L1,
    calib_key,
    crc16,
    load_calib,
    run_harvest_plan_on_rig,
    s16,
    save_calib,
    split_pulses,
    u16,
)
from orchard_fem.actuator.harvest_bridge import plan_harvest_execution


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_crc16_standard_vector():
    # canonical Modbus example: 01 03 00 00 00 01 → CRC 84 0A (low byte first)
    assert crc16(bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x01])) == bytes([0x84, 0x0A])


def test_u16_s16_roundtrip():
    for v in (0, 1, -1, 9999, -9999, 32767, -32768):
        assert s16(u16(v)) == v


def test_split_pulses_decimal_signed():
    assert split_pulses(85000) == (5000, 8)
    assert split_pulses(-85000) == (-5000, -8)
    for p in (0, 1, 9999, 10000, 12345, -12345, 250000):
        lo, hi = split_pulses(p)
        assert hi * 10000 + lo == p
        assert -9999 <= lo <= 9999


# ---------------------------------------------------------------------------
# Modbus framing against a fake serial port
# ---------------------------------------------------------------------------

class FakeSerial:
    """Queues one response frame per write."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.written: list[bytes] = []
        self._buf = b""

    def reset_input_buffer(self):
        self._buf = b""

    def write(self, data):
        self.written.append(bytes(data))
        if self._responses:
            self._buf = self._responses.pop(0)

    def flush(self):
        pass

    def read(self, n):
        chunk, self._buf = self._buf[:n], self._buf[n:]
        return chunk

    def close(self):
        pass


def read_response(*regs: int) -> bytes:
    body = bytes([0x01, 0x03, 2 * len(regs)]) + b"".join(
        bytes([r >> 8, r & 0xFF]) for r in regs
    )
    return body + crc16(body)


def echo_response(request_body: bytes) -> bytes:
    return request_body + crc16(request_body)


def _drv(responses) -> DS5L1:
    d = DS5L1()
    d.ser = FakeSerial(responses)
    return d


def test_read_register_value_and_request_frame():
    d = _drv([read_response(5)])
    assert d.read("P0_01") == 5
    sent = d.ser.written[0]
    body = bytes([0x01, 0x03, 0x00, 0x01, 0x00, 0x01])   # station 1, read 1 reg @0x0001
    assert sent == body + crc16(body)


def test_read_multi_register():
    d = _drv([read_response(0x1234, 0x0002)])
    lo, hi = d.read("U0_14", count=2)
    assert (lo, hi) == (0x1234, 0x0002)


def test_read_bad_crc_retries_then_fails():
    good = read_response(5)
    bad = good[:-2] + b"\x00\x00"
    d = _drv([bad, bad, bad, bad])
    with pytest.raises(IOError, match="failed after 4 retries"):
        d.read("P0_01")
    assert len(d.ser.written) == 4


def test_read_recovers_after_one_bad_frame():
    good = read_response(7)
    bad = good[:-2] + b"\x00\x00"
    d = _drv([bad, good])
    assert d.read("P0_01") == 7


def test_write_echo_verified():
    body = bytes([0x01, 0x06, 0x21, 0x05, 0x00, 0x01])   # F1-05 = 1
    d = _drv([echo_response(body)])
    d.write("F1_05", 1, verify=False)
    assert d.ser.written[0] == body + crc16(body)


def test_write_echo_mismatch_raises():
    wrong = bytes([0x01, 0x06, 0x21, 0x05, 0x00, 0x09])
    d = _drv([echo_response(wrong)] * 4)
    with pytest.raises(IOError, match="Write F1_05"):
        d.write("F1_05", 1, verify=False)


def test_modbus_exception_response_raises():
    exc_frame = bytes([0x01, 0x83, 0x02])
    frame = exc_frame + crc16(exc_frame)
    d = _drv([frame] * 4)
    with pytest.raises(IOError, match="exception code 2"):
        d.read("P0_01")


# ---------------------------------------------------------------------------
# Calibration table store
# ---------------------------------------------------------------------------

def test_calib_roundtrip(tmp_path):
    p = tmp_path / "calib.json"
    table = {calib_key(5.0, 2.0): {"rpm": 124.5, "f_act": 1.994, "accel": 10}}
    save_calib(table, p)
    assert load_calib(p) == table


def test_load_calib_missing_file_returns_empty(tmp_path):
    assert load_calib(tmp_path / "absent.json") == {}


# ---------------------------------------------------------------------------
# Simulation → rig linkage (fake driver)
# ---------------------------------------------------------------------------

class FakeRigDriver:
    """Stands in for a connected DS5L1 in run_harvest_plan_on_rig."""

    def __init__(self, *, measured_hz: float, alarm0: int = 0, homing_first_time=False):
        self.calls: list[tuple] = []
        self._measured = measured_hz
        self._alarm = alarm0
        self._first = homing_first_time

    def alarm(self) -> int:
        return self._alarm

    def clear_alarm(self) -> int:
        self._alarm = 0
        return 0

    def init_mode(self, wait_mode: int) -> str:
        self.calls.append(("init_mode", wait_mode))
        return "ok"

    def setup_homing(self, offset_mm, reverse=True, **kw) -> bool:
        self.calls.append(("setup_homing", offset_mm, reverse))
        return self._first

    def home_center(self, status_cb=None, **kw) -> None:
        self.calls.append(("home_center",))

    def set_vibration(self, stroke_mm, rpm, accel_ms) -> None:
        self.calls.append(("set_vibration", round(stroke_mm, 3), round(rpm, 1), accel_ms))

    def start(self) -> None:
        self.calls.append(("start",))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def measure_freq(self, dur: float = 5.0):
        return self._measured

    def close(self) -> None:
        self.calls.append(("close",))


def _plan(freq=2.0, stroke=4.0, duration=0.3):
    return plan_harvest_execution(
        frequency_hz=freq, clamp_peak_to_peak_mm=stroke, duration_s=duration,
    )


def test_linkage_requires_exactly_one_of_port_or_driver():
    with pytest.raises(ValueError, match="exactly one"):
        run_harvest_plan_on_rig(_plan())
    with pytest.raises(ValueError, match="exactly one"):
        run_harvest_plan_on_rig(_plan(), port="COM8", driver=FakeRigDriver(measured_hz=2.0))


def test_linkage_full_run_saves_calibration(tmp_path):
    calib_file = tmp_path / "calib.json"
    drv = FakeRigDriver(measured_hz=2.0)
    outcome = run_harvest_plan_on_rig(
        _plan(), driver=drv, calib_path=calib_file,
    )
    assert outcome == "completed"
    names = [c[0] for c in drv.calls]
    # homing configured and executed between init and segment writes
    assert names.index("init_mode") < names.index("home_center") < names.index("set_vibration")
    assert names[-1] == "stop"            # caller owns the driver → no close
    table = json.loads(calib_file.read_text(encoding="utf-8"))
    assert calib_key(4.0, 2.0) in table
    assert table[calib_key(4.0, 2.0)]["f_act"] == pytest.approx(2.0)


def test_linkage_seeds_rpm_from_calibration_cache(tmp_path):
    calib_file = tmp_path / "calib.json"
    save_calib({calib_key(4.0, 2.0): {"rpm": 98.8, "f_act": 1.993, "accel": 10}}, calib_file)
    drv = FakeRigDriver(measured_hz=2.0)
    run_harvest_plan_on_rig(_plan(), driver=drv, calib_path=calib_file, home=False)
    first_seg = next(c for c in drv.calls if c[0] == "set_vibration")
    assert first_seg[2] == pytest.approx(98.8)


def test_linkage_homing_first_time_requires_power_cycle(tmp_path):
    drv = FakeRigDriver(measured_hz=2.0, homing_first_time=True)
    with pytest.raises(RuntimeError, match="power cycle"):
        run_harvest_plan_on_rig(_plan(), driver=drv, calib_path=tmp_path / "c.json")
    assert ("start",) not in drv.calls    # never started vibrating


def test_linkage_unclearable_alarm_refuses_to_run(tmp_path):
    class StuckAlarm(FakeRigDriver):
        def clear_alarm(self):
            return 161

    drv = StuckAlarm(measured_hz=2.0, alarm0=161)
    with pytest.raises(IOError, match="E-161"):
        run_harvest_plan_on_rig(_plan(), driver=drv, calib_path=tmp_path / "c.json")
    assert ("start",) not in drv.calls
