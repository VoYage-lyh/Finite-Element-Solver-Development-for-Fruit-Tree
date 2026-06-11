"""Tests for the harvest → DS5L1 translation/execution bridge.

Pure logic + a fake DS5L1 driver (the verified ``DS5L1`` is duck-typed via the
:class:`VibrationDriver` protocol), so no hardware/pyserial is needed.
"""
from __future__ import annotations

import pytest

from orchard_fem.actuator.harvest_bridge import (
    DS5L1Limits,
    HarvestPlan,
    calibrate_frequency,
    execute_harvest_plan,
    plan_from_harvest_parameters,
    plan_harvest_execution,
    stroke_from_amplitude_m,
)
from orchard_fem.harvest.objective import HarvestParameters


# --------------------------------------------------------------------------- #
# Translation: amplitude → stroke, frequency → seed rpm
# --------------------------------------------------------------------------- #

def test_stroke_from_amplitude_is_peak_to_peak():
    # 4 mm half peak-to-peak amplitude → 8 mm peak-to-peak stroke
    assert stroke_from_amplitude_m(0.004) == pytest.approx(8.0)


def test_seed_rpm_matches_half_period_model():
    lim = DS5L1Limits()
    rpm = lim.seed_rpm(stroke_mm=4.0, frequency_hz=2.0)
    assert rpm is not None
    # the seeded rpm should reproduce ~the target half period in the model
    assert lim.half_period_s(4.0, rpm) == pytest.approx(1.0 / (2.0 * 2.0), rel=1e-6)


def test_seed_rpm_capped_returns_none_when_unreachable():
    lim = DS5L1Limits()
    # very high frequency at a long stroke is out of the rpm-capped envelope
    assert lim.seed_rpm(stroke_mm=20.0, frequency_hz=14.0) is None
    f_max = lim.max_frequency_at_stroke(20.0)
    assert f_max < 14.0


def test_plan_feasible_point():
    plan = plan_harvest_execution(
        frequency_hz=2.0, clamp_peak_to_peak_mm=4.0, duration_s=10.0,
        excitation_label="branchA_3",
    )
    assert plan.feasible is True
    assert plan.stroke_mm == pytest.approx(4.0)
    assert plan.seed_rpm > 0.0
    assert plan.n_cycles == pytest.approx(20.0)
    assert plan.excitation_label == "branchA_3"


def test_plan_infeasible_stroke_too_large():
    plan = plan_harvest_execution(
        frequency_hz=2.0, clamp_peak_to_peak_mm=25.0, duration_s=5.0,
    )
    assert plan.feasible is False
    assert any("exceeds" in n for n in plan.notes)
    assert plan.requested_stroke_mm == pytest.approx(25.0)


def test_plan_infeasible_frequency_unreachable():
    plan = plan_harvest_execution(
        frequency_hz=14.0, clamp_peak_to_peak_mm=18.0, duration_s=5.0,
    )
    assert plan.feasible is False
    assert plan.seed_rpm == 0.0
    assert any("unreachable" in n or "exceeds" in n for n in plan.notes)


def test_plan_integrity_factor_gate():
    ok = plan_harvest_execution(
        frequency_hz=2.0, clamp_peak_to_peak_mm=4.0, duration_s=5.0,
        integrity_factor=0.4, min_integrity_factor=0.3,
    )
    assert ok.feasible is True
    fragile = plan_harvest_execution(
        frequency_hz=2.0, clamp_peak_to_peak_mm=4.0, duration_s=5.0,
        integrity_factor=0.1, min_integrity_factor=0.3,
    )
    assert fragile.feasible is False
    assert any("fragile" in n for n in fragile.notes)


def test_plan_from_harvest_parameters():
    params = HarvestParameters(
        frequency_hz=3.0, force_amplitude_n=12.0, duration_s=8.0,
        excitation_label="branchB_5",
    )
    plan = plan_from_harvest_parameters(params, clamp_displacement_amplitude_m=0.003)
    assert plan.feasible is True
    assert plan.stroke_mm == pytest.approx(6.0)   # 2 * 3 mm
    assert plan.frequency_hz == 3.0
    assert plan.excitation_label == "branchB_5"


def test_plan_rejects_bad_inputs():
    with pytest.raises(ValueError):
        plan_harvest_execution(frequency_hz=0.0, clamp_peak_to_peak_mm=4.0, duration_s=5.0)
    with pytest.raises(ValueError):
        plan_harvest_execution(frequency_hz=2.0, clamp_peak_to_peak_mm=-1.0, duration_s=5.0)
    with pytest.raises(ValueError):
        plan_harvest_execution(frequency_hz=2.0, clamp_peak_to_peak_mm=4.0, duration_s=0.0)


# --------------------------------------------------------------------------- #
# Fake driver + execution
# --------------------------------------------------------------------------- #

class FakeDriver:
    """Records the DS5L1 driver calls the executor makes."""

    def __init__(self, *, freq_sequence=None, alarm_after=None):
        self.calls: list[tuple] = []
        self._freqs = list(freq_sequence or [])
        self._alarm_after = alarm_after   # n alarm-polls before raising
        self._alarm_polls = 0
        self.stopped = False

    def init_mode(self, wait_mode: int) -> str:
        self.calls.append(("init_mode", wait_mode))
        return "ok"

    def set_vibration(self, stroke_mm, rpm, accel_ms) -> None:
        self.calls.append(("set_vibration", round(stroke_mm, 3), round(rpm, 1), accel_ms))

    def start(self) -> None:
        self.calls.append(("start",))

    def stop(self) -> None:
        self.calls.append(("stop",))
        self.stopped = True

    def alarm(self) -> int:
        self._alarm_polls += 1
        if self._alarm_after is not None and self._alarm_polls > self._alarm_after:
            return 161
        return 0

    def measure_freq(self, dur: float = 5.0):
        return self._freqs.pop(0) if self._freqs else None


def _feasible_plan() -> HarvestPlan:
    return plan_harvest_execution(
        frequency_hz=2.0, clamp_peak_to_peak_mm=4.0, duration_s=1.0,
    )


def test_execute_refuses_infeasible():
    plan = plan_harvest_execution(
        frequency_hz=2.0, clamp_peak_to_peak_mm=30.0, duration_s=1.0,
    )
    with pytest.raises(ValueError, match="infeasible"):
        execute_harvest_plan(plan, FakeDriver(), calibrate=False)


def test_execute_completes_and_sequences_calls():
    drv = FakeDriver()
    homed = {"n": 0}
    clock = {"t": 0.0}
    outcome = execute_harvest_plan(
        _feasible_plan(), drv,
        home=lambda: homed.__setitem__("n", homed["n"] + 1),
        calibrate=False,
        now=lambda: clock["t"],
        sleep=lambda dt: clock.__setitem__("t", clock["t"] + dt),
    )
    assert outcome == "completed"
    assert homed["n"] == 1
    names = [c[0] for c in drv.calls]
    # init before centring, segments written before start, stop last
    assert names.index("init_mode") < names.index("set_vibration") < names.index("start")
    assert names[-1] == "stop"
    assert drv.stopped is True


def test_execute_alarm_stop_still_disables():
    drv = FakeDriver(alarm_after=2)
    outcome = execute_harvest_plan(
        _feasible_plan(), drv, calibrate=False,
        now=lambda: 0.0,                    # never time out → exit only via alarm
        sleep=lambda dt: None,
    )
    assert outcome == "alarm_stop"
    assert drv.calls[-1] == ("stop",)


def test_execute_with_calibration_refines_rpm():
    # first measurement low, then on target → one rpm rewrite, converges
    drv = FakeDriver(freq_sequence=[1.6, 2.0])
    plan = _feasible_plan()
    clock = {"t": 0.0}
    outcome = execute_harvest_plan(
        plan, drv, calibrate=True,
        now=lambda: clock["t"],
        sleep=lambda dt: clock.__setitem__("t", clock["t"] + max(dt, 0.5)),
        calibration_kwargs={"settle": lambda dt: None, "measure_s": 0.0},
    )
    assert outcome == "completed"
    set_vibration_calls = [c for c in drv.calls if c[0] == "set_vibration"]
    # one initial apply + at least one calibration rewrite
    assert len(set_vibration_calls) >= 2


def test_calibrate_frequency_converges_on_target():
    drv = FakeDriver(freq_sequence=[2.0])   # already on target
    plan = _feasible_plan()
    out = calibrate_frequency(drv, plan, settle=lambda dt: None, measure_s=0.0)
    assert out.converged is True
    assert out.measured_hz == pytest.approx(2.0)
    assert out.iterations == 1


def test_calibrate_frequency_raises_without_cycling():
    drv = FakeDriver(freq_sequence=[])      # measure_freq returns None
    with pytest.raises(IOError, match="no segment cycling"):
        calibrate_frequency(drv, _feasible_plan(), settle=lambda dt: None, measure_s=0.0)
