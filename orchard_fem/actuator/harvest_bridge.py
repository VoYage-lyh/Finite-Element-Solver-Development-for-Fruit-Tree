"""Bridge: FEM harvest-optimiser output → DS5L1 machine settings → execution.

This is the *translation + execution* layer that feeds the parameters computed
by the simulation (:mod:`orchard_fem.harvest`) into the **real** DS5L1
reciprocating-vibration actuator.

What the simulation produces vs. what the machine consumes
----------------------------------------------------------
The optimiser works in physical quantities:

    (frequency_hz, force_amplitude_n / displacement_amplitude_m, duration_s)

The DS5L1 electric cylinder is *displacement-controlled* and consumes:

    (stroke_mm, segment_rpm, accel_ms, run_duration)

so the missing glue is a unit/semantic translation plus a feasibility/safety
gate.  :func:`plan_harvest_execution` does that translation; the resulting
:class:`HarvestPlan` is what gets executed.

How the rig actually reciprocates (bench-verified)
--------------------------------------------------
The **DS5L1S-20P4-PTA** unit works as follows (see
:mod:`orchard_fem.actuator.ds5l1` and the ``ds5l1-driver-facts`` memory):

* the *drive* loops two **relative** segments (``+S`` / ``-S``) autonomously via
  step-mode-0 + ``/CHGSTP`` forced ON (``P5-35 = n.0010``); the PC does **not**
  time reversals;
* the shake **frequency** is an emergent function of the segment **rpm** and the
  positioning-settling overhead — there is no direct "frequency" register.  It
  is seeded from the ``1/(2f) = 6·S/rpm + C`` model and then locked by *online*
  frequency calibration (counting ``U0-81`` 1↔2 segment transitions).

Execution goes through the verified driver
(:class:`orchard_fem.actuator.ds5l1.DS5L1`) — represented here by the
:class:`VibrationDriver` protocol so this module stays free of the
``pyserial`` dependency and the executor can be unit-tested with a fake
driver.  For the full simulation→rig linkage (serial connect, homing,
calibration-table reuse) use
:func:`orchard_fem.actuator.ds5l1.run_harvest_plan_on_rig`.

Stroke convention
-----------------
On this rig the rod is homed to mid-stroke and the two relative segments make it
oscillate ``0 → +S → 0 → +S …`` — i.e. the **stroke ``S`` (mm) is the
peak-to-peak excursion** of the clamp point.  The simulation's displacement
*amplitude* (half peak-to-peak, as returned by
:func:`orchard_fem.harvest.basin.steady_amplitude`) therefore maps as
``S = 2 · amplitude``; use :func:`stroke_from_amplitude_m`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orchard_fem.harvest.objective import HarvestParameters


# --------------------------------------------------------------------------- #
# Machine envelope (verified DS5L1S-20P4-PTA values; override per rig)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DS5L1Limits:
    """Mechanical / servo envelope of the DS5L1 reciprocating rig.

    Defaults are the bench-verified values for the lab unit (10 mm-lead
    ball-screw cylinder, 17-bit motor, direct drive); see
    :mod:`orchard_fem.actuator.ds5l1`.  ``pulses_per_mm`` comes from
    ``10000 pulses/rev ÷ 10 mm/rev``.

    The frequency model mirrors the GUI's ``model_rpm`` / ``model_fmax``: in
    "wait-for-positioning" mode each half cycle must traverse the stroke ``S``
    plus a fixed positioning-settling overhead ``C`` inside ``1/(2f)`` seconds.
    With the screw mapping ``rpm → mm/s = rpm·lead/60`` the time to move ``S`` is
    ``6·S/rpm`` for a 10 mm lead, so ``1/(2f) = 6·S/rpm + C``.
    """

    pulses_per_mm: float = 1000.0
    screw_lead_mm: float = 10.0
    max_stroke_mm: float = 20.0          # effective stroke (cylinder is 51.67 mm, margin kept)
    max_freq_hz: float = 15.0
    rpm_cap: float = 1500.0              # excitation rpm ceiling (safety margin under 3000 rated)
    c_overhead_s: float = 0.025          # half-cycle positioning-settling overhead [s]

    def __post_init__(self) -> None:
        for name in ("pulses_per_mm", "screw_lead_mm", "max_stroke_mm",
                     "max_freq_hz", "rpm_cap"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.c_overhead_s < 0.0:
            raise ValueError("c_overhead_s must be non-negative.")

    @property
    def _mm_per_rpm_s(self) -> float:
        """mm travelled per second per rpm = lead/60."""
        return self.screw_lead_mm / 60.0

    def half_period_s(self, stroke_mm: float, rpm: float) -> float:
        """Modelled half-cycle time [s] at *stroke_mm* and *rpm*."""
        return stroke_mm / (rpm * self._mm_per_rpm_s) + self.c_overhead_s

    def seed_rpm(self, stroke_mm: float, frequency_hz: float) -> float | None:
        """Initial segment rpm to seek *frequency_hz* at *stroke_mm*.

        Returns ``None`` when even ``rpm_cap`` cannot reach the frequency at this
        stroke (i.e. the request is outside the envelope).  This is only a seed;
        the executor's online calibration locks the real frequency.
        """
        avail = 1.0 / (2.0 * frequency_hz) - self.c_overhead_s
        min_half = stroke_mm / (self.rpm_cap * self._mm_per_rpm_s)
        if avail <= min_half:
            return None
        return min(self.rpm_cap, stroke_mm / (avail * self._mm_per_rpm_s))

    def max_frequency_at_stroke(self, stroke_mm: float) -> float:
        """Highest reachable shake frequency [Hz] at *stroke_mm* (rpm-cap bound)."""
        return 1.0 / (2.0 * self.half_period_s(stroke_mm, self.rpm_cap))

    def rpm_for(self, stroke_mm: float, frequency_hz: float, c_overhead_s: float) -> float | None:
        """rpm to hit *frequency_hz* given a *measured* overhead ``C`` (calibration)."""
        avail = 1.0 / (2.0 * frequency_hz) - c_overhead_s
        min_half = stroke_mm / (self.rpm_cap * self._mm_per_rpm_s)
        if avail <= min_half:
            return None
        return min(self.rpm_cap, stroke_mm / (avail * self._mm_per_rpm_s))


# --------------------------------------------------------------------------- #
# Translation: physical amplitude → machine stroke
# --------------------------------------------------------------------------- #


def stroke_from_amplitude_m(amplitude_m: float) -> float:
    """Peak-to-peak stroke [mm] for a displacement *amplitude* (half p-p) [m].

    The rig oscillates ``0 → +S → 0``, so the peak-to-peak excursion equals the
    stroke ``S``; a half-peak-to-peak amplitude ``A`` therefore needs ``S = 2A``.
    Pair with :func:`orchard_fem.harvest.basin.steady_amplitude`, whose return
    value is exactly that half peak-to-peak amplitude.
    """
    if amplitude_m < 0.0:
        raise ValueError("amplitude_m must be non-negative.")
    return 2.0 * amplitude_m * 1000.0


# --------------------------------------------------------------------------- #
# The executable plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HarvestPlan:
    """Machine-ready parameters translated from a harvest-optimiser result.

    Parameters
    ----------
    stroke_mm:
        Peak-to-peak clamp excursion to command (segment ``±S``).
    seed_rpm:
        Initial segment rpm; refined in place by online calibration.
    accel_ms:
        Segment accel/decel time [ms].
    frequency_hz / duration_s:
        Target shake frequency and run duration carried from the simulation.
    n_cycles:
        Applied cycle count ``f·T`` (drives the fatigue tiers of the objective).
    feasible:
        ``True`` when the request is within the envelope (and the robustness
        gate, if used).  :func:`execute_harvest_plan` refuses an infeasible plan.
    notes:
        Human-readable feasibility / clamping diagnostics.
    excitation_label:
        Excitation position identifier carried for traceability.
    requested_stroke_mm:
        Stroke asked for before any envelope clamping (== ``stroke_mm`` when feasible).
    """

    stroke_mm: float
    seed_rpm: float
    accel_ms: int
    frequency_hz: float
    duration_s: float
    n_cycles: float
    feasible: bool
    notes: tuple[str, ...] = ()
    excitation_label: str = ""
    requested_stroke_mm: float = 0.0

    def summary(self) -> str:
        head = "FEASIBLE" if self.feasible else "INFEASIBLE"
        s = (f"[{head}] {self.frequency_hz:g} Hz × {self.duration_s:g} s "
             f"(n={self.n_cycles:g}) → stroke {self.stroke_mm:.2f} mm, "
             f"seed {self.seed_rpm:.0f} rpm, accel {self.accel_ms} ms")
        if self.excitation_label:
            s += f" @ {self.excitation_label}"
        if self.notes:
            s += "\n  - " + "\n  - ".join(self.notes)
        return s


def plan_harvest_execution(
    *,
    frequency_hz: float,
    clamp_peak_to_peak_mm: float,
    duration_s: float,
    limits: DS5L1Limits | None = None,
    accel_ms: int = 10,
    excitation_label: str = "",
    integrity_factor: float | None = None,
    min_integrity_factor: float = 0.0,
) -> HarvestPlan:
    """Translate one optimiser working point into an executable :class:`HarvestPlan`.

    Parameters
    ----------
    frequency_hz:
        Target shake frequency [Hz].
    clamp_peak_to_peak_mm:
        Peak-to-peak displacement the actuator must impose at the clamp [mm]
        (== the commanded stroke ``S``).  From a displacement *amplitude* use
        :func:`stroke_from_amplitude_m`.
    duration_s:
        Working duration [s]; sets ``n = f·T``.
    limits:
        Machine envelope; defaults to :class:`DS5L1Limits`.
    accel_ms:
        Segment accel/decel ramp time [ms].
    integrity_factor / min_integrity_factor:
        Optional robustness gate.  When *integrity_factor* (from
        :func:`orchard_fem.harvest.basin.compute_basin_ccm`) is supplied and
        falls below *min_integrity_factor*, the plan is marked infeasible — the
        high-amplitude working state is too fragile to start-up disturbance to
        rely on in the field.

    Returns
    -------
    HarvestPlan
        ``feasible`` reflects all gates; on any failure the offending reason is
        recorded in ``notes`` and ``seed_rpm`` falls back to ``0``.
    """
    lim = limits or DS5L1Limits()
    notes: list[str] = []
    feasible = True

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    if clamp_peak_to_peak_mm <= 0.0:
        raise ValueError("clamp_peak_to_peak_mm must be positive.")

    requested = clamp_peak_to_peak_mm
    stroke = requested

    if stroke > lim.max_stroke_mm:
        feasible = False
        notes.append(
            f"stroke {stroke:.2f} mm exceeds the {lim.max_stroke_mm:g} mm "
            f"effective stroke; reduce the target amplitude or re-clamp closer "
            f"to a higher-compliance node."
        )

    if frequency_hz > lim.max_freq_hz:
        feasible = False
        notes.append(f"frequency {frequency_hz:g} Hz exceeds the {lim.max_freq_hz:g} Hz rig limit.")

    seed = lim.seed_rpm(min(stroke, lim.max_stroke_mm), frequency_hz)
    if seed is None:
        feasible = False
        f_max = lim.max_frequency_at_stroke(min(stroke, lim.max_stroke_mm))
        notes.append(
            f"{frequency_hz:g} Hz unreachable at {stroke:.2f} mm "
            f"(max ≈ {f_max:.2f} Hz here under the {lim.rpm_cap:g} rpm cap); "
            f"lower the frequency or the stroke."
        )
        seed = 0.0

    if integrity_factor is not None and integrity_factor < min_integrity_factor:
        feasible = False
        notes.append(
            f"integrity factor {integrity_factor:.2f} < required "
            f"{min_integrity_factor:.2f}: the working state is too fragile to "
            f"start-up disturbance."
        )

    if feasible:
        notes.append("within envelope; seed rpm will be refined by online calibration.")

    return HarvestPlan(
        stroke_mm=stroke,
        seed_rpm=seed,
        accel_ms=int(accel_ms),
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        n_cycles=frequency_hz * duration_s,
        feasible=feasible,
        notes=tuple(notes),
        excitation_label=excitation_label,
        requested_stroke_mm=requested,
    )


def plan_from_harvest_parameters(
    params: "HarvestParameters",
    *,
    clamp_displacement_amplitude_m: float,
    limits: DS5L1Limits | None = None,
    accel_ms: int = 10,
    integrity_factor: float | None = None,
    min_integrity_factor: float = 0.0,
) -> HarvestPlan:
    """Build a plan straight from a :class:`~orchard_fem.harvest.objective.HarvestParameters`.

    The optimiser parameterises the working point by *force* amplitude; the rig
    needs the *displacement* it produces at the clamp.  Supply that displacement
    amplitude (half peak-to-peak [m]) — e.g. from the basin SDOF
    :func:`~orchard_fem.harvest.basin.steady_amplitude`, or, in the linear
    regime, ``force_amplitude_n · |H_excitation(ω)|`` from the FRF.
    """
    return plan_harvest_execution(
        frequency_hz=params.frequency_hz,
        clamp_peak_to_peak_mm=stroke_from_amplitude_m(clamp_displacement_amplitude_m),
        duration_s=params.duration_s,
        limits=limits,
        accel_ms=accel_ms,
        excitation_label=getattr(params, "excitation_label", "") or "",
        integrity_factor=integrity_factor,
        min_integrity_factor=min_integrity_factor,
    )


# --------------------------------------------------------------------------- #
# Execution against the verified driver
# --------------------------------------------------------------------------- #


class VibrationDriver(Protocol):
    """Subset of the verified ``DS5L1`` driver that the executor needs.

    :class:`orchard_fem.actuator.ds5l1.DS5L1` satisfies this structurally;
    keeping it a protocol means this module never imports ``pyserial`` and the
    executor is testable with a fake driver.
    """

    def init_mode(self, wait_mode: int) -> str: ...
    def set_vibration(self, stroke_mm: float, rpm: float, accel_ms: int) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def alarm(self) -> int: ...
    def measure_freq(self, dur: float = 5.0) -> float | None: ...


@dataclass(frozen=True)
class CalibrationOutcome:
    rpm: float
    measured_hz: float | None
    converged: bool
    iterations: int


def calibrate_frequency(
    driver: VibrationDriver,
    plan: HarvestPlan,
    *,
    limits: DS5L1Limits | None = None,
    max_iter: int = 4,
    tol_frac: float = 0.02,
    tol_abs_hz: float = 0.03,
    measure_s: float = 5.0,
    settle: Callable[[float], None] = time.sleep,
    settle_s: float = 1.0,
) -> CalibrationOutcome:
    """Lock the shake frequency by iterating segment rpm against measured frequency.

    Uses the same reverse-solve as the GUI's online calibration: from a measured
    frequency, back out the *actual* per-cycle overhead ``C`` at this rpm, then
    solve the rpm that would hit the target.  The drive must already be running
    (:meth:`start`); rewrites take effect live via :meth:`set_vibration`.
    """
    lim = limits or DS5L1Limits()
    rpm = plan.seed_rpm
    f_target = plan.frequency_hz
    measured: float | None = None
    converged = False
    used = 0
    for used in range(1, max_iter + 1):
        measured = driver.measure_freq(measure_s)
        if measured is None:
            raise IOError("Frequency calibration failed: no segment cycling detected.")
        if abs(measured - f_target) <= max(tol_abs_hz, tol_frac * f_target):
            converged = True
            break
        # back out the realised overhead at the current rpm, then resolve rpm
        c_meas = 1.0 / (2.0 * measured) - lim.half_period_s(plan.stroke_mm, rpm) + lim.c_overhead_s
        rpm_new = lim.rpm_for(plan.stroke_mm, f_target, c_meas)
        if rpm_new is None:
            rpm = lim.rpm_cap
            driver.set_vibration(plan.stroke_mm, rpm, plan.accel_ms)
            break
        if abs(rpm_new - rpm) < 1.0:
            converged = True
            break
        rpm = rpm_new
        driver.set_vibration(plan.stroke_mm, rpm, plan.accel_ms)
        settle(settle_s)
    return CalibrationOutcome(rpm=rpm, measured_hz=measured, converged=converged, iterations=used)


def execute_harvest_plan(
    plan: HarvestPlan,
    driver: VibrationDriver,
    *,
    home: Callable[[], None] | None = None,
    calibrate: bool = True,
    limits: DS5L1Limits | None = None,
    alarm_poll_s: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    on_status: Callable[[str], None] | None = None,
    on_calibrated: Callable[[CalibrationOutcome], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    calibration_kwargs: dict | None = None,
) -> str:
    """Drive the rig through one harvest run.

    Returns ``"completed"``, ``"alarm_stop"``, or ``"user_stop"``.

    Sequence (mirrors the verified bring-up): initialise internal-position mode
    while disabled → optional centring → write the ``±S`` segments → start the
    autonomous segment loop → optionally calibrate frequency online → poll the
    alarm word for the run duration → stop (always, via ``finally``).

    Parameters
    ----------
    plan:
        Output of :func:`plan_harvest_execution`; refused if not ``feasible``.
    driver:
        A :class:`VibrationDriver` (the real ``DS5L1`` instance).
    home:
        Optional centring callback run before motion (e.g.
        ``lambda: drv.home_center()`` or a contactless recentre).  Strongly
        recommended on hardware so the ``0 → +S`` stroke stays inside the
        cylinder; omitted only when the caller has already centred.
    calibrate:
        Run :func:`calibrate_frequency` after start to lock the shake frequency.
    on_calibrated:
        Called with the :class:`CalibrationOutcome` once calibration finishes —
        e.g. to persist the ``(stroke, frequency) → rpm`` point to a table.
    should_stop:
        Polled once per alarm cycle; returning ``True`` ends the run early with
        outcome ``"user_stop"`` (front-end stop button / e-stop request).

    Raises
    ------
    ValueError
        If *plan* is infeasible.
    """
    if not plan.feasible:
        raise ValueError("Refusing to execute an infeasible plan:\n" + plan.summary())

    def status(msg: str) -> None:
        if on_status is not None:
            on_status(msg)

    status("Initialising internal-position mode (servo disabled)…")
    driver.init_mode(0)                     # wait-for-positioning → exact amplitude
    if home is not None:
        status("Centring to mid-stroke…")
        home()
    status(f"Applying segments ±{plan.stroke_mm:.2f} mm @ {plan.seed_rpm:.0f} rpm…")
    driver.set_vibration(plan.stroke_mm, plan.seed_rpm, plan.accel_ms)
    driver.start()
    outcome = "completed"
    try:
        if calibrate:
            status("Calibrating shake frequency online…")
            cal = calibrate_frequency(driver, plan, limits=limits, **(calibration_kwargs or {}))
            note = (f"locked {cal.measured_hz:.2f} Hz @ {cal.rpm:.0f} rpm"
                    if cal.measured_hz is not None else "calibration inconclusive")
            status(f"Calibration {'converged' if cal.converged else 'capped'}: {note}.")
            if on_calibrated is not None:
                on_calibrated(cal)
        status(f"Running for {plan.duration_s:g} s…")
        t0 = now()
        while now() - t0 < plan.duration_s:
            if should_stop is not None and should_stop():
                outcome = "user_stop"
                status("Stop requested — stopping.")
                break
            if driver.alarm() != 0:
                outcome = "alarm_stop"
                status("Alarm raised — stopping.")
                break
            sleep(alarm_poll_s)
    finally:
        driver.stop()
        status("Servo stopped.")
    return outcome
