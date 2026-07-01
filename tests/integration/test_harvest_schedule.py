"""Tests for the multi-stage harvest schedule (greedy + fatigue duration + executor).

Pure logic with a fake branch-outcome grid and a fake driver — no dolfinx/hardware.
"""
from __future__ import annotations

import pytest

from orchard_fem.actuator.harvest_bridge import (
    HarvestSchedule,
    HarvestStage,
    execute_harvest_schedule,
    plan_harvest_execution,
)
from orchard_fem.workflows.harvest_schedule import (
    BranchOutcome,
    StageDurationModel,
    build_multiclamp_schedule,
    compute_harvest_schedule,
    compute_multiclamp_harvest_schedule,
)


# --------------------------------------------------------------------------- #
# Duration model
# --------------------------------------------------------------------------- #

def test_duration_model_threshold_fruit_runs_reference_cycles():
    m = StageDurationModel(reference_cycles=10.0, safety_factor=1.5, min_cycles=3.0)
    # marginal fruit (r=1): cycles = 10*1.5 = 15 → at 5 Hz, 3.0 s
    assert m.cycles(1.0) == pytest.approx(15.0)
    assert m.duration_s(5.0, 1.0) == pytest.approx(15.0 / 5.0)


def test_duration_model_easy_fruit_runs_fewer_cycles_but_floored():
    m = StageDurationModel(reference_cycles=10.0, safety_factor=1.5,
                           ratio_exponent=1.0, min_cycles=3.0)
    # well above threshold (r=10): 15/10 = 1.5 < floor → min_cycles 3.0
    assert m.cycles(10.0) == pytest.approx(3.0)
    # moderately above (r=2): 15/2 = 7.5 > floor
    assert m.cycles(2.0) == pytest.approx(7.5)


def test_duration_model_capped():
    m = StageDurationModel(max_duration_s=2.0)
    assert m.duration_s(0.5, 1.0) == 2.0     # would be large at low f → capped


def test_duration_model_reference_cycles_drives_duration():
    # reference_cycles is the root parameter: threshold fruit (r=1) → N cycles ÷ f
    assert StageDurationModel(reference_cycles=50.0).duration_s(6.5, 1.0) == \
        pytest.approx(50.0 / 6.5)
    # the old default of 10 gave the unrealistically short ~1.5 s
    assert StageDurationModel(reference_cycles=10.0).duration_s(6.5, 1.0) == \
        pytest.approx(10.0 / 6.5)


# --------------------------------------------------------------------------- #
# Greedy schedule over a fake grid
# --------------------------------------------------------------------------- #

def _grid() -> dict:
    # (f, A_mm) -> BranchOutcome, all cells within the MEASURED rig envelope
    # (stroke=2A; bench max ≈7.5 Hz @stroke5, ≈7.8 Hz @stroke10):
    #   5Hz/5mm (stroke10 ✓) hits {b1,b2};
    #   7Hz/5mm (✓) hits {b3};
    #   7Hz/2.5mm (stroke5, max 7.5Hz ✓) hits {b2,b3,b4}.
    return {
        (5.0, 5.0): BranchOutcome(frozenset({"b1", "b2"}),
                                  {"b1": 1.2, "b2": 1.05}, 2.0e6, 8),
        (7.0, 5.0): BranchOutcome(frozenset({"b3"}),
                                  {"b3": 1.5}, 1.0e6, 3),
        (7.0, 2.5): BranchOutcome(frozenset({"b2", "b3", "b4"}),
                                  {"b2": 1.1, "b3": 1.1, "b4": 1.1}, 5.0e6, 6),
    }


def test_schedule_greedy_picks_best_then_covers_remaining():
    sched = compute_harvest_schedule(
        _grid(), n_fruit_branches=4, clamp_label="trunk@0.40",
        target_coverage=0.95, max_stages=5,
    )
    assert sched.clamp_label == "trunk@0.40"
    # stage 1: 5 Hz/5mm (2 new branches, first of the tied-max scores) wins
    assert sched.stages[0].plan.frequency_hz == 5.0
    assert sched.stages[0].new_branches == ("b1", "b2")
    # subsequent stages pick up b3 then b4 until full coverage
    covered = set()
    for s in sched.stages:
        covered.update(s.new_branches)
    assert {"b1", "b2", "b3", "b4"} <= covered
    assert sched.final_coverage == pytest.approx(1.0)
    assert sched.feasible


def _two_clamp_grids() -> dict:
    # Energy reach is local: the left grip only sheds the left branches {b1,b2}
    # (b3/b4 attenuate to nothing); the right grip only reaches {b3,b4}. One clamp
    # can never exceed 2/4 coverage — the tree needs both grips.
    left = {
        (5.0, 5.0): BranchOutcome(frozenset({"b1", "b2"}), {"b1": 1.2, "b2": 1.1}, 2.0e6, 8),
        (7.0, 5.0): BranchOutcome(frozenset({"b1"}), {"b1": 1.3}, 1.0e6, 4),
    }
    right = {
        (6.0, 5.0): BranchOutcome(frozenset({"b3", "b4"}), {"b3": 1.1, "b4": 1.05}, 3.0e6, 6),
    }
    return {"left@0.00": left, "right@0.00": right}


def test_multiclamp_covers_what_one_clamp_cannot():
    grids = _two_clamp_grids()
    # single clamp caps at 2/4 (energy can't reach the other half)
    one = compute_harvest_schedule(grids["left@0.00"], n_fruit_branches=4)
    assert one.final_coverage == pytest.approx(0.5)
    # multi-clamp reaches full coverage by moving the grip once
    multi = compute_multiclamp_harvest_schedule(grids, n_fruit_branches=4)
    covered = set()
    for s in multi.stages:
        covered.update(s.new_branches)
    assert {"b1", "b2", "b3", "b4"} <= covered
    assert multi.final_coverage == pytest.approx(1.0)
    assert multi.clamp_labels == ("left@0.00", "right@0.00")
    assert multi.n_reclamps == 1                       # exactly one repositioning
    # the per-stage clamp rides on the plan's excitation label
    assert multi.stages[0].plan.excitation_label == "left@0.00"
    assert multi.stages[-1].plan.excitation_label == "right@0.00"


def test_multiclamp_exhausts_current_clamp_before_reclamping():
    # left grip reaches {b1,b2} across two cells; the scheduler should use BOTH
    # left cells (no re-clamp) before moving to the right grip.
    grids = _two_clamp_grids()
    multi = compute_multiclamp_harvest_schedule(grids, n_fruit_branches=4, max_stages=6)
    labels = [s.plan.excitation_label for s in multi.stages]
    # all 'left' stages precede the first 'right' stage (no ping-ponging)
    first_right = labels.index("right@0.00")
    assert all(label == "left@0.00" for label in labels[:first_right])


def test_multiclamp_empty_grids():
    sched = compute_multiclamp_harvest_schedule({}, n_fruit_branches=4)
    assert sched.stages == ()
    assert sched.feasible is False


def test_schedule_summary_label_fn_and_reclamps():
    grids = _two_clamp_grids()
    multi = compute_multiclamp_harvest_schedule(grids, n_fruit_branches=4)
    text = multi.summary(label_fn=lambda raw: raw.replace("left", "1").replace("right", "2"))
    assert "1@0.00" in text and "2@0.00" in text          # labels translated
    assert "re-clamp" in text                              # multi-clamp header


def test_schedule_durations_are_fatigue_derived():
    sched = compute_harvest_schedule(_grid(), n_fruit_branches=4)
    dur = StageDurationModel()
    s0 = sched.stages[0]
    # governing ratio of stage 1 = min(1.2, 1.05) = 1.05
    assert s0.plan.duration_s == pytest.approx(dur.duration_s(5.0, 1.05))


def test_schedule_stops_at_coverage_target():
    sched = compute_harvest_schedule(
        _grid(), n_fruit_branches=4, target_coverage=0.5,  # 2/4 = 0.5 reached at stage 1
    )
    assert len(sched.stages) == 1
    assert sched.final_coverage == pytest.approx(0.5)


def test_schedule_empty_grid_gives_empty_schedule():
    sched = compute_harvest_schedule({}, n_fruit_branches=4)
    assert sched.stages == ()
    assert sched.feasible is False             # empty schedule is not executable


def test_schedule_skips_rig_infeasible_cells():
    # b1 reachable at a feasible cell; b2 only via A=15mm (stroke 30mm > 20mm limit).
    grid = {
        (5.0, 5.0): BranchOutcome(frozenset({"b1"}), {"b1": 1.1}, 1.0e6, 2),
        (5.0, 15.0): BranchOutcome(frozenset({"b2"}), {"b2": 1.1}, 1.0e6, 2),
    }
    sched = compute_harvest_schedule(grid, n_fruit_branches=2)
    covered = {b for s in sched.stages for b in s.new_branches}
    assert covered == {"b1"}            # b2's only cell is unreachable → skipped
    assert sched.feasible               # every selected stage is runnable


# --------------------------------------------------------------------------- #
# Staged executor with a fake driver
# --------------------------------------------------------------------------- #

class FakeDriver:
    def __init__(self, *, alarm_at_stage=None):
        self.calls: list[tuple] = []
        self._alarm_at_stage = alarm_at_stage
        self._set_vib_count = 0

    def init_mode(self, wait_mode):
        self.calls.append(("init_mode", wait_mode))
        return "ok"

    def set_vibration(self, stroke_mm, rpm, accel_ms):
        self._set_vib_count += 1
        self.calls.append(("set_vibration", round(stroke_mm, 2), round(rpm, 1)))

    def start(self):
        self.calls.append(("start",))

    def stop(self):
        self.calls.append(("stop",))

    def alarm(self):
        # raise an alarm only while the configured stage's segments are active
        if self._alarm_at_stage is not None and self._set_vib_count >= self._alarm_at_stage:
            return 161
        return 0

    def measure_freq(self, dur=5.0):
        return None     # not used (calibrate=False in tests)


def _two_stage_schedule() -> HarvestSchedule:
    p1 = plan_harvest_execution(frequency_hz=5.0, clamp_peak_to_peak_mm=8.0, duration_s=0.5)
    # 7 Hz @stroke 10mm is within the measured envelope (max ≈7.8 Hz there).
    p2 = plan_harvest_execution(frequency_hz=7.0, clamp_peak_to_peak_mm=10.0, duration_s=0.5)
    return HarvestSchedule(
        stages=(
            HarvestStage(1, p1, ("b1", "b2"), 0.5, 2.0e6, 8),
            HarvestStage(2, p2, ("b3",), 0.75, 1.0e6, 3),
        ),
        clamp_label="trunk@0.40", target_coverage=0.95,
    )


def test_execute_schedule_runs_stages_in_order_started_once():
    drv = FakeDriver()
    seen_stages: list[int] = []
    clock = {"t": 0.0}
    outcome = execute_harvest_schedule(
        _two_stage_schedule(), drv, calibrate=False,
        on_stage=lambda s: seen_stages.append(s.index),
        now=lambda: clock["t"],
        sleep=lambda dt: clock.__setitem__("t", clock["t"] + max(dt, 0.5)),
    )
    assert outcome == "completed"
    assert seen_stages == [1, 2]                      # both stages, in order
    names = [c[0] for c in drv.calls]
    assert names.count("start") == 1                  # started once, not per stage
    assert names.count("set_vibration") == 2          # one live rewrite per stage
    assert names[-1] == "stop"
    # set_vibration of stage 1 happens before start; stage 2's after (live switch)
    assert names.index("set_vibration") < names.index("start")


def test_execute_schedule_recenters_between_stages():
    # With a recenter callback, the rod is returned to mid-stroke between stages,
    # so the drive is stopped+restarted each stage (not started once).
    drv = FakeDriver()
    recenters = {"n": 0}
    clock = {"t": 0.0}
    outcome = execute_harvest_schedule(
        _two_stage_schedule(), drv, calibrate=False,
        recenter=lambda: recenters.__setitem__("n", recenters["n"] + 1),
        now=lambda: clock["t"],
        sleep=lambda dt: clock.__setitem__("t", clock["t"] + max(dt, 0.5)),
    )
    assert outcome == "completed"
    assert recenters["n"] == 1                         # once: between the 2 stages, not before stage 1
    names = [c[0] for c in drv.calls]
    assert names.count("start") == 2                   # restarted from centre each stage
    assert names.count("set_vibration") == 2


def test_execute_schedule_alarm_aborts_whole_sequence():
    drv = FakeDriver(alarm_at_stage=1)                # alarm during stage 1
    outcome = execute_harvest_schedule(
        _two_stage_schedule(), drv, calibrate=False,
        now=lambda: 0.0, sleep=lambda dt: None,
    )
    assert outcome == "alarm_stop"
    names = [c[0] for c in drv.calls]
    assert names.count("set_vibration") == 1          # stage 2 never started
    assert names[-1] == "stop"


def test_execute_schedule_refuses_infeasible():
    bad = plan_harvest_execution(frequency_hz=5.0, clamp_peak_to_peak_mm=30.0, duration_s=1.0)
    sched = HarvestSchedule(
        stages=(HarvestStage(1, bad, ("b1",), 1.0, 1e6, 1),),
        clamp_label="x", target_coverage=0.95,
    )
    with pytest.raises(ValueError, match="infeasible"):
        execute_harvest_schedule(sched, FakeDriver(), calibrate=False)


# --------------------------------------------------------------------------- #
# Shared multi-clamp builder (console + generate_all_figures)
# --------------------------------------------------------------------------- #

def test_build_multiclamp_schedule_takes_frequency_dict(monkeypatch):
    """Regression: the shared builder consumes a ``{clamp_label: freqs}`` dict and
    builds one grid per clamp. The console passes ClampRecommendation-derived
    frequencies (those objects have ``.points`` but no ``.front``), so the builder
    must key off the dict and never touch clamp-object attributes."""
    import orchard_fem.workflows.harvest_schedule as hs

    seen_grids: dict[str, list[float]] = {}

    def fake_grid(model, clamp_label, freqs, amps, **kw):
        seen_grids[clamp_label] = list(freqs)
        return {"clamp": clamp_label}

    captured: dict = {}

    def fake_schedule(grids, **kw):
        captured["clamps"] = list(grids.keys())
        captured["max_stages"] = kw.get("max_stages")
        captured["target"] = kw.get("target_coverage")
        return "SCHED"

    monkeypatch.setattr(hs, "build_branch_outcome_grid", fake_grid)
    monkeypatch.setattr(hs, "compute_multiclamp_harvest_schedule", fake_schedule)

    class _Fruit:
        def __init__(self, b):
            self.branch_id = b

    class _Model:
        fruits = [_Fruit("b1"), _Fruit("b2")]

    clamp_freqs = {"trunk@0.25": [8.0, 5.0], "b1@0.5": [7.0]}
    out = build_multiclamp_schedule(
        _Model(), clamp_freqs, [5.0, 10.0], max_stages=10, target_coverage=0.9,
    )

    assert out == "SCHED"
    assert set(seen_grids) == {"trunk@0.25", "b1@0.5"}
    assert seen_grids["trunk@0.25"] == [5.0, 8.0]        # sorted per clamp
    assert captured["clamps"] == ["trunk@0.25", "b1@0.5"]
    assert captured["max_stages"] == 10                  # our new default is 10
    assert captured["target"] == pytest.approx(0.9)
