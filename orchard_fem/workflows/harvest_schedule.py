"""Multi-stage harvest schedule — the staged-adjustment-sequence innovation.

A single ``(f, A)`` work point only resonates the subset of branches whose
modes it hits; one frequency cannot shake a whole tree clean.  The packaged
port of ``verify_pareto_end_to_end.py``'s ``_greedy_sequence`` builds a
**sequence** of work points: at each stage it picks the ``(f, A)`` cell that
activates the most *new* branches per unit of trunk stress, accumulates the
covered branches, and stops at a coverage target (or no-new-branches, or a
stage cap).  That is the "during-operation parameter-adjustment schedule".

This module turns that sequence into something the cylinder can run:

    branch-outcome grid on the best clamp
        → greedy stage selection (new-branches-per-MPa)
        → per-stage duration from the fatigue model (cycles-to-detach ÷ f)
        → a HarvestSchedule of executable HarvestPlans

The per-stage **duration** is derived from the cumulative-detachment fatigue
law (:class:`~orchard_fem.harvest.objective.DetachmentFatigueLaw`): a marginal
fruit at the detachment threshold (load ratio ``r ≈ 1``) needs
``reference_cycles`` oscillations to abscise, so a stage runs that many cycles
(scaled down for fruit well above threshold), and ``duration = cycles / f``.

The greedy logic and the duration model are pure and unit-testable; the FE grid
builder (:func:`build_branch_outcome_grid`) is the only dolfinx-bound piece and
is injectable, so :func:`compute_harvest_schedule` can be exercised with a fake
grid.  Execute the result with
:func:`orchard_fem.actuator.harvest_bridge.execute_harvest_schedule`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from orchard_fem.actuator.harvest_bridge import (
    DS5L1Limits,
    HarvestSchedule,
    HarvestStage,
    plan_harvest_execution,
)


# --------------------------------------------------------------------------- #
# Per-(f, A) branch outcome (one cell of the scan grid on the chosen clamp)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BranchOutcome:
    """Detachment outcome at one ``(f, A)`` cell on the chosen clamp.

    Parameters
    ----------
    detached_branches:
        Ids of branches with at least one fruit detached at this work point.
    branch_governing_ratio:
        Per detached branch, the **minimum** load ratio ``r = F_inertia /
        F_detach`` among that branch's detached fruit — i.e. the hardest-to-shed
        (closest-to-threshold) fruit there, which governs how long the stage
        must run.
    trunk_stress_pa:
        Peak trunk bending stress at this work point [Pa].
    n_detached_fruits:
        Number of fruit detached at this work point.
    """

    detached_branches: frozenset[str]
    branch_governing_ratio: dict[str, float]
    trunk_stress_pa: float
    n_detached_fruits: int


BranchOutcomeGrid = dict[tuple[float, float], BranchOutcome]


# --------------------------------------------------------------------------- #
# Per-stage duration from the fatigue model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageDurationModel:
    """Fatigue-anchored per-stage run duration.

    A stage must oscillate long enough for its *governing* (closest-to-threshold)
    newly-targeted fruit to fatigue off, so the duration is **cycle-based**:

        cycles   = reference_cycles · safety_factor / r**ratio_exponent  (≥ min_cycles)
        duration = cycles / f                                           (≤ max_duration_s)

    where ``reference_cycles`` is the cycles needed to shake a *threshold* fruit
    (``r = 1``) off; fruit well above threshold come off in fewer.

    ``reference_cycles`` is **the** physical/calibration parameter — set it from
    measured shake-duration vs removal-efficiency data
    (cf. :class:`~orchard_fem.harvest.objective.DetachmentFatigueLaw`).  Real
    vibratory harvesting needs tens–hundreds of cycles, not the law's idealised
    10, so the default is a realistic placeholder of 50 (the old value of 10 gave
    unrealistically short ~2 s stages).  No hard time floor is applied — the
    duration follows the physics; the front-end exposes this as "Detach cycles".
    """

    reference_cycles: float = 50.0
    safety_factor: float = 1.0
    ratio_exponent: float = 1.0
    min_cycles: float = 3.0
    max_duration_s: float = 60.0

    def cycles(self, governing_ratio: float) -> float:
        r = max(float(governing_ratio), 1.0)
        n = self.reference_cycles * self.safety_factor / (r ** self.ratio_exponent)
        return max(self.min_cycles, n)

    def duration_s(self, frequency_hz: float, governing_ratio: float) -> float:
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive.")
        return min(self.max_duration_s, self.cycles(governing_ratio) / frequency_hz)


# --------------------------------------------------------------------------- #
# Greedy multi-stage scheduler (pure — ported from the e2e study)
# --------------------------------------------------------------------------- #


def compute_harvest_schedule(
    grid: BranchOutcomeGrid,
    *,
    n_fruit_branches: int,
    clamp_label: str = "",
    target_coverage: float = 0.95,
    max_stages: int = 5,
    limits: DS5L1Limits | None = None,
    accel_ms: int = 10,
    duration_model: StageDurationModel | None = None,
) -> HarvestSchedule:
    """Greedily build a staged schedule from a branch-outcome *grid*.

    At each stage, pick the ``(f, A)`` that activates the most *new* branches per
    unit of trunk stress; assign it a fatigue-derived duration; accumulate
    coverage; stop at *target_coverage*, when no cell adds new branches, or after
    *max_stages*.  Each stage is wrapped in a :class:`HarvestPlan`
    (:func:`plan_harvest_execution`) so the rig envelope / feasibility is applied
    per stage and the result is directly executable.

    Parameters
    ----------
    grid:
        ``{(f_hz, A_mm): BranchOutcome}`` on the chosen clamp.  ``A_mm`` is the
        displacement amplitude (half peak-to-peak); the rig stroke is ``2·A``.
    n_fruit_branches:
        Number of fruit-bearing branches (denominator of coverage).
    """
    dur = duration_model or StageDurationModel()
    lim = limits or DS5L1Limits()
    n_total = max(int(n_fruit_branches), 1)

    def _cell_runnable(f: float, a: float) -> bool:
        """Whether the rig can actually execute (f, stroke=2A) — skip if not."""
        stroke = 2.0 * a
        return (stroke <= lim.max_stroke_mm and f <= lim.max_freq_hz
                and lim.seed_rpm(stroke, f) is not None)

    activated: set[str] = set()
    stages: list[HarvestStage] = []
    for stage_index in range(1, max_stages + 1):
        best_score = 0.0
        best: tuple[float, float, set[str], BranchOutcome] | None = None
        for (f, a), info in grid.items():
            new = set(info.detached_branches) - activated
            sigma = info.trunk_stress_pa
            if not new or sigma <= 0.0 or not _cell_runnable(f, a):
                continue
            score = len(new) / sigma            # new branches per unit stress
            if score > best_score:
                best_score = score
                best = (f, a, new, info)
        if best is None:
            break                               # no cell adds new branches

        f, a, new, info = best
        governing_ratio = min(info.branch_governing_ratio[b] for b in new)
        duration_s = dur.duration_s(f, governing_ratio)
        plan = plan_harvest_execution(
            frequency_hz=f,
            clamp_peak_to_peak_mm=2.0 * a,      # stroke S = 2·A
            duration_s=duration_s,
            limits=limits,
            accel_ms=accel_ms,
            excitation_label=clamp_label,
        )
        activated |= new
        coverage = len(activated) / n_total
        stages.append(HarvestStage(
            index=stage_index,
            plan=plan,
            new_branches=tuple(sorted(new)),
            cumulative_coverage=coverage,
            trunk_stress_pa=info.trunk_stress_pa,
            n_detached_fruits=info.n_detached_fruits,
        ))
        if coverage >= target_coverage:
            break

    return HarvestSchedule(
        stages=tuple(stages),
        clamp_label=clamp_label,
        target_coverage=target_coverage,
    )


# --------------------------------------------------------------------------- #
# FE backend: build the branch-outcome grid (dolfinx; injectable for tests)
# --------------------------------------------------------------------------- #


def build_branch_outcome_grid(
    model: Any,
    clamp_label: str,
    f_grid: list[float],
    a_grid_mm: list[float],
    *,
    theta: dict[str, float] | None = None,
    limits: DS5L1Limits | None = None,
    n_jobs: int = -1,
    progress_cb: Callable[[str, float], None] | None = None,
) -> BranchOutcomeGrid:
    """Solve each *runnable* ``(f, A)`` cell and tabulate branch outcomes.

    Per-fruit inertia ``F = m·ω²·|u|`` vs. detachment ``F = k·d`` aggregated to
    per-branch detachment + governing load ratio, plus the trunk stress — all
    from **one** solve per cell (the shared
    :func:`~orchard_fem.calibration.fenicsx_bridge.build_outcome_solver`, which
    replaces the old separate fruit and stress solves).  Cells the rig can't
    execute are skipped, and the independent solves fan out across processes
    (``n_jobs`` <=0 = all cores, 1 = serial).  Requires dolfinx; imported lazily.
    """
    from dataclasses import replace as _replace

    from orchard_fem.calibration.fenicsx_bridge import build_outcome_solver
    from orchard_fem.domain import ExcitationKind
    from orchard_fem.workflows._solve_pool import (
        resolve_n_jobs,
        solve_outcomes_parallel,
    )

    lim = limits or DS5L1Limits()
    theta = theta or {
        "E": float(model.materials[0].youngs_modulus),
        "rho": float(model.materials[0].density),
    }

    def _runnable(f: float, a: float) -> bool:
        stroke = 2.0 * a
        return (stroke <= lim.max_stroke_mm and f <= lim.max_freq_hz
                and lim.seed_rpm(stroke, f) is not None)

    work_points = [
        (clamp_label, float(f), float(a))
        for f in f_grid for a in a_grid_mm
        if _runnable(float(f), float(a))
    ]
    if not work_points:
        return {}

    disp_model = _replace(model, excitation=_replace(
        model.excitation, kind=ExcitationKind.HARMONIC_DISPLACEMENT))
    n = len(work_points)
    n_jobs_r = resolve_n_jobs(n_jobs, n)
    done = 0

    def _on_done(_c: str, _f: float, _a: float) -> None:
        nonlocal done
        done += 1
        if progress_cb is not None:
            progress_cb(f"grid {clamp_label}: {done}/{n} solves", done / n)

    if n_jobs_r > 1:
        solver_kw = dict(amplitude_unit="mm", coverage_mode="branch")
        outcomes = solve_outcomes_parallel(
            disp_model, solver_kw, work_points, theta, n_jobs_r, on_done=_on_done)
    else:
        solve = build_outcome_solver(
            disp_model, amplitude_unit="mm", coverage_mode="branch")
        outcomes = {}
        for clamp, f, a in work_points:
            outcomes[(clamp, f, a)] = solve(theta, f, a, clamp)
            _on_done(clamp, f, a)

    grid: BranchOutcomeGrid = {}
    for (_clamp, f, a), o in outcomes.items():
        grid[(f, a)] = BranchOutcome(
            detached_branches=o.detached_branches,
            branch_governing_ratio=dict(o.branch_governing_ratio),
            trunk_stress_pa=o.trunk_stress_pa,
            n_detached_fruits=o.n_detached_fruits,
        )
    return grid
