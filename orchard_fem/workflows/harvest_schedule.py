"""Multi-stage harvest schedule — the staged-adjustment-sequence innovation.

A single ``(f, A)`` work point only resonates the subset of branches whose
modes it hits; one frequency cannot shake a whole tree clean.  The packaged
port of ``generate_all_figures.py``'s ``_greedy_sequence`` builds a
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


def _order_stages_grouped_by_level(
    raw: list[tuple[str, float, float, Any]],
    n_total: int,
    dur: StageDurationModel,
    limits: DS5L1Limits | None,
    accel_ms: int,
) -> list[HarvestStage]:
    """Reorder the greedy clamp-blocks so NON-TRUNK (primary-branch) grips run
    first and the trunk last, grouping by level.

    The greedy picks re-clamps purely by "most new branches", which makes the
    operator hop primary→trunk→primary — costly, since each re-clamp is a physical
    reposition. Branch coverage is the UNION over stages, which is order-independent,
    so we can safely regroup: all primary-branch clamps first (in their greedy
    order), then the trunk. Coverage/new-branch attribution and indices are
    recomputed for the new order, and stages that add no NEW branch (already
    covered by an earlier block) are dropped.
    """
    if not raw:
        return []

    def _is_trunk(clamp_label: str) -> bool:
        return clamp_label.split("@", 1)[0] == "trunk"

    # Group consecutive same-clamp entries into blocks (greedy makes them contiguous).
    blocks: list[tuple[str, list[tuple[float, float, Any]]]] = []
    for clamp, f, a, info in raw:
        if blocks and blocks[-1][0] == clamp:
            blocks[-1][1].append((f, a, info))
        else:
            blocks.append((clamp, [(f, a, info)]))
    ordered = ([b for b in blocks if not _is_trunk(b[0])]
               + [b for b in blocks if _is_trunk(b[0])])

    stages: list[HarvestStage] = []
    activated: set[str] = set()
    for clamp, cells in ordered:
        # Within a clamp, cover with the FEWEST cells: take the highest-coverage
        # (usually highest-amplitude) cell first, so a near-duplicate lower cell
        # whose branches are a subset (e.g. 5.0 Hz/15 mm ⊂ 5.1 Hz/20 mm) drops out
        # as redundant instead of becoming a wasteful extra stage. Amplitude
        # increases detachment monotonically, so the big cell dominates the small.
        cells = sorted(cells, key=lambda c: len(c[2].detached_branches), reverse=True)
        for f, a, info in cells:
            new = set(info.detached_branches) - activated
            if not new:
                continue  # already covered by an earlier cell/block → drop
            governing_ratio = min(info.branch_governing_ratio[b] for b in new)
            plan = plan_harvest_execution(
                frequency_hz=f,
                clamp_peak_to_peak_mm=2.0 * a,
                duration_s=dur.duration_s(f, governing_ratio),
                limits=limits,
                accel_ms=accel_ms,
                excitation_label=clamp,
            )
            activated |= new
            stages.append(HarvestStage(
                index=len(stages) + 1,
                plan=plan,
                new_branches=tuple(sorted(new)),
                cumulative_coverage=len(activated) / max(n_total, 1),
                trunk_stress_pa=info.trunk_stress_pa,
                n_detached_fruits=info.n_detached_fruits,
            ))
    return stages


def compute_multiclamp_harvest_schedule(
    grids: dict[str, BranchOutcomeGrid],
    *,
    n_fruit_branches: int,
    target_coverage: float = 0.95,
    max_stages: int | None = None,
    limits: DS5L1Limits | None = None,
    accel_ms: int = 10,
    duration_model: StageDurationModel | None = None,
) -> HarvestSchedule:
    """Greedy multi-**clamp** schedule: cover the tree by moving the grip between
    energy-reachable regions, not by frequency-sweeping a single clamp.

    A clamp's excitation energy attenuates with distance, so one grip can only
    shed fruit on the branches it actually reaches — exciting the left of the tree
    will not detach fruit at the far-right tips. Each clamp's ``(f, A)`` grid is
    therefore its *reachable set*. The scheduler **exhausts the current clamp's
    reachable-but-unharvested branches first** (cheap — no re-clamp), then moves
    to the clamp that newly activates the most branches, minimising the physical
    re-clamping a single-clamp frequency sweep can never avoid (it simply leaves
    the unreachable branches uncovered).

    *grids* maps each candidate ``branch_id@s`` clamp to its
    :class:`BranchOutcomeGrid`. The chosen clamp of each stage is carried on
    ``stage.plan.excitation_label``; :attr:`HarvestSchedule.n_reclamps` counts the
    repositionings the operator/rig must perform.
    """
    dur = duration_model or StageDurationModel()
    lim = limits or DS5L1Limits()
    n_total = max(int(n_fruit_branches), 1)
    # max_stages=None → no cap: each stage activates ≥1 NEW branch, so the greedy
    # loop terminates naturally at ≤ n_total stages (or earlier when no clamp can
    # add a branch / target coverage is met). Cover as many branches as possible.
    if max_stages is None:
        max_stages = n_total

    def _runnable(f: float, a: float) -> bool:
        stroke = 2.0 * a
        return (stroke <= lim.max_stroke_mm and f <= lim.max_freq_hz
                and lim.seed_rpm(stroke, f) is not None)

    def _best_cell(clamp: str, done: set[str]):
        """Best runnable (f, a, new, info) on *clamp* adding new branches, or None."""
        best_score = 0.0
        best = None
        for (f, a), info in grids[clamp].items():
            new = set(info.detached_branches) - done
            sigma = info.trunk_stress_pa
            if not new or sigma <= 0.0 or not _runnable(f, a):
                continue
            score = len(new) / sigma
            if score > best_score:
                best_score = score
                best = (f, a, new, info)
        return best

    def _reach(clamp: str) -> set[str]:
        out: set[str] = set()
        for (f, a), info in grids[clamp].items():
            if _runnable(f, a):
                out |= set(info.detached_branches)
        return out

    if not grids:
        return HarvestSchedule(stages=(), clamp_label="", target_coverage=target_coverage)

    # Start on the clamp that can reach the most branches (fewest re-clamps later).
    current = max(grids, key=lambda c: len(_reach(c)))

    activated: set[str] = set()
    raw: list[tuple[str, float, float, Any]] = []  # (clamp, f, a, outcome)
    for _stage_index in range(1, max_stages + 1):
        best = _best_cell(current, activated)
        if best is None:
            # Current grip exhausted its reachable branches → re-clamp to the one
            # that newly activates the most branches (tie-break: per-stress score).
            options = [(c, _best_cell(c, activated)) for c in grids if c != current]
            options = [(c, b) for c, b in options if b is not None]
            if not options:
                break
            current, best = max(
                options, key=lambda cb: (len(cb[1][2]), len(cb[1][2]) / cb[1][3].trunk_stress_pa),
            )

        f, a, new, info = best
        activated |= new
        raw.append((current, f, a, info))
        if len(activated) / n_total >= target_coverage:
            break

    stages = _order_stages_grouped_by_level(raw, n_total, dur, limits, accel_ms)
    return HarvestSchedule(
        stages=tuple(stages),
        clamp_label=(stages[0].plan.excitation_label if stages else ""),
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
    polynomial_degree: int = 1,
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
    # Empty θ preserves the real composite section; never default to materials[0]
    # (pith) — _apply_theta_to_model would homogenise the tree ~8× too soft and
    # crush coverage. θ is a calibration override, not a material pass-through.
    theta = theta or {}

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
        solver_kw = dict(amplitude_unit="mm", coverage_mode="branch",
                         polynomial_degree=polynomial_degree)
        outcomes = solve_outcomes_parallel(
            disp_model, solver_kw, work_points, theta, n_jobs_r, on_done=_on_done)
    else:
        solve = build_outcome_solver(
            disp_model, amplitude_unit="mm", coverage_mode="branch",
            polynomial_degree=polynomial_degree)
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


def build_multiclamp_schedule(
    scheduling_model: Any,
    clamp_frequencies_hz: dict[str, list[float]],
    amplitude_grid_mm: list[float],
    *,
    limits: DS5L1Limits | None = None,
    polynomial_degree: int = 2,
    target_coverage: float = 0.95,
    max_stages: int | None = None,
    duration_model: StageDurationModel | None = None,
    progress_cb: Callable[[str, float], None] | None = None,
):
    """Build the multi-clamp staged schedule shared by the console and
    ``generate_all_figures``, so the figure schedule == the rig-executed schedule.

    ``clamp_frequencies_hz`` maps each candidate clamp label to the drive
    frequencies to scan for it (each clamp on its own local-mode frequencies);
    the scheduler then moves the grip between energy-reachable regions. The
    caller extracts these because the two front-ends carry different clamp types.
    ``scheduling_model`` MUST be the prepared (fruited + damped) model from
    :func:`~orchard_fem.workflows.harvest_recommendation.build_scheduling_model`,
    or coverage drifts from the recommendation.
    """
    n_branches = max(len({f.branch_id for f in scheduling_model.fruits}), 1)
    grids: dict[str, BranchOutcomeGrid] = {}
    for clamp_label, freqs in clamp_frequencies_hz.items():
        grids[clamp_label] = build_branch_outcome_grid(
            scheduling_model, clamp_label, sorted(freqs), amplitude_grid_mm,
            limits=limits, polynomial_degree=polynomial_degree,
            progress_cb=progress_cb,
        )
    return compute_multiclamp_harvest_schedule(
        grids, n_fruit_branches=n_branches, target_coverage=target_coverage,
        max_stages=max_stages, limits=limits, duration_model=duration_model,
    )
