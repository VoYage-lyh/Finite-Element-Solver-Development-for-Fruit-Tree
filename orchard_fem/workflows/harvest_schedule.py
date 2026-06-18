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
    progress_cb: Callable[[str, float], None] | None = None,
) -> BranchOutcomeGrid:
    """Solve the FE problem on each ``(f, A)`` cell and tabulate branch outcomes.

    Ports ``verify_pareto_end_to_end.py``'s ``_compute_fruit_outcomes`` (per-fruit
    inertia ``F = m·ω²·|u|`` vs. detachment ``F = k·d``) and aggregates to
    per-branch detachment + governing load ratio, plus the trunk stress from the
    Pareto evaluator.  Requires dolfinx; imported lazily here.
    """
    import numpy as np

    from orchard_fem.calibration.fenicsx_bridge import (
        _apply_theta_to_model,
        _parse_clamp_label,
        build_fenicsx_pareto_evaluator,
    )
    from orchard_fem.domain import ExcitationKind
    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    from orchard_fem.topology import ObservationPoint

    theta = theta or {
        "E": float(model.materials[0].youngs_modulus),
        "rho": float(model.materials[0].density),
    }
    # Trunk stress comes from the displacement-excitation Pareto evaluator.
    from dataclasses import replace as _replace

    disp_model = _replace(model, excitation=_replace(
        model.excitation, kind=ExcitationKind.HARMONIC_DISPLACEMENT))
    stress_eval = build_fenicsx_pareto_evaluator(
        disp_model, amplitude_unit="mm", coverage_mode="branch")

    branch_id, target_s = _parse_clamp_label(clamp_label)
    d_detach_default = 0.010

    grid: BranchOutcomeGrid = {}
    n_total = len(f_grid) * len(a_grid_mm)
    done = 0
    for f in f_grid:
        for a in a_grid_mm:
            cloned = _apply_theta_to_model(model, theta)
            cloned = _replace(cloned, excitation=_replace(
                cloned.excitation,
                kind=ExcitationKind.HARMONIC_DISPLACEMENT,   # impose A mm at the clamp
                target_branch_id=branch_id,
                target_s=target_s if target_s is not None else cloned.excitation.target_s,
                amplitude=float(a) * 1.0e-3,
                driving_frequency_hz=float(f),
            ), analysis=_replace(
                cloned.analysis,
                frequency_start_hz=float(f),
                frequency_end_hz=float(f) + 1.0e-6,
                frequency_steps=1,
            ))
            extras: list = []
            fruit_keys: list = []
            for fruit in cloned.fruits:
                oid = f"__sched_fruit_{fruit.fruit_id}"
                extras.append(ObservationPoint(
                    observation_id=oid, target_type="fruit",
                    target_id=fruit.fruit_id, target_node="tip",
                    target_components=[fruit.target_component]))
                fruit_keys.append((oid, fruit))
            cloned = _replace(cloned, observations=list(cloned.observations) + extras)

            exp = solve_embedded_beam_frequency_response_experiment(
                cloned, polynomial_degree=1)
            point = exp.result.points[0]
            name_to_idx = {n: i for i, n in enumerate(exp.result.observation_names)}
            d_detach = (float(cloned.fruit_policy.detachment_displacement_m)
                        if cloned.fruit_policy is not None else d_detach_default)
            omega = 2.0 * np.pi * float(f)

            governing: dict[str, float] = {}
            n_detached = 0
            for oid, fruit in fruit_keys:
                idx = name_to_idx.get(oid)
                if idx is None or fruit.stiffness <= 0.0:
                    continue
                u_mag = float(point.observation_magnitudes[idx])
                inertia = fruit.mass * omega * omega * u_mag
                detach_force = fruit.stiffness * d_detach
                if detach_force <= 0.0:
                    continue
                ratio = inertia / detach_force
                if ratio >= 1.0:                # detached (binary inertia criterion)
                    n_detached += 1
                    prev = governing.get(fruit.branch_id)
                    governing[fruit.branch_id] = ratio if prev is None else min(prev, ratio)

            stress_pa = float(stress_eval(theta, float(f), float(a), clamp_label).trunk_max_stress)
            grid[(float(f), float(a))] = BranchOutcome(
                detached_branches=frozenset(governing.keys()),
                branch_governing_ratio=governing,
                trunk_stress_pa=stress_pa,
                n_detached_fruits=n_detached,
            )
            done += 1
            if progress_cb is not None:
                progress_cb(f"grid {clamp_label}: f={f:g} A={a:g} → "
                            f"{len(governing)} branches, σ {stress_pa / 1e6:.2f} MPa",
                            done / n_total)
    return grid
