"""Time-dependent harvest objective with two-tier damage model.

Extends the single-frequency inertia-force detachment criterion
(:mod:`orchard_fem.harvest.detachment`) into the full vibration-harvesting
design objective over the working parameters

    (frequency f, force amplitude A, excitation position, duration T)

by adding three pieces of physics the static criterion lacks:

1. **Cumulative detachment over time.**  A fruit whose per-cycle inertia force
   does not immediately exceed the detachment force can still abscise after
   enough loading cycles (cyclic weakening of the stem attachment).  Modelled
   with a Basquin/Miner-type fatigue law: cycles-to-detach ``N(r)`` as a
   function of the load ratio ``r = F_inertia / F_detach``, with detachment when
   the applied cycles ``n = f·T`` reach ``N(r)``.  Recovers the original binary
   criterion in the ``r ≥ 1`` (immediate) limit.

2. **Clamp-point damage — soft penalty.**  The actuator grip imposes a cyclic
   stress on the branch at the clamp; accumulated fatigue damage there is a
   *soft penalty* subtracted from the objective (some grip marking is tolerable).

3. **Branch fracture / throw-off — hard constraint.**  If the working
   parameters are too aggressive the branch cracks and is flung off — a
   catastrophic outcome.  Treated as a *hard constraint*: any parameter set that
   causes instantaneous over-stress (``σ ≥ σ_ultimate``) or fatigue fracture
   within the working duration (``n ≥ N_fracture(σ)``) is infeasible.

Energy is intentionally not modelled (out of scope per project decision).

Stress inputs
-------------
The damage tiers operate on the dynamic stress amplitudes at the clamp and at
the most-stressed branch section.  These are *inputs* to
:func:`evaluate_harvest_objective`.  Obtaining them rigorously requires a
curvature-based stress post-processor (``σ = M·c/I = E·κ·c``), for which the FRF
solver already retains ``observation_complex`` phase data; see
:func:`scale_stress_with_amplitude` for the linear-regime amplitude scaling used
until that post-processor lands.

See :doc:`/docs/pinn_harvest_research_plan` §5 for how this objective fits the
zero-shot robust-optimisation pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Fatigue laws
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetachmentFatigueLaw:
    """Cycles-to-detach as a function of the inertia/detachment load ratio.

    The cycles-to-detach curve is

        N(r) = reference_cycles · ((1 - r_e) / (r - r_e)) ** exponent   for r_e < r < 1
        N(r) = 0     (immediate)                                        for r >= 1
        N(r) = +inf  (never)                                            for r <= r_e

    where ``r_e`` is :attr:`endurance_ratio`.  The curve equals
    ``reference_cycles`` at ``r = 1`` and diverges as ``r`` approaches the
    endurance ratio, so fruits well below threshold never abscise while those
    just below threshold abscise after a finite number of cycles.

    All three constants are calibratable against measured shake-duration vs.
    removal-efficiency data.

    Parameters
    ----------
    endurance_ratio:
        Load ratio below which a fruit never detaches, ``r_e`` ∈ (0, 1).
    reference_cycles:
        Cycles-to-detach at the detachment threshold ``r = 1``.
    exponent:
        Steepness of the cycles-to-detach curve (Basquin-like).
    """

    endurance_ratio: float = 0.3
    reference_cycles: float = 10.0
    exponent: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.endurance_ratio < 1.0:
            raise ValueError("endurance_ratio must lie in (0, 1).")
        if self.reference_cycles <= 0.0:
            raise ValueError("reference_cycles must be positive.")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be positive.")

    def cycles_to_detach(self, load_ratio: float) -> float:
        """Return ``N(r)`` — cycles needed to detach at *load_ratio* ``r``."""
        if load_ratio >= 1.0:
            return 0.0
        if load_ratio <= self.endurance_ratio:
            return math.inf
        base = (1.0 - self.endurance_ratio) / (load_ratio - self.endurance_ratio)
        return self.reference_cycles * (base ** self.exponent)

    def detached(self, load_ratio: float, n_cycles: float) -> bool:
        """``True`` when *n_cycles* applied cycles detach a fruit at *load_ratio*."""
        if n_cycles <= 0.0:
            return False
        return n_cycles >= self.cycles_to_detach(load_ratio)


@dataclass(frozen=True)
class StressFatigueLaw:
    """Basquin S-N law: cycles-to-failure as a function of stress amplitude.

        N(σ) = reference_cycles · (σ / reference_stress) ** (-exponent)

    Used both for clamp-point fatigue (soft penalty) and branch fatigue fracture
    (hard constraint), with different constants.

    Parameters
    ----------
    reference_stress:
        Stress amplitude [Pa] at which failure occurs in ``reference_cycles``.
    reference_cycles:
        Cycles-to-failure at ``reference_stress``.
    exponent:
        Basquin exponent (slope of the S-N curve on log-log axes).
    """

    reference_stress: float
    reference_cycles: float = 1.0e6
    exponent: float = 6.0

    def __post_init__(self) -> None:
        if self.reference_stress <= 0.0:
            raise ValueError("reference_stress must be positive.")
        if self.reference_cycles <= 0.0:
            raise ValueError("reference_cycles must be positive.")
        if self.exponent <= 0.0:
            raise ValueError("exponent must be positive.")

    def cycles_to_failure(self, stress_pa: float) -> float:
        """Return ``N(σ)`` — cycles to failure at stress amplitude *stress_pa*."""
        if stress_pa <= 0.0:
            return math.inf
        return self.reference_cycles * (stress_pa / self.reference_stress) ** (-self.exponent)


# --------------------------------------------------------------------------- #
# Working parameters, configuration, result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HarvestParameters:
    """The working parameters being optimised.

    Parameters
    ----------
    frequency_hz:
        Excitation frequency [Hz].
    force_amplitude_n:
        Excitation force amplitude [N].
    duration_s:
        Working duration [s]; sets the applied cycle count ``n = f·T``.
    excitation_label:
        Identifier of the excitation position (e.g. ``"<branch>_<node>"``).
        Carried for bookkeeping; the response/stress inputs are assumed to have
        been computed for this position.
    """

    frequency_hz: float
    force_amplitude_n: float
    duration_s: float
    excitation_label: str = ""

    @property
    def n_cycles(self) -> float:
        """Applied cycle count ``n = f · T``."""
        return self.frequency_hz * self.duration_s


@dataclass(frozen=True)
class HarvestObjectiveConfig:
    """Tunable laws and thresholds for the harvest objective.

    Parameters
    ----------
    detachment_fatigue:
        Cumulative-detachment law.
    clamp_fatigue:
        S-N law for clamp-point fatigue (soft-penalty tier).
    clamp_penalty_weight:
        Weight ``λ`` of the clamp-damage soft penalty in the objective.
    branch_fatigue:
        S-N law for branch fatigue fracture (hard-constraint tier).
    branch_ultimate_stress_pa:
        Instantaneous fracture stress ``σ_ultimate`` [Pa]; exceeding it in a
        single cycle fractures the branch (the "crack and throw-off" failure).
    """

    detachment_fatigue: DetachmentFatigueLaw = field(default_factory=DetachmentFatigueLaw)
    clamp_fatigue: StressFatigueLaw = field(
        default_factory=lambda: StressFatigueLaw(reference_stress=8.0e6)
    )
    clamp_penalty_weight: float = 0.25
    branch_fatigue: StressFatigueLaw = field(
        default_factory=lambda: StressFatigueLaw(reference_stress=1.5e7)
    )
    branch_ultimate_stress_pa: float = 3.0e7


@dataclass(frozen=True)
class HarvestObjectiveResult:
    """Outcome of evaluating the harvest objective for one parameter set.

    Parameters
    ----------
    detached_fraction:
        Fraction of fruit removed within the working duration ∈ [0, 1].
    n_detached / total_fruits:
        Detached count and total.
    clamp_damage:
        Miner damage at the clamp ``n / N_clamp(σ_clamp)`` (≥ 1 ⇒ clamp failure).
    clamp_penalty:
        ``clamp_penalty_weight × clamp_damage`` subtracted from the objective.
    branch_fracture:
        ``True`` when the branch fractures (hard-constraint violation).
    fracture_mode:
        ``"overstress"``, ``"fatigue"``, or ``None``.
    feasible:
        ``not branch_fracture``.
    objective:
        ``detached_fraction − clamp_penalty`` when feasible, else ``-inf``.
    n_cycles:
        Applied cycle count ``f·T``.
    """

    detached_fraction: float
    n_detached: int
    total_fruits: int
    clamp_damage: float
    clamp_penalty: float
    branch_fracture: bool
    fracture_mode: str | None
    feasible: bool
    objective: float
    n_cycles: float


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_harvest_objective(
    load_ratios: list[float],
    params: HarvestParameters,
    *,
    branch_peak_stress_pa: float,
    clamp_stress_pa: float,
    config: HarvestObjectiveConfig | None = None,
    fruit_weights: list[float] | None = None,
) -> HarvestObjectiveResult:
    """Evaluate the time-dependent harvest objective for one parameter set.

    Parameters
    ----------
    load_ratios:
        Per-fruit load ratio ``r_i = F_inertia,i / F_detach,i`` at this
        ``(f, A, position)``.  Derive from a :class:`~orchard_fem.harvest.
        detachment.DetachmentResult` via :func:`load_ratios_from_detachment`
        (which applies the amplitude scaling for *A*).
    params:
        Working parameters; supplies ``n = f·T``.
    branch_peak_stress_pa:
        Peak dynamic bending-stress amplitude over the branch network [Pa] at
        this ``(f, A, position)`` — drives the fracture hard constraint.
    clamp_stress_pa:
        Dynamic stress amplitude at the clamp [Pa] — drives the soft penalty.
    config:
        Laws and thresholds; defaults to :class:`HarvestObjectiveConfig`.
    fruit_weights:
        Optional per-fruit weights (e.g. fruit count represented) for the
        detached-fraction average; defaults to uniform.

    Returns
    -------
    HarvestObjectiveResult
    """
    cfg = config or HarvestObjectiveConfig()
    n = params.n_cycles

    # --- Tier 0: branch fracture hard constraint ------------------------------
    if branch_peak_stress_pa >= cfg.branch_ultimate_stress_pa:
        fracture, mode = True, "overstress"
    elif n >= cfg.branch_fatigue.cycles_to_failure(branch_peak_stress_pa):
        fracture, mode = True, "fatigue"
    else:
        fracture, mode = False, None

    # --- Cumulative detachment over time --------------------------------------
    total = len(load_ratios)
    weights = fruit_weights if fruit_weights is not None else [1.0] * total
    if len(weights) != total:
        raise ValueError("fruit_weights length must match load_ratios.")

    detached_mask = [cfg.detachment_fatigue.detached(r, n) for r in load_ratios]
    n_detached = sum(1 for d in detached_mask if d)
    w_total = sum(weights)
    detached_fraction = (
        sum(w for w, d in zip(weights, detached_mask) if d) / w_total
        if w_total > 0.0
        else 0.0
    )

    # --- Clamp-point damage soft penalty --------------------------------------
    # cycles_to_failure → +inf at zero stress (never fails) ⇒ n / inf = 0 damage.
    clamp_cycles = cfg.clamp_fatigue.cycles_to_failure(clamp_stress_pa)
    clamp_damage = n / clamp_cycles
    clamp_penalty = cfg.clamp_penalty_weight * clamp_damage

    # --- Combine --------------------------------------------------------------
    feasible = not fracture
    objective = (detached_fraction - clamp_penalty) if feasible else -math.inf

    return HarvestObjectiveResult(
        detached_fraction=detached_fraction,
        n_detached=n_detached,
        total_fruits=total,
        clamp_damage=clamp_damage,
        clamp_penalty=clamp_penalty,
        branch_fracture=fracture,
        fracture_mode=mode,
        feasible=feasible,
        objective=objective,
        n_cycles=n,
    )


# --------------------------------------------------------------------------- #
# Adapters / helpers
# --------------------------------------------------------------------------- #


def load_ratios_from_detachment(
    detachment_result,
    force_amplitude_n: float,
    *,
    reference_amplitude_n: float = 1.0,
) -> list[float]:
    """Per-fruit load ratios scaled to the working force amplitude *A*.

    The :class:`~orchard_fem.harvest.detachment.DetachmentResult` is computed at
    a reference excitation amplitude (the FRF default is 1 N).  In the linear
    regime the inertia force scales linearly with the force amplitude, so

        r_i(A) = (F_inertia,i / F_detach,i) · (A / A_ref).

    In the nonlinear large-amplitude regime this linear scaling is an
    approximation to be replaced by the learned-closure amplitude-dependent
    response (see research plan §4).

    Parameters
    ----------
    detachment_result:
        A :class:`~orchard_fem.harvest.detachment.DetachmentResult`.
    force_amplitude_n:
        Working force amplitude ``A`` [N].
    reference_amplitude_n:
        Amplitude at which *detachment_result* was computed [N].

    Returns
    -------
    list[float]
        Per-fruit load ratios at amplitude ``A``.
    """
    if reference_amplitude_n <= 0.0:
        raise ValueError("reference_amplitude_n must be positive.")
    scale = force_amplitude_n / reference_amplitude_n
    ratios: list[float] = []
    for s in detachment_result.states:
        if s.detachment_force_n <= 0.0:
            ratios.append(math.inf if s.inertia_force_n > 0.0 else 0.0)
        else:
            ratios.append((s.inertia_force_n / s.detachment_force_n) * scale)
    return ratios


def scale_stress_with_amplitude(
    reference_stress_pa: float,
    force_amplitude_n: float,
    *,
    reference_amplitude_n: float = 1.0,
) -> float:
    """Linear-regime amplitude scaling for a dynamic stress amplitude.

    Placeholder until a curvature-based stress post-processor
    (``σ = E·κ·c`` from ``observation_complex``) is available.  Scales a stress
    computed at *reference_amplitude_n* to the working amplitude *A* assuming
    the linear ``σ ∝ A`` relationship.
    """
    if reference_amplitude_n <= 0.0:
        raise ValueError("reference_amplitude_n must be positive.")
    return reference_stress_pa * (force_amplitude_n / reference_amplitude_n)
