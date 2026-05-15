"""Pareto-based work-parameter selection for vibration harvesting.

Two-objective version oriented at oil-tea (Camellia oleifera) harvesting where
every branch bears fruit:

* Maximise **detachment coverage** — the fraction of fruits whose inertia
  force exceeds the local attachment-detachment force at the chosen ``(f, A)``.
* Minimise **trunk peak bending stress** — the worst dynamic stress at the
  trunk-root section, computed from the FE curvature ``E·r·|κ|``. Since the
  dynamic curvature naturally scales with the input frequency near resonance,
  this captures the "higher frequency → more damage" intuition without an
  ad-hoc weighting coefficient.

Pareto front:

* Non-dominated set on the discrete ``(f, A)`` scan grid.
* Knee selection by the normalised-min-distance-to-utopia-point rule
  (Eq. 11 of the paper), independent of subjective weights.
* Optional hard-constraint filtering (``coverage ≥ min``, ``stress ≤ max``).
* Posterior propagation: aggregate Pareto knees across MCMC samples → 90 % CI.

The module is pure NumPy and decoupled from the FEM solver. The caller supplies
a callable that evaluates ``y`` for a given ``(params, f, A, clamp_node)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
#  Data types
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HarvestObjective:
    """Two-objective evaluation of a harvest work point.

    Attributes
    ----------
    detachment_coverage:
        Fraction of fruits (∈ [0, 1]) whose inertia force exceeds the local
        attachment-detachment force at the chosen ``(f, A)``. **Maximised.**
    trunk_max_stress:
        Peak dynamic bending stress at the trunk root [Pa], computed from the
        FE curvature: ``E · r_outer · |κ|`` where ``κ`` is the curvature
        magnitude vector ``√(κ_y² + κ_z²)``. **Minimised.**
    """

    detachment_coverage: float
    trunk_max_stress: float

    def as_minimisation_vector(self) -> np.ndarray:
        """Return the 2-vector used by :func:`non_dominated_mask`.

        ``detachment_coverage`` is negated so the whole vector is minimised
        (Pareto-dominance convention).
        """
        return np.array(
            [-self.detachment_coverage, self.trunk_max_stress], dtype=float
        )


@dataclass(frozen=True)
class ParetoKnee:
    """Knee point on a single Pareto front."""

    frequency_hz: float
    amplitude: float
    detachment_coverage: float
    trunk_max_stress: float


@dataclass(frozen=True)
class ParetoFront:
    """Pareto front at one candidate clamp point."""

    clamp_node: str
    frequencies_hz: np.ndarray             # (n_candidates,)
    amplitudes: np.ndarray                 # (n_candidates,)
    objectives: np.ndarray                 # (n_candidates, 3) — minimisation form
    non_dominated_index: np.ndarray        # indices into the candidate arrays
    knee_index: int                        # index into non_dominated_index

    @property
    def knee(self) -> ParetoKnee:
        i = self.non_dominated_index[self.knee_index]
        obj = self.objectives[i]
        return ParetoKnee(
            frequency_hz=float(self.frequencies_hz[i]),
            amplitude=float(self.amplitudes[i]),
            detachment_coverage=float(-obj[0]),
            trunk_max_stress=float(obj[1]),
        )


@dataclass(frozen=True)
class ParetoRecommendation:
    """Aggregated knee statistics over a posterior sample.

    Stores the 5%/50%/95% quantiles of every knee coordinate, matching the
    paper's Table 6 reporting format.
    """

    clamp_node: str
    frequency_hz_median: float
    frequency_hz_ci: tuple[float, float]
    amplitude_median: float
    amplitude_ci: tuple[float, float]
    detachment_coverage_median: float
    trunk_max_stress_median: float
    n_samples: int


# ────────────────────────────────────────────────────────────────────────────
#  Non-dominated sort
# ────────────────────────────────────────────────────────────────────────────
def non_dominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated points (minimisation form).

    Implements the canonical ``O(n²)`` dominance check. Adequate up to a few
    thousand candidates; for larger grids, swap in a fast non-dominated sort.

    Parameters
    ----------
    objectives:
        Shape ``(n, d)``. Each row is one candidate's objective vector
        (lower is better in every dimension).

    Returns
    -------
    np.ndarray
        Boolean array of length ``n``; ``True`` for non-dominated rows.
    """
    n = objectives.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # j dominates i if j ≤ i in every dim and < i in at least one dim.
        diff = objectives - objectives[i]
        dominates = np.all(diff <= 0.0, axis=1) & np.any(diff < 0.0, axis=1)
        if dominates.any():
            mask[i] = False
    return mask


def find_knee_min_distance(front_objectives: np.ndarray) -> int:
    """Return the index of the knee point on a non-dominated set.

    The knee is defined as the point with the smallest Euclidean norm after
    min–max normalisation of each objective on the front (Eq. 11 of the
    paper). This is invariant to objective scale and bias.

    Parameters
    ----------
    front_objectives:
        Shape ``(m, d)`` — only the non-dominated rows.
    """
    m = front_objectives.shape[0]
    if m == 0:
        raise ValueError("Empty Pareto front.")
    if m == 1:
        return 0
    lo = front_objectives.min(axis=0)
    hi = front_objectives.max(axis=0)
    span = np.where(hi - lo > 1.0e-15, hi - lo, 1.0)
    normalised = (front_objectives - lo) / span
    distances = np.linalg.norm(normalised, axis=1)
    return int(np.argmin(distances))


# ────────────────────────────────────────────────────────────────────────────
#  Front construction from a (params, freq, amp, clamp) grid
# ────────────────────────────────────────────────────────────────────────────
ObjectiveEvaluator = Callable[
    [dict[str, float], float, float, str], HarvestObjective
]


def pareto_front_from_grid(
    clamp_node: str,
    frequencies_hz: Sequence[float],
    amplitudes: Sequence[float],
    evaluator: ObjectiveEvaluator,
    params: dict[str, float],
    *,
    coverage_min: float | None = None,
    stress_max: float | None = None,
) -> ParetoFront:
    """Evaluate every ``(f, A)`` combo and extract the non-dominated set.

    Parameters
    ----------
    clamp_node:
        Identifier passed through to the evaluator (e.g. ``"trunk@0.50"``).
    frequencies_hz, amplitudes:
        Scan grids on the two work parameters.
    evaluator:
        Callable returning a :class:`HarvestObjective` for each grid point.
    params:
        Forward-operator parameter dict (e.g. one posterior sample).
    coverage_min, stress_max:
        Hard constraints applied as a feasible-region filter before
        non-dominated sorting (constraints decouple from Pareto ordering).

    Returns
    -------
    ParetoFront
        With both the feasible candidate cloud and the non-dominated subset.
    """
    n_f, n_a = len(frequencies_hz), len(amplitudes)
    n = n_f * n_a
    freq_arr = np.empty(n)
    amp_arr = np.empty(n)
    objectives = np.empty((n, 2))
    feasible = np.ones(n, dtype=bool)

    idx = 0
    for f in frequencies_hz:
        for A in amplitudes:
            obj = evaluator(params, float(f), float(A), clamp_node)
            freq_arr[idx] = f
            amp_arr[idx] = A
            objectives[idx] = obj.as_minimisation_vector()
            if coverage_min is not None and obj.detachment_coverage < coverage_min:
                feasible[idx] = False
            if stress_max is not None and obj.trunk_max_stress > stress_max:
                feasible[idx] = False
            idx += 1

    if not feasible.any():
        # No feasible point — return empty front so caller can detect.
        return ParetoFront(
            clamp_node=clamp_node,
            frequencies_hz=freq_arr,
            amplitudes=amp_arr,
            objectives=objectives,
            non_dominated_index=np.array([], dtype=int),
            knee_index=0,
        )

    feas_idx = np.flatnonzero(feasible)
    mask = non_dominated_mask(objectives[feas_idx])
    nd_global = feas_idx[mask]
    knee = find_knee_min_distance(objectives[nd_global])

    return ParetoFront(
        clamp_node=clamp_node,
        frequencies_hz=freq_arr,
        amplitudes=amp_arr,
        objectives=objectives,
        non_dominated_index=nd_global,
        knee_index=int(knee),
    )


# ────────────────────────────────────────────────────────────────────────────
#  Posterior propagation
# ────────────────────────────────────────────────────────────────────────────
def propagate_posterior_to_pareto(
    posterior_samples: Sequence[dict[str, float]],
    clamp_node: str,
    frequencies_hz: Sequence[float],
    amplitudes: Sequence[float],
    evaluator: ObjectiveEvaluator,
    *,
    credible_alpha: float = 0.90,
    coverage_min: float | None = None,
    stress_max: float | None = None,
    progress: bool = False,
) -> ParetoRecommendation:
    """Aggregate Pareto knees across posterior samples → (f*, A*) with CI.

    For each posterior sample we compute the Pareto front, take its knee, and
    accumulate the knee coordinates. The returned :class:`ParetoRecommendation`
    holds the median and centred *credible_alpha* CI on each coordinate —
    paper's Table 6 format.
    """
    if not 0.0 < credible_alpha < 1.0:
        raise ValueError("credible_alpha must lie in (0, 1).")
    lo_q = (1.0 - credible_alpha) / 2.0
    hi_q = 1.0 - lo_q

    knees_f = []
    knees_A = []
    knees_cov = []
    knees_stress = []
    iterator = enumerate(posterior_samples)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(posterior_samples),
                            desc=f"Pareto@{clamp_node}")
        except ImportError:
            pass
    for _, params in iterator:
        front = pareto_front_from_grid(
            clamp_node, frequencies_hz, amplitudes, evaluator, params,
            coverage_min=coverage_min,
            stress_max=stress_max,
        )
        if front.non_dominated_index.size == 0:
            continue
        knee = front.knee
        knees_f.append(knee.frequency_hz)
        knees_A.append(knee.amplitude)
        knees_cov.append(knee.detachment_coverage)
        knees_stress.append(knee.trunk_max_stress)

    if not knees_f:
        raise RuntimeError(
            "No posterior sample produced a feasible Pareto front. Relax "
            "constraints or check the evaluator."
        )

    arr_f = np.asarray(knees_f)
    arr_A = np.asarray(knees_A)
    return ParetoRecommendation(
        clamp_node=clamp_node,
        frequency_hz_median=float(np.median(arr_f)),
        frequency_hz_ci=(float(np.quantile(arr_f, lo_q)),
                         float(np.quantile(arr_f, hi_q))),
        amplitude_median=float(np.median(arr_A)),
        amplitude_ci=(float(np.quantile(arr_A, lo_q)),
                      float(np.quantile(arr_A, hi_q))),
        detachment_coverage_median=float(np.median(knees_cov)),
        trunk_max_stress_median=float(np.median(knees_stress)),
        n_samples=len(knees_f),
    )


__all__ = [
    "HarvestObjective",
    "ObjectiveEvaluator",
    "ParetoFront",
    "ParetoKnee",
    "ParetoRecommendation",
    "find_knee_min_distance",
    "non_dominated_mask",
    "pareto_front_from_grid",
    "propagate_posterior_to_pareto",
]
