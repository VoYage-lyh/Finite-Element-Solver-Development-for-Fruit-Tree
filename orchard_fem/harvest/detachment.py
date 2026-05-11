"""Fruit detachment criterion for vibration harvesting.

Evaluates whether each fruit detaches under harmonic excitation at a given
frequency.  The inertia-force criterion is:

    F_inertia = m × ω² × |H(ω)|
    F_detach  = k × d_detach
    detached  ← F_inertia ≥ F_detach

where *|H(ω)|* is the compliance magnitude [m/N] of the fruit's attachment
node DOF taken from the batch FRF results, *k* is the fruit attachment
stiffness [N/m], and *d_detach* is the detachment displacement [m].

The detachment displacement is resolved in priority order:
1. Per-call keyword ``detachment_displacement_m``.
2. ``model.fruit_policy.detachment_displacement_m`` (if a policy exists).
3. Default 0.010 m (10 mm, typical for apples).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchard_fem.domain import OrchardModel
    from orchard_fem.workflows.batch_excitation import BatchExcitationResult

_DEFAULT_DETACHMENT_DISPLACEMENT_M = 0.010


@dataclass(frozen=True)
class FruitDetachmentState:
    """Detachment state of one fruit at a specific excitation frequency.

    Parameters
    ----------
    fruit_index:
        Index into ``model.fruits``.
    branch_id:
        Branch the fruit is attached to.
    location_s:
        Normalised arc-length position along the branch [0, 1].
    mass_kg:
        Fruit mass [kg].
    detachment_force_n:
        Required force to detach: ``k × d_detach`` [N].
    inertia_force_n:
        Applied inertia force: ``m × ω² × |H(ω)|`` [N].
    detached:
        ``True`` when ``inertia_force_n >= detachment_force_n``.
    """

    fruit_index: int
    branch_id: str
    location_s: float
    mass_kg: float
    detachment_force_n: float
    inertia_force_n: float
    detached: bool


@dataclass(frozen=True)
class DetachmentResult:
    """Detachment outcome at one frequency.

    Parameters
    ----------
    frequency_hz:
        Excitation frequency [Hz].
    n_detached:
        Number of fruits that detach.
    total_fruits:
        Total number of fruits in the model.
    detachment_fraction:
        ``n_detached / total_fruits`` ∈ [0, 1].
    states:
        Per-fruit detachment states.
    """

    frequency_hz: float
    n_detached: int
    total_fruits: int
    detachment_fraction: float
    states: list[FruitDetachmentState]


@dataclass(frozen=True)
class DetachmentSpectrum:
    """Detachment fraction vs. frequency sweep.

    Parameters
    ----------
    frequencies_hz:
        Frequency grid [Hz].
    detachment_fractions:
        Detachment fraction at each frequency.
    results:
        Full per-frequency detachment results.
    """

    frequencies_hz: list[float]
    detachment_fractions: list[float]
    results: list[DetachmentResult]


def _resolve_detachment_displacement(
    model: "OrchardModel",
    override: float | None,
) -> float:
    if override is not None:
        return override
    if model.fruit_policy is not None:
        return model.fruit_policy.detachment_displacement_m
    return _DEFAULT_DETACHMENT_DISPLACEMENT_M


def _find_best_response_at_frequency(
    batch_results: "list[BatchExcitationResult]",
    frequency_hz: float,
) -> float:
    """Return the max compliance magnitude across all excitation specs at *frequency_hz*.

    Linear interpolation is used when the queried frequency lies between grid points.
    """
    best = 0.0
    for br in batch_results:
        points = br.result.points
        if not points:
            continue
        freqs = [p.frequency_hz for p in points]
        mags = [p.excitation_response_magnitude for p in points]
        mag = _interp(frequency_hz, freqs, mags)
        if mag > best:
            best = mag
    return best


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def compute_detachment_at_frequency(
    model: "OrchardModel",
    batch_results: "list[BatchExcitationResult]",
    frequency_hz: float,
    *,
    detachment_displacement_m: float | None = None,
) -> DetachmentResult:
    """Evaluate fruit detachment at one frequency.

    Parameters
    ----------
    model:
        :class:`~orchard_fem.domain.OrchardModel` containing fruit attachments.
    batch_results:
        Output of :func:`~orchard_fem.workflows.batch_excitation.run_batch_frequency_response`.
    frequency_hz:
        Query frequency [Hz].
    detachment_displacement_m:
        Override detachment displacement [m].  If ``None``, resolved from the
        model fruit policy or the default 10 mm.

    Returns
    -------
    DetachmentResult
    """
    d_detach = _resolve_detachment_displacement(model, detachment_displacement_m)
    omega = 2.0 * math.pi * frequency_hz
    omega2 = omega * omega
    compliance = _find_best_response_at_frequency(batch_results, frequency_hz)

    states: list[FruitDetachmentState] = []
    for fruit_index, fruit in enumerate(model.fruits):
        f_detach = fruit.stiffness * d_detach
        f_inertia = fruit.mass * omega2 * compliance
        states.append(
            FruitDetachmentState(
                fruit_index=fruit_index,
                branch_id=fruit.branch_id,
                location_s=fruit.location_s,
                mass_kg=fruit.mass,
                detachment_force_n=f_detach,
                inertia_force_n=f_inertia,
                detached=f_inertia >= f_detach,
            )
        )

    n_detached = sum(1 for s in states if s.detached)
    total = len(states)
    fraction = n_detached / total if total > 0 else 0.0

    return DetachmentResult(
        frequency_hz=frequency_hz,
        n_detached=n_detached,
        total_fruits=total,
        detachment_fraction=fraction,
        states=states,
    )


def compute_detachment_spectrum(
    model: "OrchardModel",
    batch_results: "list[BatchExcitationResult]",
    *,
    detachment_displacement_m: float | None = None,
) -> DetachmentSpectrum:
    """Compute detachment fraction across the full FRF frequency grid.

    Uses the frequency grid from the first batch result.

    Parameters
    ----------
    model:
        Orchard model with fruit attachments.
    batch_results:
        Batch FRF results from
        :func:`~orchard_fem.workflows.batch_excitation.run_batch_frequency_response`.
    detachment_displacement_m:
        Override detachment displacement [m].

    Returns
    -------
    DetachmentSpectrum
    """
    if not batch_results or not batch_results[0].result.points:
        return DetachmentSpectrum(frequencies_hz=[], detachment_fractions=[], results=[])

    frequencies = [p.frequency_hz for p in batch_results[0].result.points]
    results: list[DetachmentResult] = []

    for freq in frequencies:
        dr = compute_detachment_at_frequency(
            model,
            batch_results,
            freq,
            detachment_displacement_m=detachment_displacement_m,
        )
        results.append(dr)

    return DetachmentSpectrum(
        frequencies_hz=frequencies,
        detachment_fractions=[r.detachment_fraction for r in results],
        results=results,
    )
