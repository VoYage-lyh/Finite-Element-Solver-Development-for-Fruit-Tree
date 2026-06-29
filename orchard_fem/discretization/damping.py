from __future__ import annotations

from math import pi

from orchard_fem.domain import OrchardModel
from orchard_fem.materials.base import evaluate_branch_section_state


def rayleigh_from_band_zeta(
    zeta: float, f_lo_hz: float, f_hi_hz: float,
) -> tuple[float, float]:
    """Rayleigh ``(alpha, beta)`` giving damping ratio *zeta* at both band ends.

    Classic two-frequency tuning: ``zeta(w) = alpha/(2w) + beta*w/2`` is matched to
    *zeta* at ``f_lo`` and ``f_hi`` (dipping to ``2*zeta*sqrt(w1 w2)/(w1+w2)`` in
    between), so damping stays ~constant across the harvest band instead of the
    physically wrong β-only ramp (``zeta = beta*w/2``, rising linearly). Real
    green-wood ζ ≈ 0.05–0.15; the model's β=1e-4 gives only ~0.25 %.
    """
    if zeta < 0.0 or f_lo_hz <= 0.0 or f_hi_hz <= 0.0:
        raise ValueError("zeta must be >=0 and frequencies positive.")
    w1, w2 = 2.0 * pi * min(f_lo_hz, f_hi_hz), 2.0 * pi * max(f_lo_hz, f_hi_hz)
    if w1 == w2:
        return zeta * w1, zeta / w1
    alpha = 2.0 * zeta * w1 * w2 / (w1 + w2)
    beta = 2.0 * zeta / (w1 + w2)
    return alpha, beta


def paper_zeta_of_frequency(f_hz: float) -> float:
    """Frequency-dependent modal damping ratio for *Prunus cerasifera*.

    Power-law fit ``ζ = 0.883·f^-0.866`` (f in Hz) to the branch-hierarchy median
    damping ratios measured by free-decay on 15 trees in Liu et al., *Biosystems
    Engineering* 265 (2026) 104444 (Fig. 20): trunk ζ≈0.35 @≈2.75 Hz, primary
    ζ≈0.18 @≈7.45 Hz, secondary ζ≈0.14, tertiary ζ≈0.09 @≈12.5 Hz. Damping
    DECREASES with frequency (modal compartmentalisation — the trunk/low-frequency
    global modes dissipate the most energy, slender high-frequency tip modes the
    least). Clamped to the measured envelope [0.05, 0.40].
    """
    if f_hz <= 0.0:
        return 0.40
    zeta = 0.883 * f_hz ** (-0.866)
    return min(0.40, max(0.05, zeta))


def rayleigh_from_paper_zeta(f_lo_hz: float, f_hi_hz: float) -> tuple[float, float]:
    """Mass-proportional Rayleigh ``(alpha, 0)`` reproducing the paper's ζ(f) trend.

    Mass-proportional damping gives ``ζ(ω) = alpha/(2ω)`` — monotonically DECREASING
    with frequency, which matches the measured hierarchy (heavy at the
    low-frequency trunk, light at high-frequency tips) far better than two-point
    band tuning (:func:`rayleigh_from_band_zeta`, deliberately flat across the band,
    contradicting Fig. 20). ``alpha`` is calibrated so ζ equals the paper power law
    at the band's geometric-mean frequency; because each driven clamp responds
    narrow-band near its own resonance, this yields ≈the paper ζ at every drive
    frequency (exponent -1 vs the fitted -0.866 differ only mildly over 3–15 Hz).
    """
    f_lo, f_hi = min(f_lo_hz, f_hi_hz), max(f_lo_hz, f_hi_hz)
    f_ref = (f_lo * f_hi) ** 0.5 if f_lo > 0.0 else f_hi
    zeta_ref = paper_zeta_of_frequency(f_ref)
    alpha = 2.0 * zeta_ref * (2.0 * pi * f_ref)  # ζ = alpha/(2ω) ⇒ alpha = 4π·f·ζ
    return alpha, 0.0


def trapezoidal_average(states, getter) -> float:
    if not states:
        return 0.0
    if len(states) == 1:
        return getter(states[0])

    total = 0.0
    total_span = states[-1].station - states[0].station
    if total_span <= 1.0e-12:
        return getter(states[0])

    for left, right in zip(states, states[1:]):
        span = right.station - left.station
        total += 0.5 * (getter(left) + getter(right)) * span
    return total / total_span


def compute_default_damping_ratio(model: OrchardModel, material_lookup: dict[str, object]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0

    for branch in model.branches:
        profile_states = [
            evaluate_branch_section_state(branch, material_lookup, profile.station)
            for profile in branch.section_series.profiles
        ]
        average_mass = trapezoidal_average(profile_states, lambda state: state.mass_per_length)
        average_damping = trapezoidal_average(profile_states, lambda state: state.damping_ratio)
        branch_mass = average_mass * max(branch.path.length(), 1.0e-6)
        weighted_sum += branch_mass * average_damping
        total_weight += branch_mass

    return weighted_sum / total_weight if total_weight > 0.0 else 0.0


def apply_rayleigh_damping(
    model: OrchardModel,
    mass,
    stiffness,
    damping,
    material_lookup: dict[str, object],
) -> None:
    alpha = model.analysis.rayleigh_alpha
    beta = model.analysis.rayleigh_beta

    if abs(alpha) < 1.0e-14 and abs(beta) < 1.0e-14:
        zeta = compute_default_damping_ratio(model, material_lookup)
        omega_ref = 2.0 * 3.14159265358979323846 * max(model.analysis.frequency_start_hz, 0.1)
        beta = (2.0 * zeta / omega_ref) if omega_ref > 0.0 else 0.0

    for row_index in range(len(mass)):
        for column_index in range(len(mass[row_index])):
            damping[row_index][column_index] += (
                alpha * mass[row_index][column_index]
            ) + (
                beta * stiffness[row_index][column_index]
            )
