from __future__ import annotations

from orchard_fem.domain import OrchardModel
from orchard_fem.materials.base import evaluate_branch_section_state


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
