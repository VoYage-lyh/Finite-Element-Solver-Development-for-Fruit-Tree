from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from orchard_fem.discretization import LinearDynamicAssemblyResult
from orchard_fem.domain import ExcitationKind


@dataclass(frozen=True)
class TimeExcitationState:
    signal_value: float
    equivalent_load: float


def default_driving_frequency_hz(excitation, analysis) -> float:
    if excitation.driving_frequency_hz > 0.0:
        return excitation.driving_frequency_hz
    return max(analysis.frequency_start_hz, 0.1)


def build_time_excitation_state(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
    time_seconds: float,
) -> TimeExcitationState:
    excitation_dof = assembled.excitation_dof
    phase_radians = excitation.phase_degrees * (pi / 180.0)
    omega = 2.0 * pi * default_driving_frequency_hz(excitation, analysis)
    angle = (omega * time_seconds) + phase_radians
    displacement = excitation.amplitude * sin(angle)
    velocity = excitation.amplitude * omega * cos(angle)
    acceleration = -excitation.amplitude * omega * omega * sin(angle)

    if excitation.kind == ExcitationKind.HARMONIC_FORCE:
        return TimeExcitationState(signal_value=displacement, equivalent_load=displacement)

    if excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        equivalent_load = (
            (assembled.stiffness_matrix[excitation_dof][excitation_dof] * displacement)
            + (assembled.damping_matrix[excitation_dof][excitation_dof] * velocity)
            + (assembled.mass_matrix[excitation_dof][excitation_dof] * acceleration)
        )
        return TimeExcitationState(signal_value=displacement, equivalent_load=equivalent_load)

    if excitation.kind == ExcitationKind.HARMONIC_ACCELERATION:
        equivalent_load = assembled.mass_matrix[excitation_dof][excitation_dof] * acceleration
        return TimeExcitationState(signal_value=acceleration, equivalent_load=equivalent_load)

    raise ValueError(f"Unsupported excitation kind: {excitation.kind}")


def build_time_load_vector(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
    time_seconds: float,
) -> list[float]:
    load = [0.0 for _ in assembled.dof_labels]
    load[assembled.excitation_dof] = build_time_excitation_state(
        assembled,
        excitation,
        analysis,
        time_seconds,
    ).equivalent_load
    return load


def build_frequency_excitation_load(
    stiffness_matrix: list[list[float]],
    mass_matrix: list[list[float]],
    damping_matrix: list[list[float]],
    excitation_dof: int,
    excitation,
    omega: float,
) -> tuple[float, float]:
    phase_radians = excitation.phase_degrees * (pi / 180.0)
    base_real = excitation.amplitude * cos(phase_radians)
    base_imag = excitation.amplitude * sin(phase_radians)

    if excitation.kind == ExcitationKind.HARMONIC_FORCE:
        return base_real, base_imag

    if excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        real_scale = stiffness_matrix[excitation_dof][excitation_dof]
        imag_scale = omega * damping_matrix[excitation_dof][excitation_dof]
        return (
            (base_real * real_scale) - (base_imag * imag_scale),
            (base_real * imag_scale) + (base_imag * real_scale),
        )

    if excitation.kind == ExcitationKind.HARMONIC_ACCELERATION:
        scale = mass_matrix[excitation_dof][excitation_dof]
        return base_real * scale, base_imag * scale

    raise ValueError(f"Unsupported excitation kind: {excitation.kind}")
