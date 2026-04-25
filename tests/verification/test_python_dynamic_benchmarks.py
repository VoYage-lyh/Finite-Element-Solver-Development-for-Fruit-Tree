from __future__ import annotations

from math import pi, sqrt

import pytest

from orchard_fem.discretization import (
    LinearDynamicAssemblyResult,
    NonlinearLinkDefinition,
    NonlinearLinkKind,
)
from orchard_fem.domain import (
    AnalysisMode,
    AnalysisSettings,
    ExcitationKind,
    HarmonicExcitation,
)
from orchard_fem.dynamics import solve_frequency_response_system, solve_time_history_system
from orchard_fem.verification import build_hinged_two_bar_system, solve_generalized_frequencies


def _estimate_steady_amplitude(response) -> float:
    start_index = len(response.points) // 2
    return max(
        (abs(point.observation_values[0]) for point in response.points[start_index:]),
        default=0.0,
    )


def _build_duffing_system() -> LinearDynamicAssemblyResult:
    return LinearDynamicAssemblyResult(
        stiffness_matrix=[[100.0]],
        mass_matrix=[[1.0]],
        damping_matrix=[[0.5]],
        gravity_load=[0.0],
        dof_labels=["duffing"],
        branch_nodes={},
        branch_elements={},
        fruit_dofs={},
        nonlinear_links=[
            NonlinearLinkDefinition(
                label="duffing_ground",
                first_dof=0,
                second_dof=-1,
                kind=NonlinearLinkKind.CUBIC_SPRING,
                linear_stiffness=100.0,
                cubic_stiffness=2.0e4,
            )
        ],
        excitation_dof=0,
        observation_names=["x"],
        observation_dofs=[0],
    )


def test_python_duffing_peak_matches_backbone_estimate() -> None:
    pytest.importorskip("petsc4py")

    system = _build_duffing_system()
    analysis = AnalysisSettings(
        mode=AnalysisMode.FREQUENCY_RESPONSE,
        frequency_start_hz=1.35,
        frequency_end_hz=2.15,
        frequency_steps=21,
        time_step_seconds=0.001,
        total_time_seconds=25.0,
        output_stride=10,
        max_nonlinear_iterations=50,
        nonlinear_tolerance=1.0e-8,
        rayleigh_alpha=0.0,
        rayleigh_beta=0.0,
    )
    excitation = HarmonicExcitation(
        kind=ExcitationKind.HARMONIC_FORCE,
        target_branch_id="duffing",
        target_component="ux",
        amplitude=0.2,
        phase_degrees=0.0,
        driving_frequency_hz=0.0,
    )
    sweep = solve_frequency_response_system(system, excitation, analysis)
    peak_point = max(sweep.points, key=lambda point: point.observation_magnitudes[0])
    peak_frequency = peak_point.frequency_hz
    peak_amplitude = peak_point.observation_magnitudes[0]

    assert peak_amplitude > 0.0
    predicted_peak = sqrt(100.0 + (0.75 * 2.0e4 * peak_amplitude * peak_amplitude)) / (2.0 * pi)
    relative_error = abs(peak_frequency - predicted_peak) / predicted_peak
    linear_frequency = sqrt(100.0) / (2.0 * pi)
    assert relative_error < 0.05
    assert peak_frequency > linear_frequency * 1.03


def test_python_hinged_two_bar_matches_rigid_link_estimate() -> None:
    pytest.importorskip("slepc4py")

    first_length = 1.0
    second_length = 1.0
    rotational_stiffness = 500.0
    tip_mass = 1.0
    system = build_hinged_two_bar_system(
        first_length=first_length,
        second_length=second_length,
        youngs_modulus=1.0e15,
        density=1.0e-9,
        area=1.0e-2,
        inertia=1.0e-4,
        rotational_stiffness=rotational_stiffness,
        tip_mass=tip_mass,
    )

    frequency = solve_generalized_frequencies(
        system.stiffness,
        system.mass,
        fixed_dofs=[0, 1],
        count=1,
    )[0]
    expected = sqrt(rotational_stiffness / (tip_mass * second_length * second_length)) / (2.0 * pi)
    assert frequency == pytest.approx(expected, rel=0.03)
