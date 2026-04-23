from __future__ import annotations

from math import pi, sqrt

import pytest

from orchard_fem.verification import (
    build_uniform_planar_beam,
    solve_generalized_frequencies,
    solve_static_system,
)


def _circular_area(radius: float) -> float:
    return pi * radius * radius


def _circular_inertia(radius: float) -> float:
    return pi * radius**4 / 4.0


def _l2_norm(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values))


def test_python_cantilever_first_mode_matches_analytic_reference() -> None:
    pytest.importorskip("slepc4py")

    radius = 0.02
    area = _circular_area(radius)
    inertia = _circular_inertia(radius)
    system = build_uniform_planar_beam(
        num_elements=10,
        length=1.0,
        youngs_modulus=1.0e10,
        density=750.0,
        area=area,
        inertia=inertia,
    )

    frequency = solve_generalized_frequencies(
        system.stiffness,
        system.mass,
        fixed_dofs=[0, 1],
        count=1,
    )[0]

    beta_1 = 1.875104068711961
    expected_frequency = (
        (beta_1**2) * sqrt((1.0e10 * inertia) / (750.0 * area * (1.0**4))) / (2.0 * pi)
    )
    assert frequency == pytest.approx(expected_frequency, rel=0.05)


def test_python_cantilever_modal_convergence_matches_expected_rate() -> None:
    pytest.importorskip("slepc4py")

    radius = 0.02
    length = 1.0
    youngs_modulus = 1.0e10
    density = 800.0
    area = _circular_area(radius)
    inertia = _circular_inertia(radius)
    betas = [
        1.875104068711961,
        4.694091132974175,
        7.854757438237613,
    ]

    previous_errors: list[float] = []
    for num_elements in [2, 4, 8, 16, 32]:
        system = build_uniform_planar_beam(
            num_elements=num_elements,
            length=length,
            youngs_modulus=youngs_modulus,
            density=density,
            area=area,
            inertia=inertia,
        )
        frequencies = solve_generalized_frequencies(
            system.stiffness,
            system.mass,
            fixed_dofs=[0, 1],
            count=3,
        )

        relative_errors: list[float] = []
        for mode_index, beta in enumerate(betas):
            expected_frequency = (beta * beta) * sqrt(
                (youngs_modulus * inertia) / (density * area * (length**4))
            ) / (2.0 * pi)
            relative_errors.append(abs(frequencies[mode_index] - expected_frequency) / expected_frequency)

        if previous_errors:
            assert (_l2_norm(previous_errors) / _l2_norm(relative_errors)) >= 3.0
        previous_errors = relative_errors

        if num_elements == 32:
            for relative_error in relative_errors:
                assert relative_error < 0.005


def test_python_simple_beam_static_midspan_matches_analytic_reference() -> None:
    pytest.importorskip("petsc4py")

    length = 1.0
    youngs_modulus = 1.0e10
    density = 800.0
    radius = 0.02
    area = _circular_area(radius)
    inertia = _circular_inertia(radius)
    force = 100.0
    num_elements = 20

    system = build_uniform_planar_beam(
        num_elements=num_elements,
        length=length,
        youngs_modulus=youngs_modulus,
        density=density,
        area=area,
        inertia=inertia,
    )
    load = [0.0 for _ in range(len(system.stiffness))]
    mid_node = num_elements // 2
    load[2 * mid_node] = force

    displacement = solve_static_system(
        system.stiffness,
        load,
        fixed_dofs=[0, 2 * num_elements],
    )
    midspan_deflection = displacement[2 * mid_node]
    expected = force * (length**3) / (48.0 * youngs_modulus * inertia)
    assert midspan_deflection == pytest.approx(expected, rel=1.0e-3)
