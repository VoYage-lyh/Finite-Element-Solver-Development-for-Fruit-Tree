"""Tests for beam-element stress recovery (no FEniCSx required).

Verified against the analytic Euler-Bernoulli cantilever: a single element,
node 1 clamped, transverse tip load P at node 2, has nodal displacements
``v2 = P L³/(3EI)``, ``θ2 = P L²/(2EI)``, for which the recovered root moment
must equal ``P·L`` and the tip moment must vanish.
"""
from __future__ import annotations

import math

import pytest

from orchard_fem.discretization.beam.types import BeamElementProperties
from orchard_fem.harvest.stress_recovery import (
    element_peak_stress,
    extreme_fibre_distance,
    recover_element_end_forces,
)


def _circular_props(radius: float, E: float, length: float) -> BeamElementProperties:
    area = math.pi * radius**2
    inertia = math.pi * radius**4 / 4.0
    return BeamElementProperties(
        youngs_modulus=E,
        shear_modulus=E / 2.6,
        area=area,
        iy=inertia,
        iz=inertia,
        torsion_constant=2.0 * inertia,
        density=600.0,
        length=length,
    )


def _cantilever_tip_load_displacements(P: float, E: float, inertia: float, L: float):
    """Local 12-vector for a clamped-node1 element under transverse tip load P (in y)."""
    v2 = P * L**3 / (3.0 * E * inertia)
    theta2 = P * L**2 / (2.0 * E * inertia)
    u = [0.0] * 12
    u[7] = v2        # uy at node 2
    u[11] = theta2   # rz at node 2
    return u


def test_cantilever_root_moment_equals_PL():
    P, E, L, r = 100.0, 1.0e10, 1.0, 0.05
    props = _circular_props(r, E, L)
    inertia = math.pi * r**4 / 4.0
    u = _cantilever_tip_load_displacements(P, E, inertia, L)

    forces = recover_element_end_forces(props, u)
    # Root bending moment magnitude = P·L; tip moment ≈ 0.
    assert abs(forces.moment_z_node1) == pytest.approx(P * L, rel=1e-9)
    assert abs(forces.moment_z_node2) == pytest.approx(0.0, abs=1e-6)
    # No axial, no torsion under pure transverse tip load.
    assert abs(forces.axial) == pytest.approx(0.0, abs=1e-9)
    assert abs(forces.torsion) == pytest.approx(0.0, abs=1e-9)


def test_cantilever_peak_stress_matches_Mc_over_I():
    P, E, L, r = 100.0, 1.0e10, 1.0, 0.05
    props = _circular_props(r, E, L)
    inertia = math.pi * r**4 / 4.0
    u = _cantilever_tip_load_displacements(P, E, inertia, L)

    sigma = element_peak_stress(props, u)
    # σ = M·c/I with M = P·L, c = r (circular).
    expected = (P * L) * r / inertia
    assert sigma == pytest.approx(expected, rel=1e-6)


def test_extreme_fibre_distance_circular_is_radius():
    r = 0.037
    area = math.pi * r**2
    inertia = math.pi * r**4 / 4.0
    assert extreme_fibre_distance(area, inertia) == pytest.approx(r, rel=1e-12)


def test_axial_stress_recovery():
    E, L, r = 1.0e10, 1.0, 0.05
    props = _circular_props(r, E, L)
    area = math.pi * r**2
    delta = 1.0e-4   # axial stretch at node 2
    u = [0.0] * 12
    u[6] = delta     # ux at node 2
    forces = recover_element_end_forces(props, u)
    # N = EA/L · δ ; σ_axial = N/A = E·δ/L.
    assert abs(forces.axial) == pytest.approx(E * area / L * delta, rel=1e-9)
    assert element_peak_stress(props, u) == pytest.approx(E * delta / L, rel=1e-9)


def test_complex_amplitudes_give_magnitude_stress():
    P, E, L, r = 100.0, 1.0e10, 1.0, 0.05
    props = _circular_props(r, E, L)
    inertia = math.pi * r**4 / 4.0
    base = _cantilever_tip_load_displacements(P, E, inertia, L)
    # Complex amplitude with a 30° phase: magnitude should match the real case.
    phase = complex(math.cos(math.radians(30)), math.sin(math.radians(30)))
    u = [complex(x) * phase for x in base]

    sigma = element_peak_stress(props, u)
    expected = (P * L) * r / inertia
    assert sigma == pytest.approx(expected, rel=1e-6)


def test_invalid_displacement_length():
    props = _circular_props(0.05, 1.0e10, 1.0)
    with pytest.raises(ValueError, match="12 components"):
        recover_element_end_forces(props, [0.0] * 6)


def test_extreme_fibre_invalid_area():
    with pytest.raises(ValueError, match="area"):
        extreme_fibre_distance(0.0, 1.0)
