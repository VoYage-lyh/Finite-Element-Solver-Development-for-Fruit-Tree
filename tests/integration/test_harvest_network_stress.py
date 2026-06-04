"""Tests for network-level stress aggregation (no FEniCSx required).

Two layers:
* aggregator unit tests on hand-built ``BranchElementState`` instances, verified
  against the analytic cantilever and against rotation invariance (exercising the
  ``u_local = T·u_global`` transform);
* an assembler integration test that the native assembler now populates element
  ``properties`` and extreme-fibre distances for stress recovery.
"""
from __future__ import annotations

import math

import pytest

from orchard_fem.discretization.beam.types import BeamElementProperties
from orchard_fem.discretization.types import BranchElementState
from orchard_fem.harvest.stress_recovery import (
    clamp_stress_from_solution,
    network_peak_stress,
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


def _identity12() -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(12)] for i in range(12)]


def _cantilever_local_u(P: float, E: float, inertia: float, L: float) -> list[float]:
    u = [0.0] * 12
    u[7] = P * L**3 / (3.0 * E * inertia)   # uy at node 2
    u[11] = P * L**2 / (2.0 * E * inertia)  # rz at node 2
    return u


def _make_element(props, *, branch_id, element_index, dofs, transform, c) -> BranchElementState:
    return BranchElementState(
        branch_id=branch_id,
        element_index=element_index,
        dofs=tuple(dofs),
        transformation_matrix=transform,
        length=props.length,
        axial_rigidity=props.youngs_modulus * props.area,
        properties=props,
        extreme_fibre_y=c,
        extreme_fibre_z=c,
    )


# ---------------------------------------------------------------------------
# Aggregator vs analytic cantilever (identity transform)
# ---------------------------------------------------------------------------

def test_network_peak_matches_cantilever():
    P, E, L, r = 100.0, 1.0e10, 1.0, 0.05
    props = _circular_props(r, E, L)
    inertia = math.pi * r**4 / 4.0
    elem = _make_element(
        props, branch_id="b", element_index=0,
        dofs=range(12), transform=_identity12(), c=r,
    )
    solution = _cantilever_local_u(P, E, inertia, L)

    res = network_peak_stress({"b": [elem]}, solution)
    expected = (P * L) * r / inertia
    assert res.peak_stress == pytest.approx(expected, rel=1e-6)
    assert res.peak_branch_id == "b"
    assert res.peak_element_index == 0
    assert res.per_branch_peak["b"] == pytest.approx(expected, rel=1e-6)


def test_clamp_stress_uses_root_element():
    P, E, L, r = 80.0, 1.0e10, 1.0, 0.04
    props = _circular_props(r, E, L)
    inertia = math.pi * r**4 / 4.0
    root = _make_element(props, branch_id="b", element_index=0,
                         dofs=range(12), transform=_identity12(), c=r)
    # A second (non-root) element with zero displacement → no stress.
    tip = _make_element(props, branch_id="b", element_index=1,
                        dofs=range(12, 24), transform=_identity12(), c=r)
    solution = _cantilever_local_u(P, E, inertia, L) + [0.0] * 12

    clamp = clamp_stress_from_solution({"b": [root, tip]}, {"b"}, solution)
    assert clamp == pytest.approx((P * L) * r / inertia, rel=1e-6)
    # Unclamped query returns 0.
    assert clamp_stress_from_solution({"b": [root, tip]}, set(), solution) == 0.0


def test_transform_rotation_invariance():
    """u_global = Tᵀ·u_local should recover the same stress through T·u_global."""
    from orchard_fem.discretization.beam.transforms import build_transformation_matrix
    from orchard_fem.topology import Vec3

    P, E, L, r = 100.0, 1.0e10, 1.0, 0.05
    props = _circular_props(r, E, L)
    inertia = math.pi * r**4 / 4.0
    # A non-axis-aligned element direction → non-trivial rotation T.
    T = build_transformation_matrix(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 0.5))
    u_local = _cantilever_local_u(P, E, inertia, L)
    # u_global = Tᵀ · u_local  (T orthonormal ⇒ T·u_global = u_local).
    u_global = [sum(T[j][i] * u_local[j] for j in range(12)) for i in range(12)]

    elem = _make_element(props, branch_id="b", element_index=0,
                         dofs=range(12), transform=T, c=r)
    res = network_peak_stress({"b": [elem]}, u_global)
    assert res.peak_stress == pytest.approx((P * L) * r / inertia, rel=1e-6)


def test_peak_picks_max_branch():
    P, E, L, r = 100.0, 1.0e10, 1.0, 0.05
    props = _circular_props(r, E, L)
    inertia = math.pi * r**4 / 4.0
    stressed = _make_element(props, branch_id="hot", element_index=0,
                             dofs=range(12), transform=_identity12(), c=r)
    quiet = _make_element(props, branch_id="cold", element_index=0,
                          dofs=range(12, 24), transform=_identity12(), c=r)
    solution = _cantilever_local_u(P, E, inertia, L) + [0.0] * 12

    res = network_peak_stress({"hot": [stressed], "cold": [quiet]}, solution)
    assert res.peak_branch_id == "hot"
    assert res.per_branch_peak["cold"] == pytest.approx(0.0, abs=1e-9)


def test_missing_properties_raises():
    bare = BranchElementState(
        branch_id="b", element_index=0, dofs=tuple(range(12)),
        transformation_matrix=_identity12(), length=1.0, axial_rigidity=1.0,
    )
    with pytest.raises(ValueError, match="no properties"):
        network_peak_stress({"b": [bare]}, [0.0] * 12)


# ---------------------------------------------------------------------------
# Assembler integration: new fields are populated
# ---------------------------------------------------------------------------

def test_assembler_populates_stress_recovery_fields():
    from orchard_fem.discretization import OrchardModalAssembler
    from orchard_fem.io import load_orchard_model

    model = load_orchard_model("examples/demo_orchard.json")
    assembled = OrchardModalAssembler().assemble(model)

    elements = assembled.branch_elements["trunk"]
    assert elements, "trunk should have elements"
    for el in elements:
        assert el.properties is not None
        assert el.extreme_fibre_y > 0.0
        assert el.extreme_fibre_z > 0.0

    n_dof = len(assembled.dof_labels)
    # Zero solution → zero stress everywhere.
    zero = network_peak_stress(assembled.branch_elements, [0.0] * n_dof)
    assert zero.peak_stress == pytest.approx(0.0, abs=1e-9)

    # A non-zero displacement field produces positive stress somewhere.
    bumped = [0.0] * n_dof
    bumped[assembled.branch_elements["trunk"][-1].dofs[7]] = 0.01
    nonzero = network_peak_stress(assembled.branch_elements, bumped)
    assert nonzero.peak_stress > 0.0
