from __future__ import annotations

import json

import pytest

from orchard_fem.discretization import OrchardSystemAssembler
from orchard_fem.discretization import build_local_geometric_stiffness_matrix
from orchard_fem.io import load_orchard_model
from orchard_fem.solver_core import ModalAnalysisRequest, SLEPcModalSolver
from orchard_fem.topology import BranchPath, Vec3


def _vertical_cantilever_payload(include_gravity_prestress: bool) -> dict:
    return {
        "metadata": {"name": "vertical_gravity_cantilever"},
        "materials": [
            {
                "id": "xylem_default",
                "tissue": "xylem",
                "model": "linear",
                "density": 750.0,
                "youngs_modulus": 1.0e10,
                "poisson_ratio": 0.30,
                "damping_ratio": 0.002,
            },
            {
                "id": "pith_default",
                "tissue": "pith",
                "model": "linear",
                "density": 180.0,
                "youngs_modulus": 3.0e8,
                "poisson_ratio": 0.25,
                "damping_ratio": 0.04,
            },
            {
                "id": "phloem_default",
                "tissue": "phloem",
                "model": "linear",
                "density": 900.0,
                "youngs_modulus": 1.0e8,
                "poisson_ratio": 0.35,
                "damping_ratio": 0.06,
            },
        ],
        "branches": [
            {
                "id": "cantilever",
                "parent_branch_id": None,
                "level": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.0, 1.5],
                "discretization": {"num_elements": 16, "hotspot": False},
                "stations": [
                    {"s": 0.0, "shorthand": "circular", "outer_radius": 0.005},
                    {"s": 1.0, "shorthand": "circular", "outer_radius": 0.005},
                ],
            }
        ],
        "joints": [],
        "fruits": [],
        "clamps": [
            {
                "branch_id": "cantilever",
                "support_stiffness": 1.0,
                "support_damping": 0.0,
                "cubic_stiffness": 0.0,
            }
        ],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "cantilever",
            "target_node": "tip",
            "target_component": "uy",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "frequency_start_hz": 0.5,
            "frequency_end_hz": 4.5,
            "frequency_steps": 120,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 0.0,
            "include_gravity_prestress": include_gravity_prestress,
            "gravity_direction": [0.0, 0.0, -1.0],
            "output_csv": "unused.csv",
        },
        "observations": [
            {
                "id": "tip_uy",
                "target_type": "branch",
                "target_id": "cantilever",
                "target_node": "tip",
                "target_component": "uy",
            }
        ],
    }


def test_branch_path_inclination_angle_rad_matches_geometry() -> None:
    horizontal = BranchPath(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))
    vertical = BranchPath(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0))
    diagonal = BranchPath(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 1.0))

    assert horizontal.inclination_angle_rad() == pytest.approx(0.0)
    assert vertical.inclination_angle_rad() == pytest.approx(1.5707963267948966)
    assert diagonal.inclination_angle_rad() == pytest.approx(0.7853981633974483)


def test_local_geometric_stiffness_matrix_is_zero_for_zero_axial_force() -> None:
    matrix = build_local_geometric_stiffness_matrix(0.0, 1.2)
    assert all(value == pytest.approx(0.0) for row in matrix for value in row)


def test_gravity_prestress_adds_load_and_reduces_first_mode(tmp_path) -> None:
    pytest.importorskip("petsc4py")
    pytest.importorskip("slepc4py")

    baseline_path = tmp_path / "baseline.json"
    prestressed_path = tmp_path / "prestressed.json"
    baseline_path.write_text(json.dumps(_vertical_cantilever_payload(False)), encoding="utf-8")
    prestressed_path.write_text(json.dumps(_vertical_cantilever_payload(True)), encoding="utf-8")

    baseline_model = load_orchard_model(str(baseline_path))
    prestressed_model = load_orchard_model(str(prestressed_path))

    baseline_assembled = OrchardSystemAssembler().assemble(baseline_model)
    prestressed_assembled = OrchardSystemAssembler().assemble(prestressed_model)

    assert all(value == pytest.approx(0.0) for value in baseline_assembled.gravity_load)
    assert max(abs(value) for value in prestressed_assembled.gravity_load) > 0.0

    solver = SLEPcModalSolver()
    baseline_mode = solver.solve(
        ModalAnalysisRequest(
            num_modes=1,
            stiffness_matrix=baseline_assembled.stiffness_matrix,
            mass_matrix=baseline_assembled.mass_matrix,
            dof_labels=baseline_assembled.dof_labels,
        )
    )[0]
    prestressed_mode = solver.solve(
        ModalAnalysisRequest(
            num_modes=1,
            stiffness_matrix=prestressed_assembled.stiffness_matrix,
            mass_matrix=prestressed_assembled.mass_matrix,
            dof_labels=prestressed_assembled.dof_labels,
        )
    )[0]

    assert prestressed_mode.frequency_hz < baseline_mode.frequency_hz
