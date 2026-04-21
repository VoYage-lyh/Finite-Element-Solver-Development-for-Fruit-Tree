import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from orchard_fem.cross_section.integrator import SectionIntegrator
from orchard_fem.cross_section.tissue import RegionGeometry, SectionShapeKind, TissueRegion, TissueType
from orchard_fem.io.json_schema import build_topology_from_legacy_model, load_legacy_model
from orchard_fem.io.legacy_loader import load_orchard_model
from orchard_fem.solvers.frequency_response import FrequencyResponseRequest, PETScFrequencyResponseSolver
from orchard_fem.solvers.modal import ModalAnalysisRequest, SLEPcModalSolver
from orchard_fem.solvers.modal_assembler import OrchardModalAssembler
from orchard_fem.solvers.time_history import PETScTimeHistorySolver, TimeHistoryRequest


def test_python_topology_loader_can_read_demo_model() -> None:
    payload = load_legacy_model("examples/demo_orchard.json")
    topology = build_topology_from_legacy_model(payload)
    valid, message = topology.validate()
    assert valid, message
    assert "trunk" in topology.roots()


def test_typed_model_loader_can_read_demo_model() -> None:
    model = load_orchard_model("examples/demo_orchard.json")
    assert model.metadata.name == "three_level_demo_tree"
    assert len(model.branches) == 4
    assert model.require_branch("trunk").discretization.num_elements == 5
    assert model.excitation.target_branch_id == "trunk"
    assert model.analysis.output_csv == "demo_frequency_response.csv"
    assert model.observations[0].target_components == ["ux"]


def test_python_section_integrator_matches_simple_ellipse_area() -> None:
    region = TissueRegion(
        tissue=TissueType.XYLEM,
        material_id="xylem_default",
        geometry=RegionGeometry(
            kind=SectionShapeKind.SOLID_ELLIPSE,
            center=(0.0, 0.0),
            radii=(0.02, 0.01),
            samples=96,
        ),
    )
    properties = SectionIntegrator.integrate([region])
    expected_area = 3.14159265358979323846 * 0.02 * 0.01
    assert abs(properties.total_area - expected_area) < expected_area * 0.03


def test_slepc_modal_solver_solves_simple_generalized_eigenproblem() -> None:
    pytest.importorskip("slepc4py")
    solver = SLEPcModalSolver()
    results = solver.solve(
        ModalAnalysisRequest(
            num_modes=2,
            stiffness_matrix=[[6.0, -2.0], [-2.0, 4.0]],
            mass_matrix=[[2.0, 0.0], [0.0, 1.0]],
            dof_labels=["u1", "u2"],
        )
    )

    assert len(results) == 2
    assert results[0].frequency_hz > 0.0
    assert results[1].frequency_hz > results[0].frequency_hz
    assert results[0].dof_labels == ["u1", "u2"]


def test_python_modal_assembler_matches_demo_dof_count() -> None:
    model = load_orchard_model("examples/demo_orchard.json")
    assembled = OrchardModalAssembler().assemble(model)

    expected_branch_dofs = sum(
        6 * (max(branch.discretization.num_elements, 1) + 1) for branch in model.branches
    )
    expected_total_dofs = expected_branch_dofs + len(model.fruits)

    assert len(assembled.dof_labels) == expected_total_dofs
    assert "trunk" in assembled.branch_nodes
    assert "fruit_left_primary" in assembled.fruit_dofs


def test_python_demo_modal_chain_runs() -> None:
    pytest.importorskip("slepc4py")
    model = load_orchard_model("examples/demo_orchard.json")
    assembled = OrchardModalAssembler().assemble(model)
    modes = SLEPcModalSolver().solve(
        ModalAnalysisRequest(
            num_modes=3,
            stiffness_matrix=assembled.stiffness_matrix,
            mass_matrix=assembled.mass_matrix,
            dof_labels=assembled.dof_labels,
        )
    )

    assert len(modes) == 3
    assert all(mode.frequency_hz > 0.0 for mode in modes)
    assert modes[0].frequency_hz < modes[1].frequency_hz < modes[2].frequency_hz


def test_petsc_frequency_response_solver_writes_demo_csv(tmp_path) -> None:
    pytest.importorskip("petsc4py")

    output_path = tmp_path / "demo_frequency_response.csv"
    result = PETScFrequencyResponseSolver().solve(
        FrequencyResponseRequest(
            model_path="examples/demo_orchard.json",
            output_csv=str(output_path),
        )
    )

    model = load_orchard_model("examples/demo_orchard.json")
    assert output_path.exists()
    assert len(result.points) == model.analysis.frequency_steps
    assert result.points[0].excitation_response_magnitude >= 0.0
    assert "obs_primary_left" in result.observation_names


def test_petsc_time_history_solver_writes_demo_csv(tmp_path) -> None:
    pytest.importorskip("petsc4py")

    output_path = tmp_path / "demo_time_history.csv"
    result = PETScTimeHistorySolver().solve(
        TimeHistoryRequest(
            model_path="examples/demo_orchard_time_history.json",
            output_csv=str(output_path),
        )
    )

    model = load_orchard_model("examples/demo_orchard_time_history.json")
    total_steps = max(1, round(model.analysis.total_time_seconds / model.analysis.time_step_seconds))
    output_stride = max(model.analysis.output_stride, 1)
    expected_points = 1 + sum(
        1 for step in range(1, total_steps + 1) if step % output_stride == 0 or step == total_steps
    )

    assert output_path.exists()
    assert len(result.points) == expected_points
    assert result.points[0].time_seconds == pytest.approx(0.0)
    assert result.points[0].excitation_response_value == pytest.approx(0.0)
    assert result.points[-1].time_seconds == pytest.approx(total_steps * model.analysis.time_step_seconds)
    assert "obs_trunk_ux" in result.observation_names
    assert "obs_trunk_uy" in result.observation_names


def _load_visualize_analysis_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "visualize_analysis.py"
    spec = importlib.util.spec_from_file_location("visualize_analysis", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_minimal_multicomponent_payload() -> dict:
    return {
        "metadata": {"name": "multicomponent_cantilever"},
        "materials": [
            {
                "id": "xylem_default",
                "tissue": "xylem",
                "model": "linear",
                "density": 750.0,
                "youngs_modulus": 1.0e10,
                "poisson_ratio": 0.30,
                "damping_ratio": 0.01,
            }
        ],
        "branches": [
            {
                "id": "cantilever",
                "parent_branch_id": None,
                "level": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.0, 1.0],
                "discretization": {"num_elements": 4, "hotspot": False},
                "stations": [
                    {
                        "s": 0.0,
                        "profile_type": "parameterized",
                        "regions": [
                            {
                                "tissue": "xylem",
                                "material_id": "xylem_default",
                                "shape": {
                                    "type": "solid_ellipse",
                                    "center": [0.0, 0.0],
                                    "radii": [0.02, 0.02],
                                    "samples": 96,
                                },
                            }
                        ],
                    },
                    {
                        "s": 1.0,
                        "profile_type": "parameterized",
                        "regions": [
                            {
                                "tissue": "xylem",
                                "material_id": "xylem_default",
                                "shape": {
                                    "type": "solid_ellipse",
                                    "center": [0.0, 0.0],
                                    "radii": [0.02, 0.02],
                                    "samples": 96,
                                },
                            }
                        ],
                    },
                ],
            }
        ],
        "joints": [],
        "fruits": [],
        "clamps": [{"branch_id": "cantilever", "support_stiffness": 1.0, "support_damping": 0.0, "cubic_stiffness": 0.0}],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "cantilever",
            "target_node": "tip",
            "target_component": "ux",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 8.0,
        },
        "analysis": {
            "mode": "time_history",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 20.0,
            "frequency_steps": 10,
            "time_step_seconds": 0.002,
            "total_time_seconds": 0.05,
            "output_stride": 1,
            "max_nonlinear_iterations": 10,
            "nonlinear_tolerance": 1.0e-8,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 0.0,
            "output_csv": "cantilever_time_history.csv",
        },
        "observations": [
            {
                "id": "obs_trunk",
                "target_type": "branch",
                "target_id": "cantilever",
                "target_node": "tip",
                "target_components": ["ux", "uy"],
            }
        ],
    }


def test_multicomponent_observation_time_history_expands_csv_columns(tmp_path) -> None:
    pytest.importorskip("petsc4py")

    model_path = tmp_path / "multicomponent.json"
    output_path = tmp_path / "multicomponent_time_history.csv"
    payload = _build_minimal_multicomponent_payload()
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    model = load_orchard_model(str(model_path))
    assert model.observations[0].target_components == ["ux", "uy"]

    PETScTimeHistorySolver().solve(
        TimeHistoryRequest(
            model_path=str(model_path),
            output_csv=str(output_path),
        )
    )

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert "obs_trunk_ux" in rows[0]
    assert "obs_trunk_uy" in rows[0]
    assert len(rows) > 2
    assert len(rows[1]) == len(rows[-1])


def test_plot_trajectory_writes_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")

    visualize_analysis = _load_visualize_analysis_module()
    output_path = tmp_path / "trajectory.png"
    headers = ["time_s", "obs_trunk_ux", "obs_trunk_uy"]
    rows = [
        [index * 0.01, math.sin(index * 0.1), math.cos(index * 0.1)]
        for index in range(64)
    ]

    visualize_analysis.plot_trajectory(headers, rows, output_path, "obs_trunk", show=False)
    assert output_path.exists()
