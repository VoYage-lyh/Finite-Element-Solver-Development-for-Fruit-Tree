from __future__ import annotations

import json

import pytest

from orchard_fem.discretization import OrchardSystemAssembler
from orchard_fem.fenicsx import build_embedded_line_mesh_spec
from orchard_fem.io import load_orchard_model
from orchard_fem.io.skeleton_import import convert_skeleton_payload_to_orchard_payload
from orchard_fem.topology import BranchPath, Vec3


def _skeleton_payload() -> dict:
    return {
        "metadata": {"name": "skeleton_import_test"},
        "branches": [
            {
                "id": "trunk",
                "parent_branch_id": None,
                "level": 0,
                "points": [[0.0, 0.0, 0.0], [0.2, 0.0, 0.8], [0.0, 0.0, 1.6]],
                "outer_radius_root": 0.04,
                "outer_radius_tip": 0.025,
                "target_element_length": 0.4,
            }
        ],
        "fruits": [
            {
                "id": "fruit_tip",
                "branch_id": "trunk",
                "location_s": 1.0,
                "target_component": "uz",
                "mass": 0.2,
                "stiffness": 900.0,
                "damping": 0.3,
            }
        ],
        "clamps": [{"branch_id": "trunk", "support_stiffness": 1.0, "support_damping": 0.0}],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "trunk",
            "target_node": "tip",
            "target_component": "ux",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 2.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "solver_backend": "native",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 2.0,
            "frequency_steps": 2,
            "include_gravity_prestress": True,
            "output_csv": "unused.csv",
        },
        "observations": [
            {
                "id": "obs_tip",
                "target_type": "branch",
                "target_id": "trunk",
                "target_node": "tip",
                "target_components": ["ux", "uz"],
            },
            {"id": "obs_fruit", "target_type": "fruit", "target_id": "fruit_tip"},
        ],
    }


def test_branch_path_supports_polyline_arc_length_stationing() -> None:
    path = BranchPath(
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 1.0, 0.0),
        control_points=(Vec3(1.0, 0.0, 0.0),),
    )

    assert path.length() == pytest.approx(2.0)
    midpoint = path.point_at(0.5)
    assert midpoint.x == pytest.approx(1.0)
    assert midpoint.y == pytest.approx(0.0)
    assert midpoint.z == pytest.approx(0.0)


def test_skeleton_import_outputs_loadable_polyline_model_with_default_sections(tmp_path) -> None:
    output_payload = convert_skeleton_payload_to_orchard_payload(_skeleton_payload())
    model_path = tmp_path / "converted_orchard.json"
    model_path.write_text(json.dumps(output_payload), encoding="utf-8")

    model = load_orchard_model(str(model_path))

    assert model.branches[0].path.control_points
    assert model.branches[0].section_series.profiles[0].evaluate().total_area > 0.0
    assert model.fruits[0].target_component == "uz"


def test_fruit_gravity_load_is_applied_to_target_component(tmp_path) -> None:
    pytest.importorskip("petsc4py")

    output_payload = convert_skeleton_payload_to_orchard_payload(_skeleton_payload())
    model_path = tmp_path / "converted_orchard.json"
    model_path.write_text(json.dumps(output_payload), encoding="utf-8")

    model = load_orchard_model(str(model_path))
    assembled = OrchardSystemAssembler().assemble(model)
    fruit_dof = assembled.fruit_dofs["fruit_tip"]

    assert assembled.gravity_load[fruit_dof] == pytest.approx(-0.2 * 9.81)


def test_linear_child_branch_shares_coincident_parent_node(tmp_path) -> None:
    payload = _skeleton_payload()
    payload["branches"].append(
        {
            "id": "lateral",
            "parent_branch_id": "trunk",
            "level": 1,
            "points": [[0.0, 0.0, 1.6], [0.6, 0.0, 1.7]],
            "outer_radius": 0.015,
            "num_elements": 1,
        }
    )
    output_payload = convert_skeleton_payload_to_orchard_payload(payload)
    model_path = tmp_path / "linear_connection.json"
    model_path.write_text(json.dumps(output_payload), encoding="utf-8")

    model = load_orchard_model(str(model_path))
    spec = build_embedded_line_mesh_spec(model)

    assert spec.point_count == 7


def test_nonlinear_child_branch_keeps_independent_root_dof(tmp_path) -> None:
    payload = _skeleton_payload()
    payload["branches"].append(
        {
            "id": "lateral",
            "parent_branch_id": "trunk",
            "level": 1,
            "points": [[0.0, 0.0, 1.6], [0.6, 0.0, 1.7]],
            "outer_radius": 0.015,
            "num_elements": 1,
        }
    )
    payload["joints"] = [
        {
            "id": "joint_trunk_lateral",
            "parent_branch_id": "trunk",
            "child_branch_id": "lateral",
            "linear_stiffness_scale": 1.0,
            "law": {"type": "polynomial", "cubic_scale": 1.0e5},
        }
    ]
    output_payload = convert_skeleton_payload_to_orchard_payload(payload)
    model_path = tmp_path / "nonlinear_connection.json"
    model_path.write_text(json.dumps(output_payload), encoding="utf-8")

    model = load_orchard_model(str(model_path))
    spec = build_embedded_line_mesh_spec(model)

    assert spec.point_count == 8


def test_noncoincident_linear_child_branch_uses_dolfinx_mpc(tmp_path) -> None:
    pytest.importorskip("dolfinx")
    pytest.importorskip("dolfinx_mpc")

    from orchard_fem.fenicsx.assembly import assemble_fenicsx_system

    payload = _skeleton_payload()
    payload["analysis"]["solver_backend"] = "fenicsx"
    payload["branches"].append(
        {
            "id": "lateral",
            "parent_branch_id": "trunk",
            "level": 1,
            "points": [[0.05, 0.0, 1.45], [0.55, 0.0, 1.62]],
            "outer_radius": 0.015,
            "num_elements": 2,
        }
    )
    output_payload = convert_skeleton_payload_to_orchard_payload(payload)
    model_path = tmp_path / "noncoincident_linear_connection.json"
    model_path.write_text(json.dumps(output_payload), encoding="utf-8")

    model = load_orchard_model(str(model_path))
    assembly = assemble_fenicsx_system(model)

    assert assembly.experiment.operator_bundle.mpc is not None
    assert (
        "lateral"
        in assembly.experiment.operator_bundle.mpc_constrained_branch_ids
    )
    assert any(
        stage.name == "branch_connection_mpc" and stage.enabled
        for stage in assembly.stages
    )
