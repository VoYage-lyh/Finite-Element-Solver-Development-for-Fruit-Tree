from __future__ import annotations

import json

from orchard_fem.discretization import NonlinearLinkKind, OrchardSystemAssembler
from orchard_fem.io import load_orchard_model
from orchard_fem.topology import distance


def _auto_nonlinear_payload() -> dict:
    return {
        "metadata": {"name": "auto_nonlinear_tree"},
        "materials": [
            {
                "id": "xylem_default",
                "tissue": "xylem",
                "model": "linear",
                "density": 720.0,
                "youngs_modulus": 1.45e9,
                "poisson_ratio": 0.31,
                "damping_ratio": 0.035,
            },
            {
                "id": "pith_default",
                "tissue": "pith",
                "model": "linear",
                "density": 240.0,
                "youngs_modulus": 2.60e8,
                "poisson_ratio": 0.33,
                "damping_ratio": 0.060,
            },
            {
                "id": "phloem_default",
                "tissue": "phloem",
                "model": "linear",
                "density": 920.0,
                "youngs_modulus": 6.50e8,
                "poisson_ratio": 0.34,
                "damping_ratio": 0.055,
            },
        ],
        "branches": [
            {
                "id": "trunk",
                "parent_branch_id": None,
                "level": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.0, 1.2],
                "discretization": {"num_elements": 3, "hotspot": False},
                "stations": [
                    {"s": 0.0, "shorthand": "circular", "outer_radius": 0.05},
                    {"s": 1.0, "shorthand": "circular", "outer_radius": 0.04},
                ],
            },
            {
                "id": "primary",
                "parent_branch_id": "trunk",
                "level": 1,
                "start": [0.0, 0.0, 1.0],
                "end": [0.55, 0.0, 1.45],
                "discretization": {"num_elements": 2, "hotspot": False},
                "stations": [
                    {"s": 0.0, "shorthand": "circular", "outer_radius": 0.03},
                    {"s": 1.0, "shorthand": "circular", "outer_radius": 0.02},
                ],
            },
            {
                "id": "secondary",
                "parent_branch_id": "primary",
                "level": 2,
                "start": [0.55, 0.0, 1.45],
                "end": [0.85, 0.22, 1.75],
                "discretization": {"num_elements": 2, "hotspot": False},
                "stations": [
                    {"s": 0.0, "shorthand": "circular", "outer_radius": 0.018},
                    {"s": 1.0, "shorthand": "circular", "outer_radius": 0.014},
                ],
            },
            {
                "id": "tertiary",
                "parent_branch_id": "secondary",
                "level": 3,
                "start": [0.85, 0.22, 1.75],
                "end": [1.05, 0.35, 1.98],
                "discretization": {"num_elements": 2, "hotspot": False},
                "stations": [
                    {"s": 0.0, "shorthand": "circular", "outer_radius": 0.012},
                    {"s": 1.0, "shorthand": "circular", "outer_radius": 0.01},
                ],
            },
        ],
        "joints": [],
        "fruits": [],
        "clamps": [
            {
                "branch_id": "trunk",
                "support_stiffness": 1.0,
                "support_damping": 0.0,
                "cubic_stiffness": 0.0,
            }
        ],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "primary",
            "target_node": "tip",
            "target_component": "ux",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 6.0,
        },
        "analysis": {
            "mode": "time_history",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 12.0,
            "frequency_steps": 8,
            "time_step_seconds": 0.002,
            "total_time_seconds": 0.05,
            "output_stride": 1,
            "max_nonlinear_iterations": 12,
            "nonlinear_tolerance": 1.0e-8,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 1.0e-4,
            "auto_nonlinear_levels": [2, 3],
            "auto_nonlinear_cubic_scale": 2200000.0,
            "output_csv": "auto_nonlinear.csv",
        },
        "observations": [
            {
                "id": "obs_secondary",
                "target_type": "branch",
                "target_id": "secondary",
                "target_node": "tip",
                "target_component": "ux",
            }
        ],
    }


def _nearest_parent_ux_dof(assembled, child_branch_id: str, parent_branch_id: str) -> int:
    child_root = assembled.branch_nodes[child_branch_id][0]
    nearest_parent = min(
        assembled.branch_nodes[parent_branch_id],
        key=lambda node: distance(child_root.position, node.position),
    )
    return nearest_parent.dofs[0]


def test_auto_nonlinear_levels_inject_secondary_and_tertiary_links(tmp_path) -> None:
    model_path = tmp_path / "auto_nonlinear.json"
    model_path.write_text(json.dumps(_auto_nonlinear_payload()), encoding="utf-8")

    model = load_orchard_model(str(model_path))
    assembled = OrchardSystemAssembler().assemble(model)

    assert model.analysis.auto_nonlinear_levels == [2, 3]
    assert model.analysis.auto_nonlinear_cubic_scale == 2200000.0

    links_by_label = {link.label: link for link in assembled.nonlinear_links}
    assert set(links_by_label) == {"auto_joint:secondary", "auto_joint:tertiary"}

    secondary_link = links_by_label["auto_joint:secondary"]
    assert secondary_link.kind == NonlinearLinkKind.CUBIC_SPRING
    assert secondary_link.first_dof == assembled.branch_nodes["secondary"][0].dofs[0]
    assert secondary_link.second_dof == _nearest_parent_ux_dof(assembled, "secondary", "primary")
    assert secondary_link.cubic_stiffness == 2200000.0

    tertiary_link = links_by_label["auto_joint:tertiary"]
    assert tertiary_link.kind == NonlinearLinkKind.CUBIC_SPRING
    assert tertiary_link.first_dof == assembled.branch_nodes["tertiary"][0].dofs[0]
    assert tertiary_link.second_dof == _nearest_parent_ux_dof(assembled, "tertiary", "secondary")
    assert tertiary_link.cubic_stiffness == 2200000.0


def test_explicit_joint_blocks_duplicate_auto_injection(tmp_path) -> None:
    payload = _auto_nonlinear_payload()
    payload["joints"] = [
        {
            "id": "joint_secondary",
            "parent_branch_id": "primary",
            "child_branch_id": "secondary",
            "linear_stiffness_scale": 1.0,
        }
    ]

    model_path = tmp_path / "auto_nonlinear_explicit_joint.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    model = load_orchard_model(str(model_path))
    assembled = OrchardSystemAssembler().assemble(model)

    labels = {link.label for link in assembled.nonlinear_links}
    assert labels == {"auto_joint:tertiary"}
