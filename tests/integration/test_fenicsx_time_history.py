from __future__ import annotations

import json
import os

import pytest

from orchard_fem.fenicsx import (
    EmbeddedBeamTimeHistoryExperimentResult,
    solve_embedded_beam_time_history_experiment,
)
from orchard_fem.io import load_orchard_model


def test_embedded_beam_time_history_result_types_are_stable() -> None:
    assert (
        EmbeddedBeamTimeHistoryExperimentResult.__name__
        == "EmbeddedBeamTimeHistoryExperimentResult"
    )


def _cantilever_payload() -> dict:
    return {
        "metadata": {"name": "fenicsx_cantilever_time_history"},
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
                "discretization": {"num_elements": 16, "hotspot": False},
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
                                    "samples": 64,
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
                                    "samples": 64,
                                },
                            }
                        ],
                    },
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
            "driving_frequency_hz": 3.0,
        },
        "analysis": {
            "mode": "time_history",
            "time_step_seconds": 0.01,
            "total_time_seconds": 0.10,
            "output_stride": 1,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 1.0e-4,
            "output_csv": "unused.csv",
        },
        "observations": [
            {
                "id": "tip",
                "target_type": "branch",
                "target_id": "cantilever",
                "target_node": "tip",
                "target_component": "uy",
            }
        ],
    }


def _two_branch_polynomial_joint_payload() -> dict:
    payload = _cantilever_payload()
    payload["metadata"]["name"] = "fenicsx_two_branch_polynomial_joint_time_history"
    payload["materials"] = [
        {
            "id": "xylem_default",
            "tissue": "xylem",
            "model": "linear",
            "density": 750.0,
            "youngs_modulus": 1.0e10,
            "poisson_ratio": 0.30,
            "damping_ratio": 0.01,
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
    ]
    payload["branches"] = [
        {
            "id": "trunk",
            "parent_branch_id": None,
            "level": 0,
            "start": [0.0, 0.0, 0.0],
            "end": [0.0, 0.0, 1.0],
            "discretization": {"num_elements": 8, "hotspot": False},
            "stations": [
                {"s": 0.0, "shorthand": "circular", "outer_radius": 0.02},
                {"s": 1.0, "shorthand": "circular", "outer_radius": 0.02},
            ],
        },
        {
            "id": "branch_1",
            "parent_branch_id": "trunk",
            "level": 1,
            "start": [0.0, 0.0, 1.0],
            "end": [0.25, 0.0, 1.25],
            "discretization": {"num_elements": 6, "hotspot": False},
            "stations": [
                {"s": 0.0, "shorthand": "circular", "outer_radius": 0.012},
                {"s": 1.0, "shorthand": "circular", "outer_radius": 0.010},
            ],
        },
    ]
    payload["joints"] = [
        {
            "id": "joint_1",
            "parent_branch_id": "trunk",
            "child_branch_id": "branch_1",
            "linear_stiffness_scale": 1.0,
            "law": {
                "type": "polynomial",
                "linear_scale": 1.0,
                "cubic_scale": 2.0e5,
            },
        }
    ]
    payload["clamps"] = [
        {
            "branch_id": "trunk",
            "support_stiffness": 1.0,
            "support_damping": 0.0,
            "cubic_stiffness": 0.0,
        }
    ]
    payload["excitation"]["target_branch_id"] = "branch_1"
    payload["excitation"]["driving_frequency_hz"] = 2.0
    payload["analysis"]["total_time_seconds"] = 0.04
    payload["analysis"]["max_nonlinear_iterations"] = 8
    payload["observations"] = [
        {
            "id": "tip",
            "target_type": "branch",
            "target_id": "branch_1",
            "target_node": "tip",
            "target_component": "uy",
        }
    ]
    return payload


def _auto_and_clamp_nonlinear_payload() -> dict:
    payload = _two_branch_polynomial_joint_payload()
    payload["metadata"]["name"] = "fenicsx_auto_and_clamp_nonlinear_time_history"
    payload["joints"] = []
    payload["branches"][1]["level"] = 2
    payload["clamps"][0]["cubic_stiffness"] = 1.0e4
    payload["excitation"]["amplitude"] = 0.1
    payload["analysis"]["auto_nonlinear_levels"] = [2]
    payload["analysis"]["auto_nonlinear_cubic_scale"] = 1.0e4
    payload["analysis"]["total_time_seconds"] = 0.03
    return payload


def test_embedded_beam_time_history_experiment_smoke(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx time-history tests")

    model_path = tmp_path / "fenicsx_cantilever_time_history.json"
    model_path.write_text(json.dumps(_cantilever_payload()), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_time_history_experiment(
        model,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert result.result.observation_names == ["tip"]
    assert len(result.result.points) >= 2
    assert result.result.points[0].time_seconds == pytest.approx(0.0)
    assert result.result.points[-1].time_seconds == pytest.approx(
        model.analysis.total_time_seconds
    )
    assert result.response_mapping.excitation_dof >= 0


def test_embedded_beam_time_history_supports_polynomial_joint_law(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx time-history tests")

    model_path = tmp_path / "fenicsx_polynomial_joint_time_history.json"
    model_path.write_text(
        json.dumps(_two_branch_polynomial_joint_payload()),
        encoding="utf-8",
    )
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_time_history_experiment(
        model,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert result.experiment.operator_bundle.nonlinear_links
    assert result.result.observation_names == ["tip"]
    assert len(result.result.points) >= 2


def test_embedded_beam_time_history_supports_auto_and_clamp_nonlinear_links(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx time-history tests")

    model_path = tmp_path / "fenicsx_auto_and_clamp_nonlinear_time_history.json"
    model_path.write_text(
        json.dumps(_auto_and_clamp_nonlinear_payload()),
        encoding="utf-8",
    )
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_time_history_experiment(
        model,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    link_labels = {link.label for link in result.experiment.operator_bundle.nonlinear_links}
    assert "auto_joint:branch_1" in link_labels
    assert "clamp:trunk" in link_labels
    assert len(result.result.points) >= 2
