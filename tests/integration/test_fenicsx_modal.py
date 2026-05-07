from __future__ import annotations

import json
import os
from math import pi, sqrt

import pytest

from orchard_fem.fenicsx import (
    EmbeddedBeamModalExperimentResult,
    solve_embedded_beam_modal_experiment,
)
from orchard_fem.io import load_orchard_model


def test_embedded_beam_modal_result_types_are_stable() -> None:
    assert EmbeddedBeamModalExperimentResult.__name__ == "EmbeddedBeamModalExperimentResult"


def _cantilever_payload() -> dict:
    return {
        "metadata": {"name": "fenicsx_cantilever_modal"},
        "materials": [
            {
                "id": "xylem_default",
                "tissue": "xylem",
                "model": "linear",
                "density": 750.0,
                "youngs_modulus": 1.0e10,
                "poisson_ratio": 0.30,
                "damping_ratio": 0.0,
            }
        ],
        "branches": [
            {
                "id": "cantilever",
                "parent_branch_id": None,
                "level": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.0, 1.0],
                "discretization": {"num_elements": 24, "hotspot": False},
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
            "driving_frequency_hz": 1.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "frequency_start_hz": 0.5,
            "frequency_end_hz": 20.0,
            "frequency_steps": 50,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 0.0,
            "output_csv": "unused.csv",
        },
        "observations": [],
    }


def _cantilever_with_fruit_payload() -> dict:
    payload = _cantilever_payload()
    payload["metadata"]["name"] = "fenicsx_cantilever_modal_fruit"
    payload["fruits"] = [
        {
            "id": "fruit_1",
            "branch_id": "cantilever",
            "location_s": 1.0,
            "mass": 0.08,
            "stiffness": 1200.0,
            "damping": 0.5,
        }
    ]
    payload["observations"] = [
        {
            "id": "fruit_obs",
            "target_type": "fruit",
            "target_id": "fruit_1",
        }
    ]
    return payload


def _vertical_prestressed_payload(include_gravity_prestress: bool) -> dict:
    payload = _cantilever_payload()
    payload["metadata"]["name"] = "fenicsx_vertical_gravity_cantilever"
    payload["branches"][0]["end"] = [0.0, 0.0, 1.5]
    payload["branches"][0]["stations"] = [
        {"s": 0.0, "shorthand": "circular", "outer_radius": 0.005},
        {"s": 1.0, "shorthand": "circular", "outer_radius": 0.005},
    ]
    payload["materials"] = [
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
    ]
    payload["analysis"]["include_gravity_prestress"] = include_gravity_prestress
    payload["analysis"]["gravity_direction"] = [0.0, 0.0, -1.0]
    return payload


def _two_branch_joint_payload() -> dict:
    return {
        "metadata": {"name": "fenicsx_two_branch_joint_modal"},
        "materials": [
            {
                "id": "xylem_default",
                "tissue": "xylem",
                "model": "linear",
                "density": 750.0,
                "youngs_modulus": 1.0e10,
                "poisson_ratio": 0.30,
                "damping_ratio": 0.0,
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
                "id": "trunk",
                "parent_branch_id": None,
                "level": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.0, 1.0],
                "discretization": {"num_elements": 12, "hotspot": False},
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
                "end": [0.35, 0.0, 1.35],
                "discretization": {"num_elements": 8, "hotspot": False},
                "stations": [
                    {"s": 0.0, "shorthand": "circular", "outer_radius": 0.012},
                    {"s": 1.0, "shorthand": "circular", "outer_radius": 0.010},
                ],
            },
        ],
        "joints": [
            {
                "id": "joint_1",
                "parent_branch_id": "trunk",
                "child_branch_id": "branch_1",
                "linear_stiffness_scale": 1.0,
                "law": {"type": "none"},
            }
        ],
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
            "target_branch_id": "branch_1",
            "target_node": "tip",
            "target_component": "uy",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 1.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "frequency_start_hz": 0.5,
            "frequency_end_hz": 20.0,
            "frequency_steps": 50,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 0.0,
            "output_csv": "unused.csv",
        },
        "observations": [],
    }


def test_embedded_beam_modal_experiment_smoke() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx modal tests")

    model = load_orchard_model("examples/demo_orchard.json")
    result = solve_embedded_beam_modal_experiment(
        model,
        num_modes=1,
        polynomial_degree=1,
    )

    assert len(result.modes) == 1
    assert result.modes[0].eigenvalue > 0.0
    assert result.modes[0].frequency_hz > 0.0
    assert len(result.modes[0].mode_shape) > 0


def test_embedded_beam_cantilever_first_mode_matches_analytic_reference(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx modal tests")

    model_path = tmp_path / "fenicsx_cantilever_modal.json"
    model_path.write_text(json.dumps(_cantilever_payload()), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_modal_experiment(
        model,
        num_modes=1,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    radius = 0.02
    area = pi * radius * radius
    inertia = pi * (radius**4) / 4.0
    beta_1 = 1.875104068711961
    expected_frequency = (
        (beta_1**2) * sqrt((1.0e10 * inertia) / (750.0 * area * (1.0**4))) / (2.0 * pi)
    )
    assert result.modes[0].frequency_hz == pytest.approx(expected_frequency, rel=0.12)


def test_embedded_beam_modal_supports_fruit_attachment(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx modal tests")

    model_path = tmp_path / "fenicsx_cantilever_modal_fruit.json"
    model_path.write_text(json.dumps(_cantilever_with_fruit_payload()), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_modal_experiment(
        model,
        num_modes=1,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert "fruit_1" in result.experiment.fruit_dofs
    assert len(result.modes) == 1
    assert result.modes[0].frequency_hz > 0.0


def test_embedded_beam_modal_supports_gravity_prestress(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx modal tests")

    baseline_path = tmp_path / "fenicsx_baseline_prestress.json"
    prestressed_path = tmp_path / "fenicsx_prestressed.json"
    baseline_path.write_text(
        json.dumps(_vertical_prestressed_payload(False)),
        encoding="utf-8",
    )
    prestressed_path.write_text(
        json.dumps(_vertical_prestressed_payload(True)),
        encoding="utf-8",
    )

    baseline_model = load_orchard_model(str(baseline_path))
    prestressed_model = load_orchard_model(str(prestressed_path))

    baseline = solve_embedded_beam_modal_experiment(
        baseline_model,
        num_modes=1,
        polynomial_degree=1,
        use_model_clamps=True,
    )
    prestressed = solve_embedded_beam_modal_experiment(
        prestressed_model,
        num_modes=1,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert prestressed.experiment.operator_bundle.gravity_load_vector is not None
    assert prestressed.experiment.operator_bundle.geometric_stiffness_matrix is not None
    assert prestressed.experiment.operator_bundle.prestress_axial_forces["cantilever"]
    assert prestressed.modes[0].frequency_hz < baseline.modes[0].frequency_hz


def test_embedded_beam_modal_supports_linear_joint_constraint(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx modal tests")

    model_path = tmp_path / "fenicsx_two_branch_joint_modal.json"
    model_path.write_text(json.dumps(_two_branch_joint_payload()), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_modal_experiment(
        model,
        num_modes=1,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert len(result.modes) == 1
    assert result.modes[0].frequency_hz > 0.0
