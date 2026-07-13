from __future__ import annotations

import json
import os
from math import pi, sqrt

import pytest

from orchard_fem.fenicsx import (
    EmbeddedBeamFrequencyResponseExperimentResult,
    solve_embedded_beam_frequency_response_experiment,
)
from orchard_fem.io import load_orchard_model


def test_embedded_beam_frequency_response_result_types_are_stable() -> None:
    assert (
        EmbeddedBeamFrequencyResponseExperimentResult.__name__
        == "EmbeddedBeamFrequencyResponseExperimentResult"
    )


def _cantilever_payload() -> dict:
    return {
        "metadata": {"name": "fenicsx_cantilever_frequency_response"},
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
            "frequency_steps": 48,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 2.0e-4,
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


def _cantilever_with_fruit_payload() -> dict:
    payload = _cantilever_payload()
    payload["metadata"]["name"] = "fenicsx_cantilever_frequency_response_fruit"
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
            "id": "tip",
            "target_type": "branch",
            "target_id": "cantilever",
            "target_node": "tip",
            "target_component": "uy",
        },
        {
            "id": "fruit_obs",
            "target_type": "fruit",
            "target_id": "fruit_1",
        },
    ]
    return payload


def _cantilever_with_nonlinear_clamp_payload() -> dict:
    payload = _cantilever_payload()
    payload["metadata"]["name"] = "fenicsx_cantilever_nonlinear_frequency_response"
    payload["clamps"][0]["cubic_stiffness"] = 1.0e4
    payload["excitation"]["amplitude"] = 0.1
    payload["analysis"]["frequency_start_hz"] = 1.0
    payload["analysis"]["frequency_end_hz"] = 6.0
    payload["analysis"]["frequency_steps"] = 4
    payload["analysis"]["max_nonlinear_iterations"] = 8
    payload["analysis"]["nonlinear_tolerance"] = 1.0e-7
    return payload


def test_embedded_beam_frequency_response_experiment_smoke(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx frequency-response tests")

    model_path = tmp_path / "fenicsx_cantilever_frequency_response.json"
    model_path.write_text(json.dumps(_cantilever_payload()), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_frequency_response_experiment(
        model,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert len(result.result.points) == model.analysis.frequency_steps
    assert result.result.observation_names == ["tip"]
    assert result.response_mapping.excitation_dof >= 0
    assert result.response_mapping.observation_dofs
    assert all(point.frequency_hz > 0.0 for point in result.result.points)
    assert all(
        point.excitation_response_magnitude >= 0.0 for point in result.result.points
    )


def test_embedded_beam_frequency_response_peak_tracks_first_mode(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx frequency-response tests")

    model_path = tmp_path / "fenicsx_cantilever_frequency_response.json"
    model_path.write_text(json.dumps(_cantilever_payload()), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_frequency_response_experiment(
        model,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    peak_point = max(
        result.result.points,
        key=lambda point: point.observation_magnitudes[0],
    )

    radius = 0.02
    area = pi * radius * radius
    inertia = pi * (radius**4) / 4.0
    beta_1 = 1.875104068711961
    expected_frequency = (
        (beta_1**2) * sqrt((1.0e10 * inertia) / (750.0 * area * (1.0**4))) / (2.0 * pi)
    )
    assert peak_point.frequency_hz == pytest.approx(expected_frequency, rel=0.20)


def test_embedded_beam_frequency_response_supports_fruit_attachment(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx frequency-response tests")

    model_path = tmp_path / "fenicsx_cantilever_frequency_response_fruit.json"
    model_path.write_text(json.dumps(_cantilever_with_fruit_payload()), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_frequency_response_experiment(
        model,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert "fruit_1" in result.experiment.fruit_dofs
    assert result.result.observation_names == ["tip", "fruit_obs"]
    assert len(result.result.points) == model.analysis.frequency_steps
    assert max(point.observation_magnitudes[1] for point in result.result.points) >= 0.0


def test_embedded_beam_frequency_response_supports_harmonic_balance_nonlinear_links(tmp_path) -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx frequency-response tests")

    model_path = tmp_path / "fenicsx_cantilever_nonlinear_frequency_response.json"
    model_path.write_text(
        json.dumps(_cantilever_with_nonlinear_clamp_payload()),
        encoding="utf-8",
    )
    model = load_orchard_model(str(model_path))

    result = solve_embedded_beam_frequency_response_experiment(
        model,
        polynomial_degree=1,
        use_model_clamps=True,
    )

    assert result.experiment.operator_bundle.nonlinear_links
    assert result.result.observation_names == ["tip"]
    assert len(result.result.points) == model.analysis.frequency_steps
    assert all(point.observation_magnitudes[0] >= 0.0 for point in result.result.points)
