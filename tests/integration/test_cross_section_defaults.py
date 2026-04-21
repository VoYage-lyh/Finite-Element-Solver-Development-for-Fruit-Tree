from __future__ import annotations

import json
import math

import pytest

from orchard_fem.cross_section.defaults import make_circular_section, make_default_branch_sections
from orchard_fem.cross_section.scan_loader import load_scan_profiles
from orchard_fem.io.legacy_loader import load_orchard_model


def _shorthand_payload() -> dict:
    return {
        "metadata": {"name": "circular_shorthand_smoke"},
        "materials": [
            {
                "id": "xylem_default",
                "tissue": "xylem",
                "model": "linear",
                "density": 700.0,
                "youngs_modulus": 9.0e9,
                "poisson_ratio": 0.3,
                "damping_ratio": 0.02,
            },
            {
                "id": "pith_default",
                "tissue": "pith",
                "model": "linear",
                "density": 180.0,
                "youngs_modulus": 4.0e8,
                "poisson_ratio": 0.25,
                "damping_ratio": 0.04,
            },
            {
                "id": "phloem_default",
                "tissue": "phloem",
                "model": "linear",
                "density": 950.0,
                "youngs_modulus": 1.5e8,
                "poisson_ratio": 0.35,
                "damping_ratio": 0.08,
            },
        ],
        "branches": [
            {
                "id": "trunk",
                "parent_branch_id": None,
                "level": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.0, 1.0],
                "stations": [
                    {"s": 0.0, "shorthand": "circular", "outer_radius": 0.025},
                    {"s": 1.0, "shorthand": "circular", "outer_radius": 0.020},
                ],
                "discretization": {"num_elements": 2, "hotspot": False},
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
            "target_branch_id": "trunk",
            "target_node": "tip",
            "target_component": "ux",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 5.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 10.0,
            "frequency_steps": 5,
            "output_csv": "unused.csv",
        },
        "observations": [
            {
                "id": "obs_trunk",
                "target_type": "branch",
                "target_id": "trunk",
                "target_node": "tip",
                "target_component": "ux",
            }
        ],
    }


def test_make_circular_section_matches_outer_circle_area() -> None:
    outer_radius = 0.025
    profile = make_circular_section(station=0.0, outer_radius=outer_radius)

    properties = profile.evaluate()
    assert profile.descriptor() == "parameterized"
    assert len(profile.regions) == 3
    assert properties.total_area == pytest.approx(math.pi * outer_radius**2, rel=0.01)


def test_make_default_branch_sections_builds_tapered_series() -> None:
    series = make_default_branch_sections(
        outer_radius_root=0.03,
        outer_radius_tip=0.015,
        num_stations=3,
    )

    assert series.stations() == pytest.approx([0.0, 0.5, 1.0])
    assert len(series.profiles) == 3
    assert series.profiles[0].evaluate().total_area > series.profiles[-1].evaluate().total_area


def test_load_orchard_model_supports_circular_shorthand(tmp_path) -> None:
    model_path = tmp_path / "circular_shorthand.json"
    model_path.write_text(json.dumps(_shorthand_payload()), encoding="utf-8")

    model = load_orchard_model(str(model_path))

    assert len(model.branches) == 1
    assert len(model.branches[0].section_series.profiles) == 2
    first_profile = model.branches[0].section_series.profiles[0]
    assert first_profile.evaluate().total_area == pytest.approx(math.pi * 0.025**2, rel=0.01)


def test_circular_shorthand_requires_default_materials(tmp_path) -> None:
    payload = _shorthand_payload()
    payload["materials"] = payload["materials"][:1]

    model_path = tmp_path / "missing_materials.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Circular shorthand requires materials"):
        load_orchard_model(str(model_path))


def test_load_scan_profiles_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="3D-scan integration"):
        load_scan_profiles("profiles.csv")
