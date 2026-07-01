from __future__ import annotations

import os

import pytest

from orchard_fem.fenicsx import (
    build_model_clamp_boundary_conditions,
    build_point_clamp_boundary_conditions,
    create_embedded_beam_function_space,
)
from orchard_fem.io import load_orchard_model


def test_fenicsx_boundary_condition_helpers_are_exported() -> None:
    assert build_point_clamp_boundary_conditions is not None
    assert build_model_clamp_boundary_conditions is not None


def test_build_point_clamp_boundary_conditions_smoke() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx boundary-condition tests")

    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    space_bundle = create_embedded_beam_function_space(model, polynomial_degree=1)
    point = (
        model.require_branch("trunk").path.start.x,
        model.require_branch("trunk").path.start.y,
        model.require_branch("trunk").path.start.z,
    )
    bcs = build_point_clamp_boundary_conditions(space_bundle, point)
    assert len(bcs) == 2


def test_build_model_clamp_boundary_conditions_smoke() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx boundary-condition tests")

    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    space_bundle = create_embedded_beam_function_space(model, polynomial_degree=1)
    bcs = build_model_clamp_boundary_conditions(model, space_bundle)
    assert len(bcs) >= 2
