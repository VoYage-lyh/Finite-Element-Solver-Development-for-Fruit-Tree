from __future__ import annotations

import importlib.util
import os

import pytest

from orchard_fem.fenicsx import (
    build_embedded_beam_field_definition,
    create_embedded_beam_function_space,
    fenicsx_stack_available,
    missing_fenicsx_modules,
)
from orchard_fem.io import load_orchard_model


def test_missing_fenicsx_modules_matches_importlib() -> None:
    expected = tuple(
        module_name
        for module_name in ("dolfinx", "basix", "ufl", "mpi4py")
        if importlib.util.find_spec(module_name) is None
    )
    assert missing_fenicsx_modules() == expected
    assert fenicsx_stack_available() == (len(expected) == 0)


def test_embedded_beam_field_definition_shapes() -> None:
    if missing_fenicsx_modules():
        pytest.skip("FEniCSx stack not available in this environment")

    field_definition = build_embedded_beam_field_definition(
        geometric_dimension=3,
        polynomial_degree=1,
    )

    assert field_definition.geometric_dimension == 3
    assert field_definition.polynomial_degree == 1
    assert field_definition.displacement_element.reference_value_shape == (3,)
    assert field_definition.rotation_element.reference_value_shape == (3,)
    assert field_definition.mixed_element.num_sub_elements == 2


def test_create_embedded_beam_function_space_smoke() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx function-space tests")
    if missing_fenicsx_modules():
        pytest.skip("FEniCSx stack not available in this environment")

    model = load_orchard_model("examples/demo_orchard.json")
    bundle = create_embedded_beam_function_space(
        model,
        polynomial_degree=1,
    )

    assert bundle.mesh.topology.dim == 1
    assert bundle.mesh.geometry.dim == 3
    assert bundle.mixed_space.num_sub_spaces == 2
    assert bundle.displacement_space.ufl_element().reference_value_shape == (3,)
    assert bundle.rotation_space.ufl_element().reference_value_shape == (3,)
