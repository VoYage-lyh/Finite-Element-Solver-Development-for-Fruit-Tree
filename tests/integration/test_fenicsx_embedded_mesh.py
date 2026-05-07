from __future__ import annotations

import importlib.util
import os

import pytest

from orchard_fem.fenicsx import (
    build_embedded_line_mesh_arrays,
    build_embedded_line_mesh_spec,
    create_dolfinx_embedded_line_mesh,
    require_dolfinx,
)
from orchard_fem.io import load_orchard_model


def test_embedded_line_mesh_spec_matches_branch_element_count() -> None:
    model = load_orchard_model("examples/demo_orchard.json")
    spec = build_embedded_line_mesh_spec(model)

    expected_cells = sum(max(branch.discretization.num_elements, 1) for branch in model.branches)
    assert spec.cell_count == expected_cells
    assert len(spec.branch_ids) == expected_cells
    assert len(spec.branch_element_indices) == expected_cells
    assert spec.point_count >= len(model.branches) + 1


def test_embedded_line_mesh_spec_tracks_branch_cell_ranges() -> None:
    model = load_orchard_model("examples/demo_orchard.json")
    spec = build_embedded_line_mesh_spec(model)

    trunk = model.require_branch("trunk")
    primary_left = model.require_branch("primary_left")
    assert len(spec.cells_for_branch("trunk")) == max(trunk.discretization.num_elements, 1)
    assert len(spec.cells_for_branch("primary_left")) == max(
        primary_left.discretization.num_elements,
        1,
    )


def test_embedded_line_mesh_arrays_have_expected_shapes_and_dtypes() -> None:
    model = load_orchard_model("examples/demo_orchard.json")
    spec = build_embedded_line_mesh_spec(model)
    arrays = build_embedded_line_mesh_arrays(spec)

    assert arrays.points.shape == (spec.point_count, 3)
    assert arrays.cells.shape == (spec.cell_count, 2)
    assert arrays.branch_element_indices.shape == (spec.cell_count,)
    assert arrays.branch_ids == tuple(spec.branch_ids)
    assert arrays.points.dtype.kind == "f"
    assert arrays.cells.dtype.kind == "i"
    assert arrays.branch_element_indices.dtype.kind == "i"


def test_require_dolfinx_raises_clean_error_without_backend() -> None:
    if importlib.util.find_spec("dolfinx") is not None:
        pytest.skip("DOLFINx backend available; missing-backend path not applicable")

    with pytest.raises(RuntimeError, match="FEniCSx backend is not available"):
        require_dolfinx()


def test_create_dolfinx_embedded_line_mesh_smoke() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx mesh smoke tests")
    if importlib.util.find_spec("dolfinx") is None:
        pytest.skip("DOLFINx backend not available")

    model = load_orchard_model("examples/demo_orchard.json")
    spec = build_embedded_line_mesh_spec(model)
    mesh = create_dolfinx_embedded_line_mesh(spec)

    assert mesh.topology.dim == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.index_map(1).size_local >= 0
