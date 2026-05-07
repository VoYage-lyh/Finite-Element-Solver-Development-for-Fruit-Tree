from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orchard_fem.domain import OrchardModel
from orchard_fem.topology import Vec3


def _point_key(point: Vec3) -> tuple[float, float, float]:
    return (
        round(point.x, 12),
        round(point.y, 12),
        round(point.z, 12),
    )


@dataclass(frozen=True)
class EmbeddedLineMeshSpec:
    points: list[tuple[float, float, float]]
    cells: list[tuple[int, int]]
    branch_ids: list[str]
    branch_element_indices: list[int]

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def cells_for_branch(self, branch_id: str) -> list[int]:
        return [
            index
            for index, candidate in enumerate(self.branch_ids)
            if candidate == branch_id
        ]


@dataclass(frozen=True)
class EmbeddedLineMeshArrays:
    points: np.ndarray
    cells: np.ndarray
    branch_ids: tuple[str, ...]
    branch_element_indices: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.cells.shape[0])


def build_embedded_line_mesh_spec(model: OrchardModel) -> EmbeddedLineMeshSpec:
    point_index_by_key: dict[tuple[float, float, float], int] = {}
    points: list[tuple[float, float, float]] = []
    cells: list[tuple[int, int]] = []
    branch_ids: list[str] = []
    branch_element_indices: list[int] = []

    def register_point(point: Vec3) -> int:
        key = _point_key(point)
        existing = point_index_by_key.get(key)
        if existing is not None:
            return existing

        index = len(points)
        points.append((point.x, point.y, point.z))
        point_index_by_key[key] = index
        return index

    for branch in model.branches:
        num_elements = max(branch.discretization.num_elements, 1)
        node_indices = [
            register_point(branch.path.point_at(node_index / num_elements))
            for node_index in range(num_elements + 1)
        ]

        for element_index in range(num_elements):
            first = node_indices[element_index]
            second = node_indices[element_index + 1]
            if first == second:
                raise ValueError(
                    f"Degenerate embedded mesh cell on branch '{branch.branch_id}' "
                    f"element {element_index}"
                )
            cells.append((first, second))
            branch_ids.append(branch.branch_id)
            branch_element_indices.append(element_index)

    return EmbeddedLineMeshSpec(
        points=points,
        cells=cells,
        branch_ids=branch_ids,
        branch_element_indices=branch_element_indices,
    )


def build_embedded_line_mesh_arrays(
    source: OrchardModel | EmbeddedLineMeshSpec,
) -> EmbeddedLineMeshArrays:
    spec = source if isinstance(source, EmbeddedLineMeshSpec) else build_embedded_line_mesh_spec(source)

    points = np.asarray(spec.points, dtype=np.float64, order="C")
    if points.size == 0:
        points = np.zeros((0, 3), dtype=np.float64)
    else:
        points = points.reshape((-1, 3))

    cells = np.asarray(spec.cells, dtype=np.int64, order="C")
    if cells.size == 0:
        cells = np.zeros((0, 2), dtype=np.int64)
    else:
        cells = cells.reshape((-1, 2))

    branch_element_indices = np.asarray(
        spec.branch_element_indices,
        dtype=np.int32,
        order="C",
    )

    return EmbeddedLineMeshArrays(
        points=points,
        cells=cells,
        branch_ids=tuple(spec.branch_ids),
        branch_element_indices=branch_element_indices,
    )


def create_dolfinx_embedded_line_mesh(
    source: OrchardModel | EmbeddedLineMeshSpec,
    *,
    comm: object | None = None,
    partitioner: object | None = None,
    max_facet_to_cell_links: int = 2,
):
    from orchard_fem.fenicsx.availability import require_dolfinx

    require_dolfinx()

    import basix
    import basix.ufl
    import ufl
    from mpi4py import MPI
    from dolfinx import mesh as dolfinx_mesh

    arrays = build_embedded_line_mesh_arrays(source)
    if arrays.cell_count == 0:
        raise ValueError("Cannot create a DOLFINx mesh from an empty embedded line mesh.")

    if comm is None:
        comm = MPI.COMM_WORLD

    coordinate_element = basix.ufl.element(
        basix.ElementFamily.P,
        "interval",
        1,
        basix.LagrangeVariant.equispaced,
        shape=(arrays.points.shape[1],),
        dtype=arrays.points.dtype,
    )
    domain = ufl.Mesh(coordinate_element)
    return dolfinx_mesh.create_mesh(
        comm,
        arrays.cells,
        domain,
        arrays.points,
        partitioner=partitioner,
        max_facet_to_cell_links=max_facet_to_cell_links,
    )
