from __future__ import annotations

from typing import Any

import numpy as np

from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle


def _point_marker(point: tuple[float, float, float], atol: float):
    def marker(x) -> np.ndarray:
        return np.logical_and.reduce(
            [
                np.isclose(x[0], point[0], atol=atol),
                np.isclose(x[1], point[1], atol=atol),
                np.isclose(x[2], point[2], atol=atol),
            ]
        )

    return marker


def build_point_clamp_boundary_conditions(
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    point: tuple[float, float, float],
    *,
    atol: float = 1.0e-8,
) -> list[Any]:
    require_dolfinx()

    from dolfinx import fem as dolfinx_fem

    displacement_space, _ = space_bundle.displacement_space.collapse()
    rotation_space, _ = space_bundle.rotation_space.collapse()
    marker = _point_marker(point, atol)
    displacement_dofs = dolfinx_fem.locate_dofs_geometrical(
        (space_bundle.displacement_space, displacement_space),
        marker,
    )
    rotation_dofs = dolfinx_fem.locate_dofs_geometrical(
        (space_bundle.rotation_space, rotation_space),
        marker,
    )

    dtype = space_bundle.mesh.geometry.x.dtype
    zero_vector = np.zeros(space_bundle.mesh.geometry.dim, dtype=dtype)
    return [
        dolfinx_fem.dirichletbc(
            zero_vector,
            displacement_dofs,
            space_bundle.displacement_space,
        ),
        dolfinx_fem.dirichletbc(
            zero_vector,
            rotation_dofs,
            space_bundle.rotation_space,
        ),
    ]


def build_model_clamp_boundary_conditions(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    *,
    atol: float = 1.0e-8,
) -> list[Any]:
    boundary_conditions: list[Any] = []
    seen_points: set[tuple[float, float, float]] = set()
    for clamp in model.clamps:
        branch = model.require_branch(clamp.branch_id)
        point = (
            branch.path.start.x,
            branch.path.start.y,
            branch.path.start.z,
        )
        if point in seen_points:
            continue
        seen_points.add(point)
        boundary_conditions.extend(
            build_point_clamp_boundary_conditions(
                space_bundle,
                point,
                atol=atol,
            )
        )
    return boundary_conditions
