from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.embedded_mesh import (
    EmbeddedLineMeshSpec,
    create_dolfinx_embedded_line_mesh,
)


@dataclass(frozen=True)
class EmbeddedBeamFieldDefinition:
    geometric_dimension: int
    polynomial_degree: int
    displacement_element: Any
    rotation_element: Any
    mixed_element: Any


@dataclass(frozen=True)
class EmbeddedBeamFunctionSpaceBundle:
    mesh: Any
    field_definition: EmbeddedBeamFieldDefinition
    mixed_space: Any
    displacement_space: Any
    rotation_space: Any


def build_embedded_beam_field_definition(
    *,
    geometric_dimension: int = 3,
    polynomial_degree: int = 1,
    dtype: Any = np.float64,
) -> EmbeddedBeamFieldDefinition:
    require_dolfinx()

    if geometric_dimension <= 0:
        raise ValueError("geometric_dimension must be positive.")
    if polynomial_degree < 1:
        raise ValueError("polynomial_degree must be at least 1.")

    import basix
    import basix.ufl

    scalar_element = basix.ufl.element(
        basix.ElementFamily.P,
        "interval",
        polynomial_degree,
        basix.LagrangeVariant.equispaced,
        dtype=dtype,
    )
    displacement_element = basix.ufl.blocked_element(
        scalar_element,
        shape=(geometric_dimension,),
    )
    rotation_element = basix.ufl.blocked_element(
        scalar_element,
        shape=(geometric_dimension,),
    )
    mixed_element = basix.ufl.mixed_element(
        [displacement_element, rotation_element],
    )
    return EmbeddedBeamFieldDefinition(
        geometric_dimension=geometric_dimension,
        polynomial_degree=polynomial_degree,
        displacement_element=displacement_element,
        rotation_element=rotation_element,
        mixed_element=mixed_element,
    )


def create_embedded_beam_function_space(
    source: OrchardModel | EmbeddedLineMeshSpec | object,
    *,
    polynomial_degree: int = 1,
    comm: object | None = None,
    partitioner: object | None = None,
    max_facet_to_cell_links: int = 2,
) -> EmbeddedBeamFunctionSpaceBundle:
    require_dolfinx()

    from dolfinx import fem as dolfinx_fem

    mesh = (
        source
        if hasattr(source, "topology") and hasattr(source, "geometry")
        else create_dolfinx_embedded_line_mesh(
            source,
            comm=comm,
            partitioner=partitioner,
            max_facet_to_cell_links=max_facet_to_cell_links,
        )
    )

    field_definition = build_embedded_beam_field_definition(
        geometric_dimension=mesh.geometry.dim,
        polynomial_degree=polynomial_degree,
        dtype=mesh.geometry.x.dtype,
    )
    mixed_space = dolfinx_fem.functionspace(mesh, field_definition.mixed_element)
    return EmbeddedBeamFunctionSpaceBundle(
        mesh=mesh,
        field_definition=field_definition,
        mixed_space=mixed_space,
        displacement_space=mixed_space.sub(0),
        rotation_space=mixed_space.sub(1),
    )
