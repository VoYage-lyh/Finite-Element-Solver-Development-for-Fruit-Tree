from __future__ import annotations

from orchard_fem.discretization.beam.local_matrices import (
    build_local_mass_matrix,
    build_local_stiffness_matrix,
)
from orchard_fem.discretization.beam.transforms import (
    build_transformation_matrix,
    transform_to_global,
)
from orchard_fem.discretization.beam.types import BeamElementProperties, Matrix
from orchard_fem.topology import Vec3


def build_global_element_matrices(
    properties: BeamElementProperties,
    start: Vec3,
    end: Vec3,
) -> tuple[Matrix, Matrix]:
    local_stiffness = build_local_stiffness_matrix(properties)
    local_mass = build_local_mass_matrix(properties)
    transformation = build_transformation_matrix(start, end)
    return (
        transform_to_global(local_stiffness, transformation),
        transform_to_global(local_mass, transformation),
    )
