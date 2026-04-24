from __future__ import annotations

from orchard_fem.discretization.beam.algebra import multiply, transpose, zero_matrix12
from orchard_fem.discretization.beam.types import Matrix
from orchard_fem.topology import Vec3, dot, normalize


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return Vec3(
        x=(left.y * right.z) - (left.z * right.y),
        y=(left.z * right.x) - (left.x * right.z),
        z=(left.x * right.y) - (left.y * right.x),
    )


def build_transformation_matrix(
    start: Vec3,
    end: Vec3,
    preferred_up: Vec3 = Vec3(0.0, 0.0, 1.0),
) -> Matrix:
    local_x = normalize(Vec3(end.x - start.x, end.y - start.y, end.z - start.z))
    preferred_up = normalize(preferred_up)

    if abs(dot(local_x, preferred_up)) > 0.95:
        preferred_up = Vec3(0.0, 1.0, 0.0)
        if abs(dot(local_x, preferred_up)) > 0.95:
            preferred_up = Vec3(1.0, 0.0, 0.0)

    local_y = normalize(_cross(preferred_up, local_x))
    local_z = normalize(_cross(local_x, local_y))

    rotation = [
        [local_x.x, local_x.y, local_x.z],
        [local_y.x, local_y.y, local_y.z],
        [local_z.x, local_z.y, local_z.z],
    ]

    transformation = zero_matrix12()
    for block in range(4):
        offset = block * 3
        for row in range(3):
            for col in range(3):
                transformation[offset + row][offset + col] = rotation[row][col]
    return transformation


def transform_to_global(local_matrix: Matrix, transformation: Matrix) -> Matrix:
    return multiply(transpose(transformation), multiply(local_matrix, transformation))
