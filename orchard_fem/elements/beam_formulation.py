from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from orchard_fem.topology.tree import Vec3, dot, normalize


Matrix = list[list[float]]


@dataclass(frozen=True)
class BeamElementProperties:
    youngs_modulus: float
    shear_modulus: float
    area: float
    iy: float
    iz: float
    torsion_constant: float
    density: float
    length: float


def zero_matrix12() -> Matrix:
    return [[0.0 for _ in range(12)] for _ in range(12)]


def _transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix)]


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    inner = len(left[0])
    cols = len(right[0])
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            result[row][col] = sum(left[row][index] * right[index][col] for index in range(inner))
    return result


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return Vec3(
        x=(left.y * right.z) - (left.z * right.y),
        y=(left.z * right.x) - (left.x * right.z),
        z=(left.x * right.y) - (left.y * right.x),
    )


def _add_submatrix(target: Matrix, indices: list[int], source: list[list[float]]) -> None:
    for row in range(4):
        for col in range(4):
            target[indices[row]][indices[col]] += source[row][col]


def _build_plane_bending_stiffness(flexural_rigidity: float, length: float) -> list[list[float]]:
    l2 = length * length
    l3 = l2 * length
    scale = flexural_rigidity / l3
    return [
        [12.0 * scale, 6.0 * length * scale, -12.0 * scale, 6.0 * length * scale],
        [6.0 * length * scale, 4.0 * l2 * scale, -6.0 * length * scale, 2.0 * l2 * scale],
        [-12.0 * scale, -6.0 * length * scale, 12.0 * scale, -6.0 * length * scale],
        [6.0 * length * scale, 2.0 * l2 * scale, -6.0 * length * scale, 4.0 * l2 * scale],
    ]


def _build_plane_bending_mass(density_area: float, length: float) -> list[list[float]]:
    l2 = length * length
    scale = density_area * length / 420.0
    return [
        [156.0 * scale, 22.0 * length * scale, 54.0 * scale, -13.0 * length * scale],
        [22.0 * length * scale, 4.0 * l2 * scale, 13.0 * length * scale, -3.0 * l2 * scale],
        [54.0 * scale, 13.0 * length * scale, 156.0 * scale, -22.0 * length * scale],
        [-13.0 * length * scale, -3.0 * l2 * scale, -22.0 * length * scale, 4.0 * l2 * scale],
    ]


def build_local_stiffness_matrix(properties: BeamElementProperties) -> Matrix:
    if properties.length <= 0.0:
        raise ValueError("Beam element length must be positive")

    matrix = zero_matrix12()
    axial = properties.youngs_modulus * properties.area / properties.length
    torsion = properties.shear_modulus * properties.torsion_constant / properties.length

    matrix[0][0] = axial
    matrix[0][6] = -axial
    matrix[6][0] = -axial
    matrix[6][6] = axial

    matrix[3][3] = torsion
    matrix[3][9] = -torsion
    matrix[9][3] = -torsion
    matrix[9][9] = torsion

    bending_z = _build_plane_bending_stiffness(properties.youngs_modulus * properties.iz, properties.length)
    _add_submatrix(matrix, [1, 5, 7, 11], bending_z)

    bending_y_base = _build_plane_bending_stiffness(properties.youngs_modulus * properties.iy, properties.length)
    sign = [1.0, -1.0, 1.0, -1.0]
    bending_y = [[sign[row] * bending_y_base[row][col] * sign[col] for col in range(4)] for row in range(4)]
    _add_submatrix(matrix, [2, 4, 8, 10], bending_y)
    return matrix


def build_local_mass_matrix(properties: BeamElementProperties) -> Matrix:
    if properties.length <= 0.0:
        raise ValueError("Beam element length must be positive")

    matrix = zero_matrix12()
    density_area = properties.density * properties.area
    axial_scale = density_area * properties.length / 6.0
    torsional_scale = properties.density * properties.torsion_constant * properties.length / 6.0

    matrix[0][0] = 2.0 * axial_scale
    matrix[0][6] = axial_scale
    matrix[6][0] = axial_scale
    matrix[6][6] = 2.0 * axial_scale

    matrix[3][3] = 2.0 * torsional_scale
    matrix[3][9] = torsional_scale
    matrix[9][3] = torsional_scale
    matrix[9][9] = 2.0 * torsional_scale

    bending_z = _build_plane_bending_mass(density_area, properties.length)
    _add_submatrix(matrix, [1, 5, 7, 11], bending_z)

    bending_y_base = _build_plane_bending_mass(density_area, properties.length)
    sign = [1.0, -1.0, 1.0, -1.0]
    bending_y = [[sign[row] * bending_y_base[row][col] * sign[col] for col in range(4)] for row in range(4)]
    _add_submatrix(matrix, [2, 4, 8, 10], bending_y)
    return matrix


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
    return _multiply(_transpose(transformation), _multiply(local_matrix, transformation))


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
