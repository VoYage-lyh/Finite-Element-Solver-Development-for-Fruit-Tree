from __future__ import annotations

from orchard_fem.discretization.beam.algebra import zero_matrix12
from orchard_fem.discretization.beam.types import BeamElementProperties, Matrix


def _add_submatrix(target: Matrix, indices: list[int], source: list[list[float]]) -> None:
    for row in range(4):
        for col in range(4):
            target[indices[row]][indices[col]] += source[row][col]


def _build_plane_bending_stiffness(
    flexural_rigidity: float,
    length: float,
) -> list[list[float]]:
    l2 = length * length
    l3 = l2 * length
    scale = flexural_rigidity / l3
    return [
        [12.0 * scale, 6.0 * length * scale, -12.0 * scale, 6.0 * length * scale],
        [6.0 * length * scale, 4.0 * l2 * scale, -6.0 * length * scale, 2.0 * l2 * scale],
        [-12.0 * scale, -6.0 * length * scale, 12.0 * scale, -6.0 * length * scale],
        [6.0 * length * scale, 2.0 * l2 * scale, -6.0 * length * scale, 4.0 * l2 * scale],
    ]


def _axial_rigidity(properties: BeamElementProperties) -> float:
    if properties.axial_rigidity is not None:
        return properties.axial_rigidity
    return properties.youngs_modulus * properties.area


def _torsional_rigidity(properties: BeamElementProperties) -> float:
    if properties.torsional_rigidity is not None:
        return properties.torsional_rigidity
    return properties.shear_modulus * properties.torsion_constant


def _bending_rigidity_y(properties: BeamElementProperties) -> float:
    if properties.bending_rigidity_y is not None:
        return properties.bending_rigidity_y
    return properties.youngs_modulus * properties.iy


def _bending_rigidity_z(properties: BeamElementProperties) -> float:
    if properties.bending_rigidity_z is not None:
        return properties.bending_rigidity_z
    return properties.youngs_modulus * properties.iz


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
    axial = _axial_rigidity(properties) / properties.length
    torsion = _torsional_rigidity(properties) / properties.length

    matrix[0][0] = axial
    matrix[0][6] = -axial
    matrix[6][0] = -axial
    matrix[6][6] = axial

    matrix[3][3] = torsion
    matrix[3][9] = -torsion
    matrix[9][3] = -torsion
    matrix[9][9] = torsion

    bending_z = _build_plane_bending_stiffness(
        _bending_rigidity_z(properties),
        properties.length,
    )
    _add_submatrix(matrix, [1, 5, 7, 11], bending_z)

    bending_y_base = _build_plane_bending_stiffness(
        _bending_rigidity_y(properties),
        properties.length,
    )
    sign = [1.0, -1.0, 1.0, -1.0]
    bending_y = [
        [sign[row] * bending_y_base[row][col] * sign[col] for col in range(4)]
        for row in range(4)
    ]
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
    bending_y = [
        [sign[row] * bending_y_base[row][col] * sign[col] for col in range(4)]
        for row in range(4)
    ]
    _add_submatrix(matrix, [2, 4, 8, 10], bending_y)
    return matrix


def build_local_geometric_stiffness_matrix(axial_force: float, length: float) -> Matrix:
    if length <= 0.0:
        raise ValueError("Beam element length must be positive")

    matrix = zero_matrix12()
    scale = axial_force / (30.0 * length)
    base = [
        [36.0 * scale, 3.0 * length * scale, -36.0 * scale, 3.0 * length * scale],
        [
            3.0 * length * scale,
            4.0 * length * length * scale,
            -3.0 * length * scale,
            -1.0 * length * length * scale,
        ],
        [-36.0 * scale, -3.0 * length * scale, 36.0 * scale, -3.0 * length * scale],
        [
            3.0 * length * scale,
            -1.0 * length * length * scale,
            -3.0 * length * scale,
            4.0 * length * length * scale,
        ],
    ]
    _add_submatrix(matrix, [1, 5, 7, 11], base)

    sign = [1.0, -1.0, 1.0, -1.0]
    bending_y = [
        [sign[row] * base[row][col] * sign[col] for col in range(4)]
        for row in range(4)
    ]
    _add_submatrix(matrix, [2, 4, 8, 10], bending_y)
    return matrix
