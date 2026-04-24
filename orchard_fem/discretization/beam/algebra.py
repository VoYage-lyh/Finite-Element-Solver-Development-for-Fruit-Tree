from __future__ import annotations

from orchard_fem.discretization.beam.types import Matrix


def zero_matrix12() -> Matrix:
    return [[0.0 for _ in range(12)] for _ in range(12)]


def transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix)]


def multiply(left: Matrix, right: Matrix) -> Matrix:
    rows = len(left)
    inner = len(left[0])
    cols = len(right[0])
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            result[row][col] = sum(
                left[row][index] * right[index][col] for index in range(inner)
            )
    return result
