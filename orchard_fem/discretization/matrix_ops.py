from __future__ import annotations

from orchard_fem.discretization.types import Matrix


def zero_matrix(size: int) -> Matrix:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def scatter(global_matrix: Matrix, local_matrix: Matrix, dofs: list[int]) -> None:
    for row in range(len(dofs)):
        for col in range(len(dofs)):
            global_matrix[dofs[row]][dofs[col]] += local_matrix[row][col]


def add_pair_penalty(matrix: Matrix, first_dof: int, second_dof: int, penalty: float) -> None:
    matrix[first_dof][first_dof] += penalty
    matrix[first_dof][second_dof] -= penalty
    matrix[second_dof][first_dof] -= penalty
    matrix[second_dof][second_dof] += penalty
