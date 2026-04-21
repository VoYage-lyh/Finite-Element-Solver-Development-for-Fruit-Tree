from __future__ import annotations

from typing import TYPE_CHECKING

from orchard_fem.solvers._petsc import create_aij_matrix, require_petsc, solve_linear_system


if TYPE_CHECKING:
    from orchard_fem.solvers.modal_assembler import LinearDynamicAssemblyResult


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(row[column] * vector[column] for column in range(len(vector)))
        for row in matrix
    ]


def _local_displacement(
    transformation_matrix: list[list[float]],
    global_displacement: list[float],
    dofs: tuple[int, ...],
) -> list[float]:
    element_global = [global_displacement[dof] for dof in dofs]
    return [
        sum(transformation_matrix[row][column] * element_global[column] for column in range(len(dofs)))
        for row in range(len(dofs))
    ]


def compute_gravity_axial_forces(
    assembled: "LinearDynamicAssemblyResult",
    gravity_load: list[float],
) -> dict[str, list[float]]:
    """
    Solve the static gravity preload and recover element axial forces.
    """
    require_petsc()

    if len(gravity_load) != len(assembled.dof_labels):
        raise ValueError("gravity_load size must match the assembled DOF count")

    stiffness = create_aij_matrix(assembled.stiffness_matrix)
    static_displacement = solve_linear_system(stiffness, gravity_load)
    residual = _mat_vec(assembled.stiffness_matrix, static_displacement)
    residual_norm = max(
        (abs(expected - actual) for expected, actual in zip(gravity_load, residual)),
        default=0.0,
    )
    if residual_norm > 1.0e-5:
        raise RuntimeError(f"Gravity preload solve residual is too large: {residual_norm}")

    axial_forces: dict[str, list[float]] = {}
    for branch_id, elements in assembled.branch_elements.items():
        branch_forces: list[float] = []
        for element in elements:
            local_displacement = _local_displacement(
                element.transformation_matrix,
                static_displacement,
                element.dofs,
            )
            axial_force = element.axial_rigidity * (
                (local_displacement[6] - local_displacement[0]) / element.length
            )
            branch_forces.append(axial_force)
        axial_forces[branch_id] = branch_forces

    return axial_forces
