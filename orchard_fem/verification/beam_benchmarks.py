from __future__ import annotations

from dataclasses import dataclass

from orchard_fem.elements.beam_formulation import (
    BeamElementProperties,
    build_local_mass_matrix,
    build_local_stiffness_matrix,
)
from orchard_fem.solvers._petsc import create_aij_matrix, require_petsc, require_slepc, solve_linear_system
from orchard_fem.solvers.modal import ModalAnalysisRequest, SLEPcModalSolver


@dataclass(frozen=True)
class PlanarBeamSystem:
    stiffness: list[list[float]]
    mass: list[list[float]]


def _zeros(size: int) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _reduced_matrix(matrix: list[list[float]], free_dofs: list[int]) -> list[list[float]]:
    return [
        [matrix[row][col] for col in free_dofs]
        for row in free_dofs
    ]


def _reduced_vector(vector: list[float], free_dofs: list[int]) -> list[float]:
    return [vector[dof] for dof in free_dofs]


def _complement_dofs(total_dofs: int, fixed_dofs: list[int]) -> list[int]:
    fixed = set(fixed_dofs)
    return [dof for dof in range(total_dofs) if dof not in fixed]


def _extract_planar_block(matrix: list[list[float]]) -> list[list[float]]:
    indices = [1, 5, 7, 11]
    return [
        [matrix[row][col] for col in indices]
        for row in indices
    ]


def build_uniform_planar_beam(
    num_elements: int,
    length: float,
    youngs_modulus: float,
    density: float,
    area: float,
    inertia: float,
) -> PlanarBeamSystem:
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    nodes = num_elements + 1
    dofs = 2 * nodes
    system = PlanarBeamSystem(stiffness=_zeros(dofs), mass=_zeros(dofs))

    properties = BeamElementProperties(
        youngs_modulus=youngs_modulus,
        shear_modulus=youngs_modulus / (2.0 * (1.0 + 0.3)),
        area=area,
        iy=inertia,
        iz=inertia,
        torsion_constant=2.0 * inertia,
        density=density,
        length=length / float(num_elements),
    )

    local_stiffness = _extract_planar_block(build_local_stiffness_matrix(properties))
    local_mass = _extract_planar_block(build_local_mass_matrix(properties))

    for element_index in range(num_elements):
        element_dofs = [
            2 * element_index,
            (2 * element_index) + 1,
            2 * (element_index + 1),
            (2 * (element_index + 1)) + 1,
        ]
        for local_row, global_row in enumerate(element_dofs):
            for local_col, global_col in enumerate(element_dofs):
                system.stiffness[global_row][global_col] += local_stiffness[local_row][local_col]
                system.mass[global_row][global_col] += local_mass[local_row][local_col]

    return system


def build_hinged_two_bar_system(
    first_length: float,
    second_length: float,
    youngs_modulus: float,
    density: float,
    area: float,
    inertia: float,
    rotational_stiffness: float,
    tip_mass: float,
) -> PlanarBeamSystem:
    system = PlanarBeamSystem(stiffness=_zeros(7), mass=_zeros(7))

    first = build_uniform_planar_beam(
        num_elements=1,
        length=first_length,
        youngs_modulus=youngs_modulus,
        density=density,
        area=area,
        inertia=inertia,
    )
    second = build_uniform_planar_beam(
        num_elements=1,
        length=second_length,
        youngs_modulus=youngs_modulus,
        density=density,
        area=area,
        inertia=inertia,
    )

    first_map = [0, 1, 2, 3]
    second_map = [2, 4, 5, 6]
    for local_row, first_row in enumerate(first_map):
        for local_col, first_col in enumerate(first_map):
            system.stiffness[first_row][first_col] += first.stiffness[local_row][local_col]
            system.mass[first_row][first_col] += first.mass[local_row][local_col]
        for local_col, second_col in enumerate(second_map):
            system.stiffness[second_map[local_row]][second_col] += second.stiffness[local_row][local_col]
            system.mass[second_map[local_row]][second_col] += second.mass[local_row][local_col]

    system.stiffness[3][3] += rotational_stiffness
    system.stiffness[3][4] -= rotational_stiffness
    system.stiffness[4][3] -= rotational_stiffness
    system.stiffness[4][4] += rotational_stiffness
    system.mass[5][5] += tip_mass
    return system


def solve_static_system(
    stiffness: list[list[float]],
    force: list[float],
    fixed_dofs: list[int],
) -> list[float]:
    require_petsc()

    if len(stiffness) != len(force):
        raise ValueError("force length must match stiffness size")

    free_dofs = _complement_dofs(len(force), fixed_dofs)
    reduced_stiffness = _reduced_matrix(stiffness, free_dofs)
    reduced_force = _reduced_vector(force, free_dofs)
    reduced_solution = solve_linear_system(create_aij_matrix(reduced_stiffness), reduced_force)

    full_solution = [0.0 for _ in force]
    for index, dof in enumerate(free_dofs):
        full_solution[dof] = reduced_solution[index]
    return full_solution


def solve_generalized_frequencies(
    stiffness: list[list[float]],
    mass: list[list[float]],
    fixed_dofs: list[int],
    count: int,
) -> list[float]:
    require_slepc()

    if len(stiffness) != len(mass):
        raise ValueError("stiffness and mass must have the same size")
    if count <= 0:
        raise ValueError("count must be positive")

    free_dofs = _complement_dofs(len(stiffness), fixed_dofs)
    reduced_stiffness = _reduced_matrix(stiffness, free_dofs)
    reduced_mass = _reduced_matrix(mass, free_dofs)
    modes = SLEPcModalSolver().solve(
        ModalAnalysisRequest(
            num_modes=count,
            stiffness_matrix=reduced_stiffness,
            mass_matrix=reduced_mass,
            dof_labels=[f"dof_{dof}" for dof in free_dofs],
        )
    )
    return [mode.frequency_hz for mode in modes]
