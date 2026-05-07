from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

from orchard_fem.discretization.damping import compute_default_damping_ratio
from orchard_fem.domain import JointLawKind, OrchardModel
from orchard_fem.dynamics.excitation import build_frequency_excitation_load
from orchard_fem.dynamics.frequency_response import (
    FrequencyResponsePoint,
    FrequencyResponseResult,
)
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.dofs import (
    EmbeddedBeamResponseMapping,
    resolve_embedded_beam_response_mapping,
)
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.operators import (
    EmbeddedBeamExperimentBundle,
    build_embedded_timoshenko_experiment,
)
from orchard_fem.materials.base import build_material_lookup
from orchard_fem.numerics import require_petsc


@dataclass(frozen=True)
class EmbeddedBeamFrequencyResponseExperimentResult:
    experiment: EmbeddedBeamExperimentBundle
    response_mapping: EmbeddedBeamResponseMapping
    damping_matrix: Any
    result: FrequencyResponseResult


def _frequency_grid(analysis) -> list[float]:
    steps = max(analysis.frequency_steps, 1)
    if steps == 1:
        return [analysis.frequency_start_hz]

    return [
        analysis.frequency_start_hz
        + (
            step_index / (steps - 1)
        ) * (analysis.frequency_end_hz - analysis.frequency_start_hz)
        for step_index in range(steps)
    ]


def _require_supported_frequency_response_model(model: OrchardModel) -> None:
    unsupported_joint_laws = [
        joint.joint_id
        for joint in model.joints
        if joint.law.kind != JointLawKind.NONE
    ]
    if unsupported_joint_laws:
        raise NotImplementedError(
            "The experimental FEniCSx frequency-response branch does not yet support nonlinear joint laws. "
            f"Unsupported joints: {', '.join(sorted(unsupported_joint_laws))}."
        )
    if model.analysis.auto_nonlinear_levels:
        raise NotImplementedError(
            "The experimental FEniCSx frequency-response branch does not yet support automatic nonlinear link injection."
        )
    nonlinear_clamps = [
        clamp.branch_id
        for clamp in model.clamps
        if abs(clamp.cubic_stiffness) > 1.0e-14
    ]
    if nonlinear_clamps:
        raise NotImplementedError(
            "The experimental FEniCSx frequency-response branch does not yet support cubic clamp nonlinearities. "
            f"Unsupported clamps: {', '.join(sorted(nonlinear_clamps))}."
        )


def _resolve_rayleigh_coefficients(model: OrchardModel) -> tuple[float, float]:
    alpha = float(model.analysis.rayleigh_alpha)
    beta = float(model.analysis.rayleigh_beta)

    if abs(alpha) < 1.0e-14 and abs(beta) < 1.0e-14:
        material_lookup = build_material_lookup(model.materials)
        zeta = compute_default_damping_ratio(model, material_lookup)
        omega_ref = 2.0 * pi * max(model.analysis.frequency_start_hz, 0.1)
        beta = (2.0 * zeta / omega_ref) if omega_ref > 0.0 else 0.0

    return alpha, beta


def build_embedded_rayleigh_damping_matrix(
    stiffness_matrix: Any,
    mass_matrix: Any,
    *,
    alpha: float,
    beta: float,
) -> Any:
    require_petsc()

    from petsc4py import PETSc

    damping_matrix = mass_matrix.duplicate(copy=True)
    damping_matrix.scale(float(alpha))
    damping_matrix.axpy(
        float(beta),
        stiffness_matrix,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    damping_matrix.assemble()
    return damping_matrix


def _add_matrix_in_place(target: Any, increment: Any) -> None:
    require_petsc()

    from petsc4py import PETSc

    target.axpy(
        1.0,
        increment,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    target.assemble()


def _build_real_block_dynamic_matrix(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    *,
    omega: float,
) -> Any:
    require_petsc()

    from petsc4py import PETSc

    real_matrix = stiffness_matrix.duplicate(copy=True)
    real_matrix.axpy(
        -float(omega * omega),
        mass_matrix,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    real_matrix.assemble()

    imag_matrix = damping_matrix.duplicate(copy=True)
    imag_matrix.scale(float(omega))
    imag_matrix.assemble()

    negative_imag_matrix = imag_matrix.duplicate(copy=True)
    negative_imag_matrix.scale(-1.0)
    negative_imag_matrix.assemble()

    block_matrix = PETSc.Mat().createNest(
        [[real_matrix, negative_imag_matrix], [imag_matrix, real_matrix]]
    )
    block_matrix.assemble()

    aij_matrix = block_matrix.convert("aij")
    aij_matrix.assemble()
    return aij_matrix


def _matrix_diagonal_entry(matrix: Any, dof: int) -> float:
    values = matrix.getValues([int(dof)], [int(dof)])
    return float(values[0][0])


def _build_frequency_rhs_vector(
    block_matrix: Any,
    *,
    excitation_dof: int,
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    excitation,
    omega: float,
) -> Any:
    diagonal_stiffness = _matrix_diagonal_entry(stiffness_matrix, excitation_dof)
    diagonal_mass = _matrix_diagonal_entry(mass_matrix, excitation_dof)
    diagonal_damping = _matrix_diagonal_entry(damping_matrix, excitation_dof)

    load_real, load_imag = build_frequency_excitation_load(
        [[diagonal_stiffness]],
        [[diagonal_mass]],
        [[diagonal_damping]],
        0,
        excitation,
        omega,
    )

    size = stiffness_matrix.getSize()[0]
    rhs_vector = block_matrix.createVecRight()
    rhs_vector.set(0.0)
    rhs_vector.setValue(int(excitation_dof), float(load_real))
    rhs_vector.setValue(int(size + excitation_dof), float(load_imag))
    rhs_vector.assemblyBegin()
    rhs_vector.assemblyEnd()
    return rhs_vector


def _solve_petsc_real_block_system(block_matrix: Any, rhs_vector: Any) -> Any:
    require_petsc()

    from petsc4py import PETSc

    solution = rhs_vector.duplicate()
    solver = PETSc.KSP().create(block_matrix.getComm())
    solver.setOperators(block_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()
    solver.solve(rhs_vector, solution)

    if solver.getConvergedReason() <= 0:
        raise RuntimeError(
            f"PETSc KSP failed to converge for experimental FEniCSx frequency response "
            f"(reason={solver.getConvergedReason()})."
        )

    return solution


def _vector_value(vector: Any, global_index: int) -> float:
    values = vector.getValues([int(global_index)])
    return float(values[0])


def _extract_frequency_response_point(
    solution: Any,
    *,
    frequency_hz: float,
    system_size: int,
    response_mapping: EmbeddedBeamResponseMapping,
) -> FrequencyResponsePoint:
    excitation_real = _vector_value(solution, response_mapping.excitation_dof)
    excitation_imag = _vector_value(solution, system_size + response_mapping.excitation_dof)
    excitation_magnitude = (excitation_real**2 + excitation_imag**2) ** 0.5

    observation_magnitudes: list[float] = []
    for dof in response_mapping.observation_dofs:
        real_value = _vector_value(solution, dof)
        imag_value = _vector_value(solution, system_size + dof)
        observation_magnitudes.append((real_value**2 + imag_value**2) ** 0.5)

    return FrequencyResponsePoint(
        frequency_hz=frequency_hz,
        excitation_response_magnitude=excitation_magnitude,
        observation_magnitudes=observation_magnitudes,
    )


def solve_embedded_beam_frequency_response_experiment(
    model: OrchardModel,
    *,
    polynomial_degree: int = 1,
    spec: EmbeddedLineMeshSpec | None = None,
    shear_correction: float = 1.0,
    comm: object | None = None,
    partitioner: object | None = None,
    max_facet_to_cell_links: int = 2,
    use_model_clamps: bool = True,
    clamp_tolerance: float = 1.0e-8,
    response_tolerance: float = 1.0e-8,
) -> EmbeddedBeamFrequencyResponseExperimentResult:
    require_dolfinx()
    require_petsc()
    _require_supported_frequency_response_model(model)

    experiment = build_embedded_timoshenko_experiment(
        model,
        polynomial_degree=polynomial_degree,
        spec=spec,
        shear_correction=shear_correction,
        comm=comm,
        partitioner=partitioner,
        max_facet_to_cell_links=max_facet_to_cell_links,
        use_model_clamps=use_model_clamps,
        clamp_tolerance=clamp_tolerance,
    )
    response_mapping = resolve_embedded_beam_response_mapping(
        model,
        experiment.space_bundle,
        fruit_dofs=experiment.fruit_dofs,
        atol=response_tolerance,
    )

    alpha, beta = _resolve_rayleigh_coefficients(model)
    damping_matrix = build_embedded_rayleigh_damping_matrix(
        experiment.operator_bundle.stiffness_matrix,
        experiment.operator_bundle.mass_matrix,
        alpha=alpha,
        beta=beta,
    )
    if experiment.operator_bundle.attachment_damping_matrix is not None:
        _add_matrix_in_place(
            damping_matrix,
            experiment.operator_bundle.attachment_damping_matrix,
        )

    system_size = experiment.operator_bundle.stiffness_matrix.getSize()[0]
    points: list[FrequencyResponsePoint] = []
    for frequency_hz in _frequency_grid(model.analysis):
        omega = 2.0 * pi * frequency_hz
        block_matrix = _build_real_block_dynamic_matrix(
            experiment.operator_bundle.stiffness_matrix,
            experiment.operator_bundle.mass_matrix,
            damping_matrix,
            omega=omega,
        )
        rhs_vector = _build_frequency_rhs_vector(
            block_matrix,
            excitation_dof=response_mapping.excitation_dof,
            stiffness_matrix=experiment.operator_bundle.stiffness_matrix,
            mass_matrix=experiment.operator_bundle.mass_matrix,
            damping_matrix=damping_matrix,
            excitation=model.excitation,
            omega=omega,
        )
        solution = _solve_petsc_real_block_system(block_matrix, rhs_vector)
        points.append(
            _extract_frequency_response_point(
                solution,
                frequency_hz=frequency_hz,
                system_size=system_size,
                response_mapping=response_mapping,
            )
        )

    return EmbeddedBeamFrequencyResponseExperimentResult(
        experiment=experiment,
        response_mapping=response_mapping,
        damping_matrix=damping_matrix,
        result=FrequencyResponseResult(
            observation_names=response_mapping.observation_names,
            points=points,
        ),
    )
