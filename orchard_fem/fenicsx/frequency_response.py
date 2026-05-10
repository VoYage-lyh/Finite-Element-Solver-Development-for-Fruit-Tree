from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

from orchard_fem.discretization.damping import compute_default_damping_ratio
from orchard_fem.discretization.types import NonlinearLinkDefinition
from orchard_fem.dynamics.continuation import solve_frequency_continuation
from orchard_fem.domain import OrchardModel
from orchard_fem.dynamics.excitation import build_frequency_excitation_load
from orchard_fem.dynamics.frequency_response import (
    FrequencyResponsePoint,
    FrequencyResponseResult,
)
from orchard_fem.dynamics.harmonic_balance import first_harmonic_link_response
from orchard_fem.fenicsx.assembly import assemble_fenicsx_system
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.dofs import (
    EmbeddedBeamResponseMapping,
    resolve_embedded_beam_response_mapping,
)
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.operator_bundle import (
    EmbeddedBeamExperimentBundle,
)
from orchard_fem.fenicsx.mpc import (
    apply_mpc_to_real_block_vector_in_place,
    backsubstitute_mpc_real_block_vector,
)
from orchard_fem.fenicsx.petsc_ops import (
    accumulate_owned_matrix_value as _accumulate_owned_matrix_value,
    accumulate_owned_vector_value as _accumulate_owned_vector_value,
    add_matrix_in_place as _add_matrix_in_place,
    create_empty_aij_matrix_like as _create_empty_aij_matrix_like,
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
    del model


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


def _relative_complex_components(
    solution: Any,
    system_size: int,
    link: NonlinearLinkDefinition,
) -> tuple[float, float]:
    first_real = _vector_value(solution, link.first_dof)
    first_imag = _vector_value(solution, system_size + link.first_dof)
    if link.second_dof >= 0:
        second_real = _vector_value(solution, link.second_dof)
        second_imag = _vector_value(solution, system_size + link.second_dof)
    else:
        second_real = 0.0
        second_imag = 0.0

    real_delta = first_real - second_real
    imag_delta = first_imag - second_imag
    return real_delta, imag_delta


def _build_first_harmonic_nonlinear_vector_and_tangent(
    block_matrix: Any,
    nonlinear_links: list[NonlinearLinkDefinition],
    solution: Any,
    *,
    build_tangent: bool,
) -> tuple[Any, Any | None]:
    vector = block_matrix.createVecRight()
    vector.set(0.0)
    matrix = _create_empty_aij_matrix_like(block_matrix) if build_tangent else None
    vector_owned_rows = vector.getOwnershipRange()
    matrix_owned_rows = matrix.getOwnershipRange() if matrix is not None else (0, 0)
    system_size = block_matrix.getSize()[0] // 2

    for link in nonlinear_links:
        relative_real, relative_imag = _relative_complex_components(
            solution,
            system_size,
            link,
        )
        link_response = first_harmonic_link_response(
            link,
            relative_real,
            relative_imag,
        )
        signed_dofs = [(link.first_dof, 1.0)]
        if link.second_dof >= 0:
            signed_dofs.append((link.second_dof, -1.0))

        for row_dof, row_sign in signed_dofs:
            _accumulate_owned_vector_value(
                vector,
                vector_owned_rows,
                row_dof,
                row_sign * link_response.force_real,
            )
            _accumulate_owned_vector_value(
                vector,
                vector_owned_rows,
                system_size + row_dof,
                row_sign * link_response.force_imag,
            )
            if matrix is None:
                continue

            for column_dof, column_sign in signed_dofs:
                sign = row_sign * column_sign
                _accumulate_owned_matrix_value(
                    matrix,
                    matrix_owned_rows,
                    row_dof,
                    column_dof,
                    sign * link_response.tangent_rr,
                )
                _accumulate_owned_matrix_value(
                    matrix,
                    matrix_owned_rows,
                    row_dof,
                    system_size + column_dof,
                    sign * link_response.tangent_ri,
                )
                _accumulate_owned_matrix_value(
                    matrix,
                    matrix_owned_rows,
                    system_size + row_dof,
                    column_dof,
                    sign * link_response.tangent_ir,
                )
                _accumulate_owned_matrix_value(
                    matrix,
                    matrix_owned_rows,
                    system_size + row_dof,
                    system_size + column_dof,
                    sign * link_response.tangent_ii,
                )

    vector.assemblyBegin()
    vector.assemblyEnd()
    if matrix is not None:
        matrix.assemble()
    return vector, matrix


def _solution_relative_change(current_solution: Any, previous_solution: Any | None) -> float:
    if previous_solution is None:
        return float("inf")
    difference = current_solution.duplicate()
    difference.set(0.0)
    difference.axpy(1.0, current_solution)
    difference.axpy(-1.0, previous_solution)
    return difference.norm() / max(current_solution.norm(), 1.0)


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


def _solve_harmonic_balance_frequency_point(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    nonlinear_links: list[NonlinearLinkDefinition],
    *,
    excitation_dof: int,
    excitation,
    analysis,
    omega: float,
    initial_solution: Any | None,
    mpc: Any | None = None,
) -> Any:
    require_petsc()

    from petsc4py import PETSc

    max_iterations = max(int(analysis.max_nonlinear_iterations), 1)
    tolerance = float(analysis.nonlinear_tolerance)
    block_matrix = _build_real_block_dynamic_matrix(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        omega=omega,
    )
    rhs_vector = _build_frequency_rhs_vector(
        block_matrix,
        excitation_dof=excitation_dof,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=mass_matrix,
        damping_matrix=damping_matrix,
        excitation=excitation,
        omega=omega,
    )
    apply_mpc_to_real_block_vector_in_place(
        mpc,
        rhs_vector,
        system_size=stiffness_matrix.getSize()[0],
        template_matrix=stiffness_matrix,
    )
    linear_solution = _solve_petsc_real_block_system(block_matrix, rhs_vector)
    linear_solution = backsubstitute_mpc_real_block_vector(
        mpc,
        linear_solution,
        system_size=stiffness_matrix.getSize()[0],
        template_matrix=stiffness_matrix,
    )

    def build_residual(current_solution: Any) -> Any:
        residual = rhs_vector.duplicate()
        block_matrix.mult(current_solution, residual)
        residual.axpy(-1.0, rhs_vector)
        nonlinear_vector, _ = _build_first_harmonic_nonlinear_vector_and_tangent(
            block_matrix,
            nonlinear_links,
            current_solution,
            build_tangent=False,
        )
        residual.axpy(1.0, nonlinear_vector)
        return residual

    def build_jacobian(current_solution: Any) -> Any:
        jacobian = block_matrix.duplicate(copy=True)
        jacobian.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        jacobian.zeroEntries()
        jacobian.axpy(
            1.0,
            block_matrix,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )
        _, nonlinear_tangent = _build_first_harmonic_nonlinear_vector_and_tangent(
            block_matrix,
            nonlinear_links,
            current_solution,
            build_tangent=True,
        )
        if nonlinear_tangent is not None:
            jacobian.axpy(
                1.0,
                nonlinear_tangent,
                structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
            )
        jacobian.assemble()
        return jacobian

    def solve_correction(jacobian: Any, residual: Any) -> Any:
        correction_rhs = residual.duplicate()
        residual.copy(correction_rhs)
        correction_rhs.scale(-1.0)
        correction = correction_rhs.duplicate()
        solver = PETSc.KSP().create(block_matrix.getComm())
        solver.setOperators(jacobian)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        solver.setFromOptions()
        solver.solve(correction_rhs, correction)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError(
                "PETSc KSP failed inside FEniCSx harmonic-balance Newton "
                f"(reason={solver.getConvergedReason()})."
            )
        return correction

    def solve_from_start(start_solution: Any) -> Any:
        solution = start_solution.duplicate()
        start_solution.copy(solution)
        target_norm = tolerance * max(
            rhs_vector.norm(norm_type=PETSc.NormType.NORM_INFINITY),
            1.0,
        )

        for _ in range(max_iterations):
            residual = build_residual(solution)
            residual_norm = residual.norm(norm_type=PETSc.NormType.NORM_INFINITY)
            if residual_norm <= target_norm:
                return solution

            correction = solve_correction(build_jacobian(solution), residual)
            if correction.norm(norm_type=PETSc.NormType.NORM_INFINITY) <= (
                tolerance
                * max(solution.norm(norm_type=PETSc.NormType.NORM_INFINITY), 1.0)
            ):
                return solution

            accepted_solution = None
            accepted_norm = residual_norm
            step_scale = 1.0
            for _line_search_index in range(10):
                trial = solution.duplicate()
                solution.copy(trial)
                trial.axpy(step_scale, correction)
                trial_residual = build_residual(trial)
                trial_norm = trial_residual.norm(norm_type=PETSc.NormType.NORM_INFINITY)
                if trial_norm < accepted_norm:
                    accepted_solution = trial
                    accepted_norm = trial_norm
                    break
                step_scale *= 0.5

            if accepted_solution is None:
                solution.axpy(1.0, correction)
            else:
                solution = accepted_solution

        final_residual = build_residual(solution)
        raise RuntimeError(
            "FEniCSx harmonic-balance frequency response failed to converge "
            f"(residual={final_residual.norm(norm_type=PETSc.NormType.NORM_INFINITY):.6e}, "
            f"target={target_norm:.6e})."
        )

    def scaled_solution(base_solution: Any, scale: float) -> Any:
        scaled = base_solution.duplicate()
        base_solution.copy(scaled)
        scaled.scale(float(scale))
        return scaled

    def append_unique_start(starts: list[Any], candidate: Any) -> None:
        for existing in starts:
            difference = existing.duplicate()
            existing.copy(difference)
            difference.axpy(-1.0, candidate)
            if difference.norm() <= 1.0e-14:
                return
        starts.append(candidate)

    starts: list[Any] = []
    if initial_solution is not None:
        append_unique_start(starts, initial_solution)
    for candidate in (
        linear_solution,
        scaled_solution(linear_solution, 0.25),
        scaled_solution(linear_solution, 0.5),
        scaled_solution(linear_solution, 2.0),
    ):
        append_unique_start(starts, candidate)

    errors: list[str] = []
    for start_solution in starts:
        try:
            return solve_from_start(start_solution)
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError("; ".join(errors))


def solve_embedded_beam_frequency_response_experiment(
    model: OrchardModel,
    *,
    polynomial_degree: int = 1,
    spec: EmbeddedLineMeshSpec | None = None,
    shear_correction: float = 0.4,
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

    assembly = assemble_fenicsx_system(
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
    experiment = assembly.experiment
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
    nonlinear_links = experiment.operator_bundle.nonlinear_links
    points: list[FrequencyResponsePoint] = []
    target_frequencies = _frequency_grid(model.analysis)
    if nonlinear_links:
        continuation_points = solve_frequency_continuation(
            target_frequencies,
            lambda frequency_hz, initial_solution: _solve_harmonic_balance_frequency_point(
                experiment.operator_bundle.stiffness_matrix,
                experiment.operator_bundle.mass_matrix,
                damping_matrix,
                nonlinear_links,
                excitation_dof=response_mapping.excitation_dof,
                excitation=model.excitation,
                analysis=model.analysis,
                omega=2.0 * pi * frequency_hz,
                initial_solution=initial_solution,
                mpc=experiment.operator_bundle.mpc,
            ),
            _solution_relative_change,
        )
        points = [
            _extract_frequency_response_point(
                backsubstitute_mpc_real_block_vector(
                    experiment.operator_bundle.mpc,
                    point.state,
                    system_size=system_size,
                    template_matrix=experiment.operator_bundle.stiffness_matrix,
                ),
                frequency_hz=point.frequency_hz,
                system_size=system_size,
                response_mapping=response_mapping,
            )
            for point in continuation_points
        ]
    else:
        for frequency_hz in target_frequencies:
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
            apply_mpc_to_real_block_vector_in_place(
                experiment.operator_bundle.mpc,
                rhs_vector,
                system_size=system_size,
                template_matrix=experiment.operator_bundle.stiffness_matrix,
            )
            solution = _solve_petsc_real_block_system(block_matrix, rhs_vector)
            solution = backsubstitute_mpc_real_block_vector(
                experiment.operator_bundle.mpc,
                solution,
                system_size=system_size,
                template_matrix=experiment.operator_bundle.stiffness_matrix,
            )
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
