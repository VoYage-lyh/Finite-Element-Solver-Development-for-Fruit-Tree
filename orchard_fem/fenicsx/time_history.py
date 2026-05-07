from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

from orchard_fem.discretization.damping import compute_default_damping_ratio
from orchard_fem.dynamics.nonlinear import nonlinear_force, nonlinear_tangent
from orchard_fem.domain import ExcitationKind, OrchardModel
from orchard_fem.dynamics.excitation import (
    TimeExcitationState,
    default_driving_frequency_hz,
)
from orchard_fem.dynamics.time_history import TimeHistoryPoint, TimeHistoryResult
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.dofs import (
    EmbeddedBeamResponseMapping,
    resolve_embedded_beam_response_mapping,
)
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.frequency_response import (
    build_embedded_rayleigh_damping_matrix,
)
from orchard_fem.fenicsx.operators import (
    EmbeddedBeamExperimentBundle,
    build_embedded_timoshenko_experiment,
)
from orchard_fem.materials.base import build_material_lookup
from orchard_fem.numerics import require_petsc


@dataclass(frozen=True)
class EmbeddedBeamTimeHistoryExperimentResult:
    experiment: EmbeddedBeamExperimentBundle
    response_mapping: EmbeddedBeamResponseMapping
    damping_matrix: Any
    result: TimeHistoryResult


def _require_supported_time_history_model(model: OrchardModel) -> None:
    del model


def _resolve_rayleigh_coefficients(model: OrchardModel) -> tuple[float, float]:
    alpha = float(model.analysis.rayleigh_alpha)
    beta = float(model.analysis.rayleigh_beta)

    if abs(alpha) < 1.0e-14 and abs(beta) < 1.0e-14:
        material_lookup = build_material_lookup(model.materials)
        zeta = compute_default_damping_ratio(model, material_lookup)
        omega_ref = 2.0 * pi * max(default_driving_frequency_hz(model.excitation, model.analysis), 0.1)
        beta = (2.0 * zeta / omega_ref) if omega_ref > 0.0 else 0.0

    return alpha, beta


def _add_matrix_in_place(target: Any, increment: Any) -> Any:
    require_petsc()

    from petsc4py import PETSc

    updated = target.duplicate(copy=True)
    updated.axpy(
        1.0,
        increment,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    updated.assemble()
    return updated


def _build_direct_solver(matrix: Any) -> Any:
    require_petsc()

    from petsc4py import PETSc

    solver = PETSc.KSP().create(matrix.getComm())
    solver.setOperators(matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()
    return solver


def _copy_vector(vector: Any) -> Any:
    copied = vector.duplicate()
    copied.set(0.0)
    copied.axpy(1.0, vector)
    return copied


def _vector_value(vector: Any, global_index: int) -> float:
    values = vector.getValues([int(global_index)])
    return float(values[0])


def _solve_direct_system(solver: Any, rhs_vector: Any) -> Any:
    solution = rhs_vector.duplicate()
    solver.solve(rhs_vector, solution)
    if solver.getConvergedReason() <= 0:
        raise RuntimeError(
            "PETSc KSP failed to converge for experimental FEniCSx time history "
            f"(reason={solver.getConvergedReason()})."
        )
    return solution


def _create_empty_aij_matrix_like(matrix: Any) -> Any:
    require_petsc()

    from petsc4py import PETSc

    created = PETSc.Mat().createAIJ(size=matrix.getSize(), comm=matrix.getComm())
    created.setUp()
    return created


def _accumulate_owned_matrix_value(
    matrix: Any,
    owned_rows: tuple[int, int],
    row: int,
    column: int,
    value: float,
) -> None:
    ownership_start, ownership_end = owned_rows
    if not (ownership_start <= row < ownership_end):
        return

    from petsc4py import PETSc

    matrix.setValue(
        row,
        column,
        float(value),
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def _accumulate_owned_vector_value(
    vector: Any,
    owned_rows: tuple[int, int],
    index: int,
    value: float,
) -> None:
    ownership_start, ownership_end = owned_rows
    if not (ownership_start <= index < ownership_end):
        return

    from petsc4py import PETSc

    vector.setValue(
        index,
        float(value),
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def _evaluate_nonlinear_force_and_tangent(
    stiffness_matrix: Any,
    nonlinear_links,
    displacement: Any,
) -> tuple[Any, Any]:
    force_vector = stiffness_matrix.createVecRight()
    force_vector.set(0.0)
    tangent_matrix = _create_empty_aij_matrix_like(stiffness_matrix)
    owned_rows = tangent_matrix.getOwnershipRange()
    owned_vector_rows = force_vector.getOwnershipRange()

    for link in nonlinear_links:
        second_value = _vector_value(displacement, link.second_dof) if link.second_dof >= 0 else 0.0
        relative_displacement = _vector_value(displacement, link.first_dof) - second_value
        scalar_force = nonlinear_force(link, relative_displacement)
        scalar_tangent = nonlinear_tangent(link, relative_displacement)

        _accumulate_owned_vector_value(
            force_vector,
            owned_vector_rows,
            link.first_dof,
            scalar_force,
        )
        _accumulate_owned_matrix_value(
            tangent_matrix,
            owned_rows,
            link.first_dof,
            link.first_dof,
            scalar_tangent,
        )

        if link.second_dof >= 0:
            _accumulate_owned_vector_value(
                force_vector,
                owned_vector_rows,
                link.second_dof,
                -scalar_force,
            )
            _accumulate_owned_matrix_value(
                tangent_matrix,
                owned_rows,
                link.first_dof,
                link.second_dof,
                -scalar_tangent,
            )
            _accumulate_owned_matrix_value(
                tangent_matrix,
                owned_rows,
                link.second_dof,
                link.first_dof,
                -scalar_tangent,
            )
            _accumulate_owned_matrix_value(
                tangent_matrix,
                owned_rows,
                link.second_dof,
                link.second_dof,
                scalar_tangent,
            )

    force_vector.assemblyBegin()
    force_vector.assemblyEnd()
    tangent_matrix.assemble()
    return force_vector, tangent_matrix


def _build_effective_matrix(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    nonlinear_tangent_matrix: Any | None,
    *,
    mass_scale: float,
    damping_scale: float,
) -> Any:
    from petsc4py import PETSc

    effective_matrix = stiffness_matrix.duplicate(copy=True)
    effective_matrix.axpy(
        mass_scale,
        mass_matrix,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    effective_matrix.axpy(
        damping_scale,
        damping_matrix,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    if nonlinear_tangent_matrix is not None:
        effective_matrix.axpy(
            1.0,
            nonlinear_tangent_matrix,
            structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
        )
    effective_matrix.assemble()
    return effective_matrix


def _build_residual_vector(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    displacement: Any,
    velocity: Any,
    acceleration: Any,
    nonlinear_force_vector: Any,
    external_load_vector: Any,
) -> Any:
    residual = external_load_vector.duplicate()
    mass_matrix.mult(acceleration, residual)

    damping_term = external_load_vector.duplicate()
    damping_matrix.mult(velocity, damping_term)
    residual.axpy(1.0, damping_term)

    stiffness_term = external_load_vector.duplicate()
    stiffness_matrix.mult(displacement, stiffness_term)
    residual.axpy(1.0, stiffness_term)
    residual.axpy(1.0, nonlinear_force_vector)
    residual.axpy(-1.0, external_load_vector)
    return residual


def _assign_vector(target: Any, source: Any) -> None:
    target.set(0.0)
    target.axpy(1.0, source)


def _assign_matrix(target: Any, source: Any) -> None:
    from petsc4py import PETSc

    target.zeroEntries()
    target.axpy(
        1.0,
        source,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    target.assemble()


def _state_from_displacement(
    displacement: Any,
    displacement_predictor: Any,
    velocity_predictor: Any,
    *,
    mass_scale: float,
    gamma: float,
    dt: float,
) -> tuple[Any, Any]:
    acceleration = _copy_vector(displacement)
    acceleration.axpy(-1.0, displacement_predictor)
    acceleration.scale(mass_scale)

    velocity = _copy_vector(velocity_predictor)
    velocity_increment = _copy_vector(acceleration)
    velocity_increment.scale(gamma * dt)
    velocity.axpy(1.0, velocity_increment)
    return acceleration, velocity


def _solve_nonlinear_newmark_step_with_snes(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    nonlinear_links,
    displacement_predictor: Any,
    velocity_predictor: Any,
    *,
    excitation_dof: int,
    excitation,
    analysis,
    time_seconds: float,
    mass_scale: float,
    damping_scale: float,
    gamma: float,
    dt: float,
) -> tuple[Any, Any, Any, TimeExcitationState]:
    require_petsc()

    from petsc4py import PETSc

    load_vector, excitation_state = _build_time_load_vector(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        excitation_dof=excitation_dof,
        excitation=excitation,
        analysis=analysis,
        time_seconds=time_seconds,
    )
    residual_vector = load_vector.duplicate()
    jacobian_matrix = _build_effective_matrix(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        None,
        mass_scale=mass_scale,
        damping_scale=damping_scale,
    )

    def assemble_residual(_snes, current_displacement, residual) -> None:
        acceleration, velocity = _state_from_displacement(
            current_displacement,
            displacement_predictor,
            velocity_predictor,
            mass_scale=mass_scale,
            gamma=gamma,
            dt=dt,
        )
        nonlinear_force_vector, _ = _evaluate_nonlinear_force_and_tangent(
            stiffness_matrix,
            nonlinear_links,
            current_displacement,
        )
        computed_residual = _build_residual_vector(
            stiffness_matrix,
            mass_matrix,
            damping_matrix,
            current_displacement,
            velocity,
            acceleration,
            nonlinear_force_vector,
            load_vector,
        )
        _assign_vector(residual, computed_residual)

    def assemble_jacobian(_snes, current_displacement, jacobian, preconditioner) -> None:
        _, nonlinear_tangent_matrix = _evaluate_nonlinear_force_and_tangent(
            stiffness_matrix,
            nonlinear_links,
            current_displacement,
        )
        effective_matrix = _build_effective_matrix(
            stiffness_matrix,
            mass_matrix,
            damping_matrix,
            nonlinear_tangent_matrix,
            mass_scale=mass_scale,
            damping_scale=damping_scale,
        )
        _assign_matrix(jacobian, effective_matrix)
        if preconditioner is not jacobian:
            _assign_matrix(preconditioner, effective_matrix)

    snes = PETSc.SNES().create(stiffness_matrix.getComm())
    snes.setFunction(assemble_residual, residual_vector)
    snes.setJacobian(assemble_jacobian, jacobian_matrix, jacobian_matrix)
    snes.setTolerances(
        atol=float(analysis.nonlinear_tolerance),
        rtol=float(analysis.nonlinear_tolerance),
        max_it=max(analysis.max_nonlinear_iterations, 1),
    )
    snes.getKSP().setType(PETSc.KSP.Type.PREONLY)
    snes.getKSP().getPC().setType(PETSc.PC.Type.LU)
    snes.setFromOptions()

    displacement = _copy_vector(displacement_predictor)
    snes.solve(None, displacement)
    if snes.getConvergedReason() <= 0:
        raise RuntimeError(
            "PETSc SNES failed to converge for FEniCSx nonlinear time history "
            f"at time {time_seconds:.12g} "
            f"(reason={snes.getConvergedReason()}, iterations={snes.getIterationNumber()})."
        )

    acceleration, velocity = _state_from_displacement(
        displacement,
        displacement_predictor,
        velocity_predictor,
        mass_scale=mass_scale,
        gamma=gamma,
        dt=dt,
    )
    return displacement, velocity, acceleration, excitation_state


def _build_time_excitation_state(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    *,
    excitation_dof: int,
    excitation,
    analysis,
    time_seconds: float,
) -> TimeExcitationState:
    phase_radians = excitation.phase_degrees * (pi / 180.0)
    omega = 2.0 * pi * default_driving_frequency_hz(excitation, analysis)
    angle = (omega * time_seconds) + phase_radians
    displacement = excitation.amplitude * sin(angle)
    velocity = excitation.amplitude * omega * cos(angle)
    acceleration = -excitation.amplitude * omega * omega * sin(angle)

    diagonal_stiffness = _vector_value(stiffness_matrix.getDiagonal(), excitation_dof)
    diagonal_mass = _vector_value(mass_matrix.getDiagonal(), excitation_dof)
    diagonal_damping = _vector_value(damping_matrix.getDiagonal(), excitation_dof)

    if excitation.kind == ExcitationKind.HARMONIC_FORCE:
        return TimeExcitationState(signal_value=displacement, equivalent_load=displacement)
    if excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        equivalent_load = (
            (diagonal_stiffness * displacement)
            + (diagonal_damping * velocity)
            + (diagonal_mass * acceleration)
        )
        return TimeExcitationState(
            signal_value=displacement,
            equivalent_load=equivalent_load,
        )
    if excitation.kind == ExcitationKind.HARMONIC_ACCELERATION:
        return TimeExcitationState(
            signal_value=acceleration,
            equivalent_load=diagonal_mass * acceleration,
        )

    raise ValueError(f"Unsupported excitation kind: {excitation.kind}")


def _build_time_load_vector(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    *,
    excitation_dof: int,
    excitation,
    analysis,
    time_seconds: float,
) -> tuple[Any, TimeExcitationState]:
    state = _build_time_excitation_state(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        excitation_dof=excitation_dof,
        excitation=excitation,
        analysis=analysis,
        time_seconds=time_seconds,
    )
    load_vector = stiffness_matrix.createVecRight()
    load_vector.set(0.0)
    load_vector.setValue(int(excitation_dof), float(state.equivalent_load))
    load_vector.assemblyBegin()
    load_vector.assemblyEnd()
    return load_vector, state


def _regularize_zero_diagonal_matrix(matrix: Any, *, replacement: float = 1.0) -> Any:
    regularized = matrix.duplicate(copy=True)
    diagonal = regularized.getDiagonal()
    values = diagonal.getArray(readonly=True)
    ownership_start, ownership_end = regularized.getOwnershipRange()

    from petsc4py import PETSc

    for local_index, value in enumerate(values):
        if abs(float(value)) > 1.0e-20:
            continue
        regularized.setValue(
            ownership_start + local_index,
            ownership_start + local_index,
            float(replacement),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    regularized.assemble()
    return regularized


def _build_initial_acceleration(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    *,
    excitation_dof: int,
    excitation,
    analysis,
) -> Any:
    mass_solver = _build_direct_solver(_regularize_zero_diagonal_matrix(mass_matrix))
    load_vector, _ = _build_time_load_vector(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        excitation_dof=excitation_dof,
        excitation=excitation,
        analysis=analysis,
        time_seconds=0.0,
    )
    return _solve_direct_system(mass_solver, load_vector)


def solve_embedded_beam_time_history_experiment(
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
) -> EmbeddedBeamTimeHistoryExperimentResult:
    require_dolfinx()
    require_petsc()
    _require_supported_time_history_model(model)

    if model.analysis.time_step_seconds <= 0.0 or model.analysis.total_time_seconds <= 0.0:
        raise RuntimeError("Time-history analysis requires positive time step and total time.")

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

    alpha, beta_damping = _resolve_rayleigh_coefficients(model)
    damping_matrix = build_embedded_rayleigh_damping_matrix(
        experiment.operator_bundle.stiffness_matrix,
        experiment.operator_bundle.mass_matrix,
        alpha=alpha,
        beta=beta_damping,
    )
    if experiment.operator_bundle.attachment_damping_matrix is not None:
        damping_matrix = _add_matrix_in_place(
            damping_matrix,
            experiment.operator_bundle.attachment_damping_matrix,
        )

    dt = float(model.analysis.time_step_seconds)
    total_steps = max(1, round(model.analysis.total_time_seconds / dt))
    output_stride = max(model.analysis.output_stride, 1)
    beta = 0.25
    gamma = 0.5
    mass_scale = 1.0 / (beta * dt * dt)
    damping_scale = gamma / (beta * dt)

    stiffness_matrix = experiment.operator_bundle.stiffness_matrix
    mass_matrix = experiment.operator_bundle.mass_matrix
    excitation_dof = response_mapping.excitation_dof
    nonlinear_links = experiment.operator_bundle.nonlinear_links

    effective_matrix = _build_effective_matrix(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        None,
        mass_scale=mass_scale,
        damping_scale=damping_scale,
    )
    effective_solver = _build_direct_solver(effective_matrix)

    displacement = stiffness_matrix.createVecRight()
    displacement.set(0.0)
    velocity = stiffness_matrix.createVecRight()
    velocity.set(0.0)
    acceleration = _build_initial_acceleration(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        excitation_dof=excitation_dof,
        excitation=model.excitation,
        analysis=model.analysis,
    )

    initial_load_vector, initial_excitation_state = _build_time_load_vector(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        excitation_dof=excitation_dof,
        excitation=model.excitation,
        analysis=model.analysis,
        time_seconds=0.0,
    )
    del initial_load_vector

    points = [
        TimeHistoryPoint(
            time_seconds=0.0,
            excitation_signal_value=initial_excitation_state.signal_value,
            excitation_load_value=initial_excitation_state.equivalent_load,
            excitation_response_value=_vector_value(displacement, excitation_dof),
            observation_values=[
                _vector_value(displacement, dof)
                for dof in response_mapping.observation_dofs
            ],
        )
    ]

    for step in range(1, total_steps + 1):
        time_seconds = float(step) * dt

        displacement_predictor = _copy_vector(displacement)
        displacement_predictor.axpy(dt, velocity)
        predictor_acceleration = _copy_vector(acceleration)
        predictor_acceleration.scale(dt * dt * (0.5 - beta))
        displacement_predictor.axpy(1.0, predictor_acceleration)

        velocity_predictor = _copy_vector(velocity)
        predictor_velocity_acceleration = _copy_vector(acceleration)
        predictor_velocity_acceleration.scale(dt * (1.0 - gamma))
        velocity_predictor.axpy(1.0, predictor_velocity_acceleration)

        if nonlinear_links:
            displacement, velocity, acceleration, excitation_state = (
                _solve_nonlinear_newmark_step_with_snes(
                    stiffness_matrix,
                    mass_matrix,
                    damping_matrix,
                    nonlinear_links,
                    displacement_predictor,
                    velocity_predictor,
                    excitation_dof=excitation_dof,
                    excitation=model.excitation,
                    analysis=model.analysis,
                    time_seconds=time_seconds,
                    mass_scale=mass_scale,
                    damping_scale=damping_scale,
                    gamma=gamma,
                    dt=dt,
                )
            )
        else:
            rhs_vector, excitation_state = _build_time_load_vector(
                stiffness_matrix,
                mass_matrix,
                damping_matrix,
                excitation_dof=excitation_dof,
                excitation=model.excitation,
                analysis=model.analysis,
                time_seconds=time_seconds,
            )

            mass_term = rhs_vector.duplicate()
            mass_matrix.mult(displacement_predictor, mass_term)
            mass_term.scale(mass_scale)
            rhs_vector.axpy(1.0, mass_term)

            damping_predictor_term = rhs_vector.duplicate()
            damping_matrix.mult(displacement_predictor, damping_predictor_term)
            damping_predictor_term.scale(damping_scale)
            rhs_vector.axpy(1.0, damping_predictor_term)

            damping_velocity_term = rhs_vector.duplicate()
            damping_matrix.mult(velocity_predictor, damping_velocity_term)
            rhs_vector.axpy(-1.0, damping_velocity_term)

            displacement = _solve_direct_system(effective_solver, rhs_vector)

            acceleration = _copy_vector(displacement)
            acceleration.axpy(-1.0, displacement_predictor)
            acceleration.scale(mass_scale)

            velocity = _copy_vector(velocity_predictor)
            acceleration_velocity_increment = _copy_vector(acceleration)
            acceleration_velocity_increment.scale(gamma * dt)
            velocity.axpy(1.0, acceleration_velocity_increment)

        if step % output_stride == 0 or step == total_steps:
            points.append(
                TimeHistoryPoint(
                    time_seconds=time_seconds,
                    excitation_signal_value=excitation_state.signal_value,
                    excitation_load_value=excitation_state.equivalent_load,
                    excitation_response_value=_vector_value(displacement, excitation_dof),
                    observation_values=[
                        _vector_value(displacement, dof)
                        for dof in response_mapping.observation_dofs
                    ],
                )
            )

    return EmbeddedBeamTimeHistoryExperimentResult(
        experiment=experiment,
        response_mapping=response_mapping,
        damping_matrix=damping_matrix,
        result=TimeHistoryResult(
            observation_names=response_mapping.observation_names,
            points=points,
        ),
    )
