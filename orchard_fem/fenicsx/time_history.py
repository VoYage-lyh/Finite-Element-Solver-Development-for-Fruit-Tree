from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

from orchard_fem.discretization.damping import compute_default_damping_ratio
from orchard_fem.domain import ExcitationKind, JointLawKind, OrchardModel
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
    unsupported_joint_laws = [
        joint.joint_id
        for joint in model.joints
        if joint.law.kind != JointLawKind.NONE
    ]
    if unsupported_joint_laws:
        raise NotImplementedError(
            "The experimental FEniCSx time-history branch does not yet support nonlinear joint laws. "
            f"Unsupported joints: {', '.join(sorted(unsupported_joint_laws))}."
        )
    if model.analysis.auto_nonlinear_levels:
        raise NotImplementedError(
            "The experimental FEniCSx time-history branch does not yet support automatic nonlinear link injection."
        )
    nonlinear_clamps = [
        clamp.branch_id
        for clamp in model.clamps
        if abs(clamp.cubic_stiffness) > 1.0e-14
    ]
    if nonlinear_clamps:
        raise NotImplementedError(
            "The experimental FEniCSx time-history branch does not yet support cubic clamp nonlinearities. "
            f"Unsupported clamps: {', '.join(sorted(nonlinear_clamps))}."
        )


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


def _build_initial_acceleration(
    stiffness_matrix: Any,
    mass_matrix: Any,
    damping_matrix: Any,
    *,
    excitation_dof: int,
    excitation,
    analysis,
) -> Any:
    mass_solver = _build_direct_solver(mass_matrix)
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
    shear_correction: float = 1.0,
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
    effective_matrix.assemble()
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
