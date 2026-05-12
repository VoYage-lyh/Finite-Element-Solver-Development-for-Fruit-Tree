from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

from orchard_fem.discretization.damping import compute_default_damping_ratio
from orchard_fem.dynamics.nonlinear import nonlinear_force, nonlinear_tangent
from orchard_fem.domain import ExcitationKind, OrchardModel
from orchard_fem.dynamics.excitation import (
    TimeExcitationState,
    build_harmonic_kinematics,
    default_driving_frequency_hz,
)
from orchard_fem.dynamics.time_history import TimeHistoryPoint, TimeHistoryResult
from orchard_fem.fenicsx.assembly import assemble_fenicsx_system
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.dofs import (
    EmbeddedBeamResponseMapping,
    resolve_embedded_beam_response_mapping,
)
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.frequency_response import (
    build_embedded_rayleigh_damping_matrix,
)
from orchard_fem.fenicsx.mpc import (
    apply_mpc_to_vector_in_place,
    backsubstitute_mpc_vector,
)
from orchard_fem.fenicsx.operator_bundle import (
    EmbeddedBeamExperimentBundle,
)
from orchard_fem.fenicsx.petsc_ops import (
    accumulate_owned_matrix_value as _accumulate_owned_matrix_value,
    accumulate_owned_vector_value as _accumulate_owned_vector_value,
    add_matrix_copy as _add_matrix_in_place,
    copy_matrix_like as _extend_matrix_like,
    create_empty_aij_matrix_like as _create_empty_aij_matrix_like,
    extend_vector_like as _extend_vector_like,
)
from orchard_fem.materials.base import build_material_lookup
from orchard_fem.numerics import require_petsc


@dataclass(frozen=True)
class EmbeddedBeamTimeHistoryExperimentResult:
    experiment: EmbeddedBeamExperimentBundle
    response_mapping: EmbeddedBeamResponseMapping
    damping_matrix: Any
    result: TimeHistoryResult


@dataclass(frozen=True)
class UflElasticOperatorBridge:
    form_bundle: Any
    compiled_residual_form: Any
    compiled_jacobian_form: Any
    stiffness_augmentation_matrix: Any


def _require_supported_time_history_model(model: OrchardModel) -> None:
    if not model.branches:
        raise ValueError("Time history analysis requires at least one branch.")
    if not model.clamps:
        raise ValueError("Time history analysis requires at least one clamp boundary condition.")
    branch_ids = {b.branch_id for b in model.branches}
    if model.excitation.target_branch_id not in branch_ids:
        raise ValueError(
            f"Excitation target_branch_id '{model.excitation.target_branch_id}' "
            "does not match any branch in the model."
        )
    if model.analysis.time_step_seconds <= 0.0:
        raise ValueError(
            f"time_step_seconds must be positive, got {model.analysis.time_step_seconds}."
        )


def _resolve_rayleigh_coefficients(model: OrchardModel) -> tuple[float, float]:
    alpha = float(model.analysis.rayleigh_alpha)
    beta = float(model.analysis.rayleigh_beta)

    if abs(alpha) < 1.0e-14 and abs(beta) < 1.0e-14:
        material_lookup = build_material_lookup(model.materials)
        zeta = compute_default_damping_ratio(model, material_lookup)
        omega_ref = 2.0 * pi * max(
            default_driving_frequency_hz(model.excitation, model.analysis),
            0.1,
        )
        beta = (2.0 * zeta / omega_ref) if omega_ref > 0.0 else 0.0

    return alpha, beta


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


def _copy_backsubstituted_vector(mpc: Any | None, vector: Any) -> Any:
    copied = _copy_vector(vector)
    backsubstitute_mpc_vector(mpc, copied)
    return copied


def _vector_value(vector: Any, global_index: int) -> float:
    values = vector.getValues([int(global_index)])
    return float(values[0])


def _build_time_history_point(
    *,
    mpc: Any | None,
    time_seconds: float,
    excitation_state: TimeExcitationState,
    displacement: Any,
    acceleration: Any,
    excitation_dof: int,
    observation_dofs: list[int],
) -> TimeHistoryPoint:
    output_displacement = _copy_backsubstituted_vector(mpc, displacement)
    output_acceleration = _copy_backsubstituted_vector(mpc, acceleration)
    return TimeHistoryPoint(
        time_seconds=time_seconds,
        excitation_signal_value=excitation_state.signal_value,
        excitation_load_value=excitation_state.equivalent_load,
        excitation_response_value=_vector_value(output_displacement, excitation_dof),
        observation_values=[
            _vector_value(output_displacement, dof)
            for dof in observation_dofs
        ],
        excitation_acceleration_value=_vector_value(output_acceleration, excitation_dof),
        observation_acceleration_values=[
            _vector_value(output_acceleration, dof)
            for dof in observation_dofs
        ],
    )


def _solve_direct_system(solver: Any, rhs_vector: Any) -> Any:
    solution = rhs_vector.duplicate()
    solver.solve(rhs_vector, solution)
    if solver.getConvergedReason() <= 0:
        raise RuntimeError(
            "PETSc KSP failed to converge for experimental FEniCSx time history "
            f"(reason={solver.getConvergedReason()})."
        )
    return solution


def _set_ufl_state_from_displacement(form_bundle: Any, displacement: Any) -> None:
    state_values = form_bundle.state_function.x.array
    source_values = displacement.getArray(readonly=True)
    if state_values.size > source_values.size:
        state_values[:] = displacement.getValues(list(range(state_values.size)))
    else:
        state_values[:] = source_values[: state_values.size]
    form_bundle.state_function.x.scatter_forward()


def _assemble_ufl_elastic_residual(
    bridge: UflElasticOperatorBridge,
    template_vector: Any,
    displacement: Any,
) -> Any:
    require_dolfinx()

    from petsc4py import PETSc
    from dolfinx.fem import petsc as dolfinx_fem_petsc

    _set_ufl_state_from_displacement(bridge.form_bundle, displacement)
    residual = dolfinx_fem_petsc.assemble_vector(bridge.compiled_residual_form)
    ghost_update = getattr(residual, "ghostUpdate", None)
    if ghost_update is not None:
        ghost_update(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    residual.assemblyBegin()
    residual.assemblyEnd()
    return _extend_vector_like(template_vector, residual)


def _assemble_ufl_elastic_jacobian(
    bridge: UflElasticOperatorBridge,
    template_matrix: Any,
    displacement: Any,
) -> Any:
    require_dolfinx()

    from dolfinx.fem import petsc as dolfinx_fem_petsc

    _set_ufl_state_from_displacement(bridge.form_bundle, displacement)
    jacobian = dolfinx_fem_petsc.assemble_matrix(bridge.compiled_jacobian_form)
    jacobian.assemble()
    return _extend_matrix_like(template_matrix, jacobian)


def _build_ufl_elastic_operator_bridge(
    form_bundle: Any,
    stiffness_matrix: Any,
) -> UflElasticOperatorBridge:
    require_dolfinx()

    from dolfinx import fem as dolfinx_fem
    from petsc4py import PETSc

    compiled_residual_form = dolfinx_fem.form(form_bundle.residual_form)
    compiled_jacobian_form = dolfinx_fem.form(form_bundle.jacobian_form)
    bridge = UflElasticOperatorBridge(
        form_bundle=form_bundle,
        compiled_residual_form=compiled_residual_form,
        compiled_jacobian_form=compiled_jacobian_form,
        stiffness_augmentation_matrix=stiffness_matrix,
    )

    zero_displacement = stiffness_matrix.createVecRight()
    zero_displacement.set(0.0)
    ufl_jacobian = _assemble_ufl_elastic_jacobian(
        bridge,
        stiffness_matrix,
        zero_displacement,
    )
    augmentation_matrix = stiffness_matrix.duplicate(copy=True)
    augmentation_matrix.axpy(
        -1.0,
        ufl_jacobian,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    augmentation_matrix.assemble()
    return UflElasticOperatorBridge(
        form_bundle=form_bundle,
        compiled_residual_form=compiled_residual_form,
        compiled_jacobian_form=compiled_jacobian_form,
        stiffness_augmentation_matrix=augmentation_matrix,
    )


def _build_ufl_augmented_elastic_force(
    bridge: UflElasticOperatorBridge,
    stiffness_matrix: Any,
    displacement: Any,
) -> Any:
    elastic_force = _assemble_ufl_elastic_residual(
        bridge,
        stiffness_matrix.createVecRight(),
        displacement,
    )
    augmentation_force = stiffness_matrix.createVecRight()
    bridge.stiffness_augmentation_matrix.mult(displacement, augmentation_force)
    elastic_force.axpy(1.0, augmentation_force)
    return elastic_force


def _build_ufl_augmented_elastic_jacobian(
    bridge: UflElasticOperatorBridge,
    stiffness_matrix: Any,
    displacement: Any,
) -> Any:
    return _add_matrix_in_place(
        _assemble_ufl_elastic_jacobian(bridge, stiffness_matrix, displacement),
        bridge.stiffness_augmentation_matrix,
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


def _build_residual_vector_from_elastic_force(
    mass_matrix: Any,
    damping_matrix: Any,
    velocity: Any,
    acceleration: Any,
    elastic_force_vector: Any,
    nonlinear_force_vector: Any,
    external_load_vector: Any,
) -> Any:
    residual = external_load_vector.duplicate()
    mass_matrix.mult(acceleration, residual)

    damping_term = external_load_vector.duplicate()
    damping_matrix.mult(velocity, damping_term)
    residual.axpy(1.0, damping_term)

    residual.axpy(1.0, elastic_force_vector)
    residual.axpy(1.0, nonlinear_force_vector)
    residual.axpy(-1.0, external_load_vector)
    return residual


def _assign_vector(target: Any, source: Any) -> None:
    target.set(0.0)
    target.axpy(1.0, source)


def _set_vector_value(vector: Any, index: int, value: float) -> None:
    vector.setValue(int(index), float(value))
    vector.assemblyBegin()
    vector.assemblyEnd()


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
    ufl_bridge: UflElasticOperatorBridge,
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
    mpc: Any | None = None,
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
    apply_mpc_to_vector_in_place(mpc, load_vector)
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
        elastic_force_vector = _build_ufl_augmented_elastic_force(
            ufl_bridge,
            stiffness_matrix,
            current_displacement,
        )
        computed_residual = _build_residual_vector_from_elastic_force(
            mass_matrix,
            damping_matrix,
            velocity,
            acceleration,
            elastic_force_vector,
            nonlinear_force_vector,
            load_vector,
        )
        apply_mpc_to_vector_in_place(mpc, computed_residual)
        _assign_vector(residual, computed_residual)

    def assemble_jacobian(_snes, current_displacement, jacobian, preconditioner) -> None:
        _, nonlinear_tangent_matrix = _evaluate_nonlinear_force_and_tangent(
            stiffness_matrix,
            nonlinear_links,
            current_displacement,
        )
        elastic_jacobian = _build_ufl_augmented_elastic_jacobian(
            ufl_bridge,
            stiffness_matrix,
            current_displacement,
        )
        effective_matrix = _build_effective_matrix(
            elastic_jacobian,
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
    snes.setType(PETSc.SNES.Type.NEWTONLS)
    snes.setFunction(assemble_residual, residual_vector)
    snes.setJacobian(assemble_jacobian, jacobian_matrix, jacobian_matrix)
    snes.setTolerances(
        atol=float(analysis.nonlinear_tolerance),
        rtol=float(analysis.nonlinear_tolerance),
        max_it=max(analysis.max_nonlinear_iterations, 1),
    )
    snes.getLineSearch().setType(PETSc.SNESLineSearch.Type.BASIC)
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
    kinematics = build_harmonic_kinematics(excitation, analysis, time_seconds)

    diagonal_stiffness = _vector_value(stiffness_matrix.getDiagonal(), excitation_dof)
    diagonal_mass = _vector_value(mass_matrix.getDiagonal(), excitation_dof)
    diagonal_damping = _vector_value(damping_matrix.getDiagonal(), excitation_dof)

    if excitation.kind == ExcitationKind.HARMONIC_FORCE:
        return TimeExcitationState(
            signal_value=kinematics.displacement,
            equivalent_load=kinematics.displacement,
        )
    if excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        equivalent_load = (
            (diagonal_stiffness * kinematics.displacement)
            + (diagonal_damping * kinematics.velocity)
            + (diagonal_mass * kinematics.acceleration)
        )
        return TimeExcitationState(
            signal_value=kinematics.displacement,
            equivalent_load=equivalent_load,
        )
    if excitation.kind == ExcitationKind.HARMONIC_ACCELERATION:
        return TimeExcitationState(
            signal_value=kinematics.acceleration,
            equivalent_load=diagonal_mass * kinematics.acceleration,
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
    mpc: Any | None = None,
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
    apply_mpc_to_vector_in_place(mpc, load_vector)
    return _solve_direct_system(mass_solver, load_vector)


def _build_dirichlet_excitation_system(
    effective_matrix: Any,
    excitation_dof: int,
) -> Any:
    """Build Dirichlet-enforced solver for prescribed-displacement excitation.

    Zeros row excitation_dof and sets diagonal to 1.  The matrix column is
    intentionally kept intact so that K_eff[i, excitation_dof] * u_prescribed
    is handled implicitly through the matrix–vector product during the solve
    (no separate column-correction step needed).
    """
    dirichlet_matrix = effective_matrix.copy()
    dirichlet_matrix.zeroRows([int(excitation_dof)], diag=1.0)
    dirichlet_matrix.assemble()
    return _build_direct_solver(dirichlet_matrix)


def solve_corotational_time_history_experiment(
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
    newton_max_iter: int = 12,
    newton_tol: float = 1.0e-8,
) -> EmbeddedBeamTimeHistoryExperimentResult:
    """Solve large-deformation time history using the corotational beam formulation.

    Replaces the UFL elastic operator with explicit element-by-element corotational
    assembly at each Newton iteration of the Newmark-β corrector step.  Activated
    when ``model.analysis.use_corotational is True``.

    Parameters
    ----------
    model:
        Orchard model.  ``model.analysis.use_corotational`` should be ``True``.
    newton_max_iter:
        Maximum Newton iterations per time step.
    newton_tol:
        Convergence criterion on the L∞ residual norm relative to the load norm.

    Returns
    -------
    EmbeddedBeamTimeHistoryExperimentResult
    """
    require_dolfinx()
    require_petsc()
    _require_supported_time_history_model(model)

    if model.analysis.time_step_seconds <= 0.0 or model.analysis.total_time_seconds <= 0.0:
        raise RuntimeError("Time-history analysis requires positive time step and total time.")

    from orchard_fem.fenicsx.corotational import (
        assemble_corotational_internal_force,
        assemble_corotational_tangent_stiffness,
    )

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
    beta_newmark = 0.25
    gamma = 0.5
    mass_scale = 1.0 / (beta_newmark * dt * dt)
    damping_scale = gamma / (beta_newmark * dt)

    mass_matrix = experiment.operator_bundle.mass_matrix
    excitation_dof = response_mapping.excitation_dof
    mpc = experiment.operator_bundle.mpc

    stiffness_matrix = experiment.operator_bundle.stiffness_matrix

    # Gravity prestress is already folded into the tangent/geometric stiffness.
    # The transient unknown is an increment about that prestressed equilibrium.
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
        mpc=mpc,
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
        _build_time_history_point(
            mpc=mpc,
            time_seconds=0.0,
            excitation_state=initial_excitation_state,
            displacement=displacement,
            acceleration=acceleration,
            excitation_dof=excitation_dof,
            observation_dofs=response_mapping.observation_dofs,
        )
    ]

    for step in range(1, total_steps + 1):
        time_seconds = float(step) * dt

        # Predictor
        u_pred = _copy_vector(displacement)
        u_pred.axpy(dt, velocity)
        acc_term = _copy_vector(acceleration)
        acc_term.scale(dt * dt * (0.5 - beta_newmark))
        u_pred.axpy(1.0, acc_term)

        v_pred = _copy_vector(velocity)
        v_acc = _copy_vector(acceleration)
        v_acc.scale(dt * (1.0 - gamma))
        v_pred.axpy(1.0, v_acc)

        f_ext, excitation_state = _build_time_load_vector(
            stiffness_matrix,
            mass_matrix,
            damping_matrix,
            excitation_dof=excitation_dof,
            excitation=model.excitation,
            analysis=model.analysis,
            time_seconds=time_seconds,
        )

        # Newton corrector loop
        u_iter = _copy_vector(u_pred)
        for _newton_iter in range(newton_max_iter):
            # Corotational assembly
            K_tang = assemble_corotational_tangent_stiffness(assembly, u_iter)
            f_int = assemble_corotational_internal_force(assembly, u_iter)

            # Effective stiffness: K_eff = K_tang + mass_scale*M + damping_scale*C
            K_eff = _build_effective_matrix(
                K_tang,
                mass_matrix,
                damping_matrix,
                None,
                mass_scale=mass_scale,
                damping_scale=damping_scale,
            )

            # Residual: R = mass_scale*M*(u-u_pred) + C*(damping_scale*(u-u_pred)+v_pred)
            #              + f_int - f_ext
            delta_u = _copy_vector(u_iter)
            delta_u.axpy(-1.0, u_pred)          # delta_u = u_iter - u_pred

            r_vec = stiffness_matrix.createVecRight()
            r_vec.set(0.0)

            mass_term = stiffness_matrix.createVecRight()
            mass_matrix.mult(delta_u, mass_term)
            r_vec.axpy(mass_scale, mass_term)

            damp_arg = _copy_vector(delta_u)
            damp_arg.scale(damping_scale)
            damp_arg.axpy(1.0, v_pred)          # damping_scale*(u-u_pred) + v_pred
            damp_term = stiffness_matrix.createVecRight()
            damping_matrix.mult(damp_arg, damp_term)
            r_vec.axpy(1.0, damp_term)

            r_vec.axpy(1.0, f_int)              # r += f_int
            r_vec.axpy(-1.0, f_ext)             # r -= f_ext

            apply_mpc_to_vector_in_place(mpc, r_vec)

            residual_norm = r_vec.norm()
            f_ext_norm = f_ext.norm()
            if residual_norm <= newton_tol * max(f_ext_norm, 1.0):
                break

            newton_solver = _build_direct_solver(K_eff)
            neg_r = _copy_vector(r_vec)
            neg_r.scale(-1.0)
            correction = _solve_direct_system(newton_solver, neg_r)
            u_iter.axpy(1.0, correction)

        displacement = u_iter

        delta_u_final = _copy_vector(displacement)
        delta_u_final.axpy(-1.0, u_pred)
        acceleration = _copy_vector(delta_u_final)
        acceleration.scale(mass_scale)

        velocity = _copy_vector(v_pred)
        gamma_dt_acc = _copy_vector(acceleration)
        gamma_dt_acc.scale(gamma * dt)
        velocity.axpy(1.0, gamma_dt_acc)

        if step % output_stride == 0 or step == total_steps:
            points.append(
                _build_time_history_point(
                    mpc=mpc,
                    time_seconds=time_seconds,
                    excitation_state=excitation_state,
                    displacement=displacement,
                    acceleration=acceleration,
                    excitation_dof=excitation_dof,
                    observation_dofs=response_mapping.observation_dofs,
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
    ufl_bridge = (
        _build_ufl_elastic_operator_bridge(experiment.form_bundle, stiffness_matrix)
        if nonlinear_links
        else None
    )

    effective_matrix = _build_effective_matrix(
        stiffness_matrix,
        mass_matrix,
        damping_matrix,
        None,
        mass_scale=mass_scale,
        damping_scale=damping_scale,
    )
    effective_solver = _build_direct_solver(effective_matrix)

    dirichlet_solver: Any | None = None
    if model.excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        dirichlet_solver = _build_dirichlet_excitation_system(
            effective_matrix, excitation_dof
        )

    # Gravity prestress is already folded into the tangent/geometric stiffness.
    # The transient unknown is an increment about that prestressed equilibrium.
    displacement = stiffness_matrix.createVecRight()
    displacement.set(0.0)
    velocity = stiffness_matrix.createVecRight()
    velocity.set(0.0)
    if model.excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        initial_kinematics = build_harmonic_kinematics(model.excitation, model.analysis, 0.0)
        _set_vector_value(
            displacement,
            excitation_dof,
            initial_kinematics.displacement,
        )
        _set_vector_value(velocity, excitation_dof, initial_kinematics.velocity)
        acceleration = stiffness_matrix.createVecRight()
        acceleration.set(0.0)
        _set_vector_value(acceleration, excitation_dof, initial_kinematics.acceleration)
        initial_excitation_state = TimeExcitationState(
            signal_value=initial_kinematics.displacement,
            equivalent_load=initial_kinematics.displacement,
        )
    else:
        acceleration = _build_initial_acceleration(
            stiffness_matrix,
            mass_matrix,
            damping_matrix,
            excitation_dof=excitation_dof,
            excitation=model.excitation,
            analysis=model.analysis,
            mpc=experiment.operator_bundle.mpc,
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
        _build_time_history_point(
            mpc=experiment.operator_bundle.mpc,
            time_seconds=0.0,
            excitation_state=initial_excitation_state,
            displacement=displacement,
            acceleration=acceleration,
            excitation_dof=excitation_dof,
            observation_dofs=response_mapping.observation_dofs,
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
                    ufl_bridge,
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
                    mpc=experiment.operator_bundle.mpc,
                )
            )
        elif dirichlet_solver is not None:
            # Prescribed displacement is an increment about the prestressed equilibrium.
            # Row excitation_dof in dirichlet_matrix is [0...1...0]; the column is
            # kept intact so K_eff[i, excitation_dof]·u_prescribed is handled
            # implicitly by the matrix-vector product, so no separate column
            # correction needed (which would double-count the coupling).
            kinematics = build_harmonic_kinematics(
                model.excitation,
                model.analysis,
                time_seconds,
            )
            u_prescribed = float(kinematics.displacement)
            v_prescribed = float(kinematics.velocity)
            a_prescribed = float(kinematics.acceleration)
            excitation_state = TimeExcitationState(
                signal_value=kinematics.displacement,
                equivalent_load=u_prescribed,
            )

            rhs_vector = stiffness_matrix.createVecRight()
            rhs_vector.set(0.0)

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

            # Enforce Dirichlet: row equation → u[excitation_dof] = u_prescribed
            rhs_vector.setValue(int(excitation_dof), u_prescribed)
            rhs_vector.assemblyBegin()
            rhs_vector.assemblyEnd()

            displacement = _solve_direct_system(dirichlet_solver, rhs_vector)

            acceleration = _copy_vector(displacement)
            acceleration.axpy(-1.0, displacement_predictor)
            acceleration.scale(mass_scale)

            velocity = _copy_vector(velocity_predictor)
            gamma_dt_acc = _copy_vector(acceleration)
            gamma_dt_acc.scale(gamma * dt)
            velocity.axpy(1.0, gamma_dt_acc)

            # Enforce prescribed v and a at excitation DOF so the Newmark
            # predictor (u_pred = u + dt·v + dt²·a) stays accurate next step.
            velocity.setValue(int(excitation_dof), v_prescribed)
            velocity.assemblyBegin()
            velocity.assemblyEnd()
            acceleration.setValue(int(excitation_dof), a_prescribed)
            acceleration.assemblyBegin()
            acceleration.assemblyEnd()

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
            apply_mpc_to_vector_in_place(experiment.operator_bundle.mpc, rhs_vector)

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
                _build_time_history_point(
                    mpc=experiment.operator_bundle.mpc,
                    time_seconds=time_seconds,
                    excitation_state=excitation_state,
                    displacement=displacement,
                    acceleration=acceleration,
                    excitation_dof=excitation_dof,
                    observation_dofs=response_mapping.observation_dofs,
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
