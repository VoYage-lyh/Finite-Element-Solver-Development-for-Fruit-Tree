"""Batch frequency-response workflow.

Assembles K, M, C once for a given model and then sweeps multiple excitation
DOFs across the same frequency grid, reusing the block-matrix factorisation at
each frequency point.  The cost is O(N_freqs × N_specs) solves rather than the
O(N_specs × N_freqs) independent runs that separate full analyses would require.

Typical use:

    from orchard_fem.workflows.batch_excitation import ExcitationSpec, run_batch_frequency_response

    specs = [
        ExcitationSpec(branch_id="primary_a", node="tip", component="ux"),
        ExcitationSpec(branch_id="primary_b", node="tip", component="ux"),
    ]
    results = run_batch_frequency_response(model, specs)
    for batch_result in results:
        print(batch_result.excitation_spec, batch_result.result.points)
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

from orchard_fem.domain import OrchardModel
from orchard_fem.dynamics.frequency_response import (
    FrequencyResponsePoint,
    FrequencyResponseResult,
)


@dataclass(frozen=True)
class ExcitationSpec:
    """One excitation point specification for a batch frequency-response sweep.

    Parameters
    ----------
    branch_id:
        ID of the branch where the force is applied.
    node:
        Node designator: ``"root"``, ``"tip"``, or an integer node index string.
    component:
        DOF component, one of ``"ux"``, ``"uy"``, ``"uz"``,
        ``"rx"``, ``"ry"``, ``"rz"``.
    amplitude:
        Force amplitude [N] (default 1.0).
    phase_degrees:
        Phase offset of the harmonic load [°] (default 0.0).
    """

    branch_id: str
    node: str
    component: str
    amplitude: float = 1.0
    phase_degrees: float = 0.0


@dataclass(frozen=True)
class BatchExcitationResult:
    """Frequency-response result for one excitation specification."""

    excitation_spec: ExcitationSpec
    excitation_dof: int
    result: FrequencyResponseResult


def _resolve_excitation_dof(
    model: OrchardModel,
    spec: ExcitationSpec,
    space_bundle: Any,
    *,
    atol: float = 1.0e-8,
) -> int:
    from orchard_fem.fenicsx.dofs import (
        _target_point,
        resolve_embedded_beam_component_dof,
    )

    branch = model.require_branch(spec.branch_id)
    point = _target_point(branch, spec.node)
    return resolve_embedded_beam_component_dof(
        space_bundle,
        point,
        spec.component,
        atol=atol,
    )


def _build_batch_rhs(
    block_matrix: Any,
    *,
    excitation_dof: int,
    system_size: int,
    amplitude: float,
    phase_degrees: float,
) -> Any:
    phase_radians = phase_degrees * (pi / 180.0)
    load_real = amplitude * cos(phase_radians)
    load_imag = amplitude * sin(phase_radians)

    rhs = block_matrix.createVecRight()
    rhs.set(0.0)
    rhs.setValue(int(excitation_dof), load_real)
    rhs.setValue(int(system_size + excitation_dof), load_imag)
    rhs.assemblyBegin()
    rhs.assemblyEnd()
    return rhs


def _create_cached_ksp(block_matrix: Any) -> Any:
    """Create a PETSc KSP with LU factorisation ready for multiple RHS solves."""
    from petsc4py import PETSc

    solver = PETSc.KSP().create(block_matrix.getComm())
    solver.setOperators(block_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()
    solver.setUp()
    return solver


def _solve_with_ksp(solver: Any, rhs: Any) -> Any:
    solution = rhs.duplicate()
    solver.solve(rhs, solution)
    if solver.getConvergedReason() <= 0:
        raise RuntimeError(
            "PETSc KSP failed during batch frequency-response solve "
            f"(reason={solver.getConvergedReason()})."
        )
    return solution


def run_batch_frequency_response(
    model: OrchardModel,
    excitation_specs: list[ExcitationSpec],
    *,
    observation_dof_atol: float = 1.0e-8,
) -> list[BatchExcitationResult]:
    """Solve frequency response for multiple excitation points with a single assembly.

    The system matrices (K, M, C) are assembled once from *model*.  For every
    frequency in the model's frequency grid the 2N×2N real-block dynamic matrix
    ``A(ω) = [[K-ω²M, -ωC], [ωC, K-ω²M]]`` is built and factorised **once**,
    then the linear system is solved for each excitation spec in turn, reusing
    the cached LU factorisation.

    Parameters
    ----------
    model:
        Assembled ``OrchardModel``.  Must satisfy the standard frequency-response
        pre-conditions (at least one branch, one clamp, valid excitation target).
    excitation_specs:
        List of excitation specifications; each describes a branch node / DOF
        component where a unit harmonic force is applied.
    observation_dof_atol:
        Geometric tolerance for locating DOFs.

    Returns
    -------
    list[BatchExcitationResult]
        One result per entry in *excitation_specs*, in the same order.
    """
    from orchard_fem.fenicsx.assembly import assemble_fenicsx_system
    from orchard_fem.fenicsx.availability import require_dolfinx
    from orchard_fem.fenicsx.frequency_response import (
        _build_real_block_dynamic_matrix,
        _frequency_grid,
        _resolve_rayleigh_coefficients,
        _vector_value,
        build_embedded_rayleigh_damping_matrix,
    )
    from orchard_fem.fenicsx.mpc import (
        apply_mpc_to_real_block_vector_in_place,
        backsubstitute_mpc_real_block_vector,
    )
    from orchard_fem.fenicsx.petsc_ops import add_matrix_in_place as _add_matrix_in_place

    require_dolfinx()

    if not excitation_specs:
        return []

    assembly = assemble_fenicsx_system(model)
    experiment = assembly.experiment
    op = experiment.operator_bundle

    alpha, beta = _resolve_rayleigh_coefficients(model)
    damping_matrix = build_embedded_rayleigh_damping_matrix(
        op.stiffness_matrix,
        op.mass_matrix,
        alpha=alpha,
        beta=beta,
    )
    if op.attachment_damping_matrix is not None:
        _add_matrix_in_place(damping_matrix, op.attachment_damping_matrix)

    system_size = op.stiffness_matrix.getSize()[0]
    mpc = op.mpc

    # Resolve all excitation DOFs up front.
    excitation_dofs: list[int] = [
        _resolve_excitation_dof(
            model,
            spec,
            experiment.space_bundle,
            atol=observation_dof_atol,
        )
        for spec in excitation_specs
    ]

    # Build a per-spec accumulator for response points.
    spec_points: list[list[FrequencyResponsePoint]] = [[] for _ in excitation_specs]

    target_frequencies = _frequency_grid(model.analysis)
    for frequency_hz in target_frequencies:
        omega = 2.0 * pi * frequency_hz
        block_matrix = _build_real_block_dynamic_matrix(
            op.stiffness_matrix,
            op.mass_matrix,
            damping_matrix,
            omega=omega,
        )
        ksp = _create_cached_ksp(block_matrix)

        for spec_index, (spec, excitation_dof) in enumerate(
            zip(excitation_specs, excitation_dofs)
        ):
            rhs = _build_batch_rhs(
                block_matrix,
                excitation_dof=excitation_dof,
                system_size=system_size,
                amplitude=spec.amplitude,
                phase_degrees=spec.phase_degrees,
            )
            apply_mpc_to_real_block_vector_in_place(
                mpc,
                rhs,
                system_size=system_size,
                template_matrix=op.stiffness_matrix,
            )
            solution = _solve_with_ksp(ksp, rhs)
            solution = backsubstitute_mpc_real_block_vector(
                mpc,
                solution,
                system_size=system_size,
                template_matrix=op.stiffness_matrix,
            )

            excitation_real = _vector_value(solution, excitation_dof)
            excitation_imag = _vector_value(solution, system_size + excitation_dof)
            magnitude = (excitation_real**2 + excitation_imag**2) ** 0.5
            spec_points[spec_index].append(
                FrequencyResponsePoint(
                    frequency_hz=frequency_hz,
                    excitation_response_magnitude=magnitude,
                    observation_magnitudes=[magnitude],
                )
            )

    return [
        BatchExcitationResult(
            excitation_spec=spec,
            excitation_dof=excitation_dof,
            result=FrequencyResponseResult(
                observation_names=[f"{spec.branch_id}_{spec.node}_{spec.component}"],
                points=points,
            ),
        )
        for spec, excitation_dof, points in zip(
            excitation_specs, excitation_dofs, spec_points
        )
    ]
