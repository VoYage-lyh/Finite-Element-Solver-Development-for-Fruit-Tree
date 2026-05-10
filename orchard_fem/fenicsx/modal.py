from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.assembly import assemble_fenicsx_system
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.operator_bundle import (
    EmbeddedBeamExperimentBundle,
)
from orchard_fem.numerics import require_slepc


@dataclass(frozen=True)
class EmbeddedBeamModalResult:
    mode_index: int
    eigenvalue: float
    frequency_hz: float
    modal_mass: float
    mode_shape: list[float]


@dataclass(frozen=True)
class EmbeddedBeamModalExperimentResult:
    experiment: EmbeddedBeamExperimentBundle
    modes: list[EmbeddedBeamModalResult]


def _require_supported_modal_model(model: OrchardModel) -> None:
    if not model.branches:
        raise ValueError("Modal analysis requires at least one branch.")
    if not model.clamps:
        raise ValueError("Modal analysis requires at least one clamp boundary condition.")
    branch_ids = {b.branch_id for b in model.branches}
    if model.excitation.target_branch_id not in branch_ids:
        raise ValueError(
            f"Excitation target_branch_id '{model.excitation.target_branch_id}' "
            "does not match any branch in the model."
        )


def _solve_petsc_generalized_modes(
    stiffness_matrix: Any,
    mass_matrix: Any,
    *,
    num_modes: int,
    mpc: Any | None = None,
) -> list[EmbeddedBeamModalResult]:
    require_slepc()

    from petsc4py import PETSc
    from slepc4py import SLEPc

    if num_modes <= 0:
        raise ValueError("num_modes must be positive.")

    solver = SLEPc.EPS().create()
    solver.setOperators(stiffness_matrix, mass_matrix)
    solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

    matrix_size = stiffness_matrix.getSize()[0]
    ncv = min(matrix_size, max((2 * num_modes) + 8, 20))
    solver.setDimensions(num_modes, ncv)
    solver.setTarget(0.0)
    solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    spectral_transform = solver.getST()
    spectral_transform.setType(SLEPc.ST.Type.SINVERT)
    ksp = spectral_transform.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)

    solver.setTolerances(1.0e-10, max(500, 25 * num_modes))
    solver.setFromOptions()
    solver.solve()

    converged = solver.getConverged()
    if converged < num_modes:
        raise RuntimeError(
            "SLEPc converged only "
            f"{converged} eigenpairs, requested {num_modes} "
            f"(reason={solver.getConvergedReason()}, iterations={solver.getIterationNumber()})"
        )

    eigenvector_real = stiffness_matrix.createVecRight()
    eigenvector_imag = stiffness_matrix.createVecRight()
    mass_times_mode = stiffness_matrix.createVecRight()

    results: list[EmbeddedBeamModalResult] = []
    for mode_index in range(converged):
        eigenvalue_raw = solver.getEigenpair(mode_index, eigenvector_real, eigenvector_imag)
        if mpc is not None:
            mpc.backsubstitution(eigenvector_real)
        eigenvalue = float(
            eigenvalue_raw.real if hasattr(eigenvalue_raw, "real") else eigenvalue_raw
        )
        if eigenvalue <= 1.0e-12:
            continue

        mass_matrix.mult(eigenvector_real, mass_times_mode)
        modal_mass = float(eigenvector_real.dot(mass_times_mode))
        mode_shape = [float(value) for value in eigenvector_real.getArray(readonly=True)]
        results.append(
            EmbeddedBeamModalResult(
                mode_index=len(results) + 1,
                eigenvalue=eigenvalue,
                frequency_hz=sqrt(eigenvalue) / (2.0 * pi),
                modal_mass=modal_mass,
                mode_shape=mode_shape,
            )
        )
        if len(results) >= num_modes:
            break

    if len(results) < num_modes:
        raise RuntimeError("SLEPc returned insufficient physical modes")
    return results


def solve_embedded_beam_modal_experiment(
    model: OrchardModel,
    *,
    num_modes: int = 6,
    polynomial_degree: int = 1,
    spec: EmbeddedLineMeshSpec | None = None,
    shear_correction: float = 0.4,
    comm: object | None = None,
    partitioner: object | None = None,
    max_facet_to_cell_links: int = 2,
    use_model_clamps: bool = True,
    clamp_tolerance: float = 1.0e-8,
) -> EmbeddedBeamModalExperimentResult:
    require_dolfinx()
    _require_supported_modal_model(model)

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
    modes = _solve_petsc_generalized_modes(
        assembly.stiffness_matrix,
        assembly.mass_matrix,
        num_modes=num_modes,
        mpc=assembly.experiment.operator_bundle.mpc,
    )
    return EmbeddedBeamModalExperimentResult(
        experiment=assembly.experiment,
        modes=modes,
    )
