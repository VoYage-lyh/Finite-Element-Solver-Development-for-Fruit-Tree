from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

from orchard_fem.solvers._petsc import create_aij_matrix, require_slepc


@dataclass(frozen=True)
class ModalAnalysisRequest:
    num_modes: int
    stiffness_matrix: list[list[float]]
    mass_matrix: list[list[float]]
    dof_labels: list[str] | None = None


@dataclass(frozen=True)
class ModeResult:
    mode_index: int
    eigenvalue: float
    frequency_hz: float
    modal_mass: float
    dof_labels: list[str]
    mode_shape: list[float]


@dataclass(frozen=True)
class GeneralizedEigenSystem:
    stiffness_matrix: list[list[float]]
    mass_matrix: list[list[float]]
    dof_labels: list[str]

    @classmethod
    def from_request(cls, request: ModalAnalysisRequest) -> "GeneralizedEigenSystem":
        stiffness = [[float(value) for value in row] for row in request.stiffness_matrix]
        mass = [[float(value) for value in row] for row in request.mass_matrix]
        if not stiffness or len(stiffness) != len(stiffness[0]):
            raise ValueError("Stiffness matrix must be square")
        if len(mass) != len(stiffness) or any(len(row) != len(stiffness) for row in mass):
            raise ValueError("Mass matrix must match stiffness matrix shape")

        size = len(stiffness)
        labels = (
            request.dof_labels
            if request.dof_labels is not None
            else [f"dof_{index}" for index in range(size)]
        )
        if len(labels) != size:
            raise ValueError("DOF label count must match matrix size")

        return cls(stiffness_matrix=stiffness, mass_matrix=mass, dof_labels=labels)


def _matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(matrix[row][col] * vector[col] for col in range(len(vector)))
        for row in range(len(matrix))
    ]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class SLEPcModalSolver:
    def solve(self, request: ModalAnalysisRequest) -> list[ModeResult]:
        require_slepc()

        from petsc4py import PETSc
        from slepc4py import SLEPc

        system = GeneralizedEigenSystem.from_request(request)
        stiffness = create_aij_matrix(system.stiffness_matrix)
        mass = create_aij_matrix(system.mass_matrix)

        solver = SLEPc.EPS().create()
        solver.setOperators(stiffness, mass)
        solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        ncv = min(len(system.dof_labels), max((2 * request.num_modes) + 8, 20))
        solver.setDimensions(request.num_modes, ncv)

        # For structural modal analysis we want the eigenpairs closest to zero.
        # Shift-and-invert is much more reliable than plain SMALLEST_REAL on
        # penalty-constrained generalized systems.
        solver.setTarget(0.0)
        solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        spectral_transform = solver.getST()
        spectral_transform.setType(SLEPc.ST.Type.SINVERT)
        ksp = spectral_transform.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)

        solver.setTolerances(1.0e-10, max(500, 25 * request.num_modes))
        solver.setFromOptions()
        solver.solve()

        converged = solver.getConverged()
        if converged < request.num_modes:
            raise RuntimeError(
                "SLEPc converged only "
                f"{converged} eigenpairs, requested {request.num_modes} "
                f"(reason={solver.getConvergedReason()}, iterations={solver.getIterationNumber()})"
            )

        results: list[ModeResult] = []
        eigenvector_real = stiffness.createVecRight()
        eigenvector_imag = stiffness.createVecRight()
        for mode_index in range(converged):
            eigenvalue_raw = solver.getEigenpair(mode_index, eigenvector_real, eigenvector_imag)
            eigenvalue = float(
                eigenvalue_raw.real if hasattr(eigenvalue_raw, "real") else eigenvalue_raw
            )
            if eigenvalue <= 1.0e-12:
                continue

            vector = [float(value) for value in eigenvector_real.getArray(readonly=True)]
            modal_mass = _dot(vector, _matrix_vector_multiply(system.mass_matrix, vector))
            results.append(
                ModeResult(
                    mode_index=len(results) + 1,
                    eigenvalue=eigenvalue,
                    frequency_hz=sqrt(eigenvalue) / (2.0 * pi),
                    modal_mass=modal_mass,
                    dof_labels=system.dof_labels,
                    mode_shape=vector,
                )
            )
            if len(results) >= request.num_modes:
                break

        if len(results) < request.num_modes:
            raise RuntimeError("SLEPc returned insufficient physical modes")
        return results
