from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt


def slepc_available() -> bool:
    try:
        import slepc4py  # noqa: F401
    except ImportError:
        return False
    return True


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


def _zeros(size: int) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _identity(size: int) -> list[list[float]]:
    result = _zeros(size)
    for index in range(size):
        result[index][index] = 1.0
    return result


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    rows = len(left)
    inner = len(left[0])
    cols = len(right[0])
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            result[row][col] = sum(left[row][k] * right[k][col] for k in range(inner))
    return result


def _matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(len(vector))) for row in range(len(matrix))]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cholesky_lower(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = _zeros(size)
    for row in range(size):
        for col in range(row + 1):
            value = matrix[row][col]
            for index in range(col):
                value -= lower[row][index] * lower[col][index]

            if row == col:
                if value <= 0.0:
                    raise RuntimeError("Cholesky decomposition requires an SPD mass matrix")
                lower[row][col] = sqrt(value)
            else:
                lower[row][col] = value / lower[col][col]
    return lower


def _inverse_lower(lower: list[list[float]]) -> list[list[float]]:
    size = len(lower)
    inverse = _zeros(size)
    for col in range(size):
        inverse[col][col] = 1.0 / lower[col][col]
        for row in range(col + 1, size):
            value = 0.0
            for index in range(col, row):
                value += lower[row][index] * inverse[index][col]
            inverse[row][col] = -value / lower[row][row]
    return inverse


@dataclass(frozen=True)
class _EigenResult:
    eigenvalues: list[float]
    eigenvectors: list[list[float]]


def _jacobi_eigendecomposition(
    matrix: list[list[float]],
    tolerance: float = 1.0e-12,
    max_iterations: int = -1,
) -> _EigenResult:
    size = len(matrix)
    working = [row[:] for row in matrix]
    eigenvectors = _identity(size)
    iteration_limit = max_iterations if max_iterations > 0 else max(200, 50 * size * size)

    for _ in range(iteration_limit):
        pivot_row = 0
        pivot_col = 1
        max_off_diagonal = 0.0

        for row in range(size):
            for col in range(row + 1, size):
                value = abs(working[row][col])
                if value > max_off_diagonal:
                    max_off_diagonal = value
                    pivot_row = row
                    pivot_col = col

        if max_off_diagonal < tolerance:
            break

        app = working[pivot_row][pivot_row]
        aqq = working[pivot_col][pivot_col]
        apq = working[pivot_row][pivot_col]
        tau = (aqq - app) / (2.0 * apq)
        t = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
        c = 1.0 / sqrt(1.0 + t * t)
        s = t * c

        for index in range(size):
            if index == pivot_row or index == pivot_col:
                continue
            aip = working[pivot_row][index]
            aiq = working[pivot_col][index]
            working[pivot_row][index] = (c * aip) - (s * aiq)
            working[index][pivot_row] = working[pivot_row][index]
            working[pivot_col][index] = (s * aip) + (c * aiq)
            working[index][pivot_col] = working[pivot_col][index]

        working[pivot_row][pivot_row] = (c * c * app) - (2.0 * s * c * apq) + (s * s * aqq)
        working[pivot_col][pivot_col] = (s * s * app) + (2.0 * s * c * apq) + (c * c * aqq)
        working[pivot_row][pivot_col] = 0.0
        working[pivot_col][pivot_row] = 0.0

        for index in range(size):
            vip = eigenvectors[index][pivot_row]
            viq = eigenvectors[index][pivot_col]
            eigenvectors[index][pivot_row] = (c * vip) - (s * viq)
            eigenvectors[index][pivot_col] = (s * vip) + (c * viq)

    ordering = sorted(range(size), key=lambda index: working[index][index])
    return _EigenResult(
        eigenvalues=[working[index][index] for index in ordering],
        eigenvectors=[
            [eigenvectors[row][index] for index in ordering]
            for row in range(size)
        ],
    )


class DenseModalSolver:
    def solve(self, request: ModalAnalysisRequest) -> list[ModeResult]:
        system = GeneralizedEigenSystem.from_request(request)
        inverse_lower = _inverse_lower(_cholesky_lower(system.mass_matrix))
        transformed = _multiply(
            inverse_lower,
            _multiply(system.stiffness_matrix, _transpose(inverse_lower)),
        )
        eigen = _jacobi_eigendecomposition(transformed)

        results: list[ModeResult] = []
        for ordered_index, eigenvalue in enumerate(eigen.eigenvalues):
            if eigenvalue <= 1.0e-12:
                continue

            reduced_vector = [
                eigen.eigenvectors[row][ordered_index] for row in range(len(system.dof_labels))
            ]
            vector = _matrix_vector_multiply(_transpose(inverse_lower), reduced_vector)
            modal_mass = _dot(vector, _matrix_vector_multiply(system.mass_matrix, vector))
            frequency_hz = sqrt(eigenvalue) / (2.0 * pi)
            results.append(
                ModeResult(
                    mode_index=len(results) + 1,
                    eigenvalue=eigenvalue,
                    frequency_hz=frequency_hz,
                    modal_mass=modal_mass,
                    dof_labels=system.dof_labels,
                    mode_shape=vector,
                )
            )
            if len(results) >= request.num_modes:
                break

        if len(results) < request.num_modes:
            raise RuntimeError("Unable to extract the requested number of physical modes")
        return results


class SLEPcModalSolver:
    def solve(self, request: ModalAnalysisRequest) -> list[ModeResult]:
        if not slepc_available():
            raise RuntimeError(
                "SLEPc backend is not available. Install the conda environment from "
                "config/fenicsx_pinn_environment.yml before enabling the Python modal path."
            )

        from petsc4py import PETSc
        from slepc4py import SLEPc

        system = GeneralizedEigenSystem.from_request(request)
        size = len(system.stiffness_matrix)

        stiffness = PETSc.Mat().createDense(size=(size, size))
        stiffness.setUp()
        for row in range(size):
            stiffness.setValues(row, list(range(size)), system.stiffness_matrix[row])
        stiffness.assemble()

        mass = PETSc.Mat().createDense(size=(size, size))
        mass.setUp()
        for row in range(size):
            mass.setValues(row, list(range(size)), system.mass_matrix[row])
        mass.assemble()

        solver = SLEPc.EPS().create()
        solver.setOperators(stiffness, mass)
        solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        solver.setDimensions(request.num_modes, PETSc.DECIDE)
        solver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        solver.solve()

        converged = solver.getConverged()
        if converged < request.num_modes:
            raise RuntimeError(
                f"SLEPc converged only {converged} eigenpairs, requested {request.num_modes}"
            )

        results: list[ModeResult] = []
        eigenvector_real = stiffness.createVecRight()
        eigenvector_imag = stiffness.createVecRight()
        for mode_index in range(request.num_modes):
            eigenvalue_raw = solver.getEigenpair(mode_index, eigenvector_real, eigenvector_imag)
            eigenvalue = float(eigenvalue_raw.real if hasattr(eigenvalue_raw, "real") else eigenvalue_raw)
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

        if len(results) < request.num_modes:
            raise RuntimeError("SLEPc returned insufficient physical modes")
        return results
