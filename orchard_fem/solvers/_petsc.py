from __future__ import annotations


def require_petsc() -> None:
    try:
        import petsc4py  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PETSc backend is not available. Install `petsc4py`, `mpi4py`, and the FEniCSx "
            "toolchain by creating the conda environment from "
            f"`config/fenicsx_pinn_environment.yml`. Missing module: {exc.name}."
        ) from exc


def require_slepc() -> None:
    require_petsc()
    try:
        import slepc4py  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SLEPc backend is not available. Install `slepc4py` together with the PETSc/FEniCSx "
            "stack by creating the conda environment from "
            f"`config/fenicsx_pinn_environment.yml`. Missing module: {exc.name}."
        ) from exc


def create_aij_matrix(matrix: list[list[float]], zero_tolerance: float = 1.0e-14):
    from petsc4py import PETSc

    size = len(matrix)
    petsc_matrix = PETSc.Mat().createAIJ(size=(size, size))
    petsc_matrix.setUp()

    for row_index, row in enumerate(matrix):
        if len(row) != size:
            raise ValueError("PETSc matrix assembly requires a square matrix")

        columns: list[int] = []
        values: list[float] = []
        for column_index, value in enumerate(row):
            numeric_value = float(value)
            if abs(numeric_value) <= zero_tolerance:
                continue
            columns.append(column_index)
            values.append(numeric_value)

        if columns:
            petsc_matrix.setValues(row_index, columns, values)

    petsc_matrix.assemble()
    return petsc_matrix


def create_sequential_vector(values: list[float]):
    from petsc4py import PETSc

    vector = PETSc.Vec().createSeq(len(values))
    for index, value in enumerate(values):
        vector.setValue(index, float(value))
    vector.assemblyBegin()
    vector.assemblyEnd()
    return vector


def solve_linear_system(
    matrix,
    rhs: list[float],
    ksp_type: str = "preonly",
    pc_type: str = "lu",
) -> list[float]:
    from petsc4py import PETSc

    rhs_vector = create_sequential_vector(rhs)
    solution = rhs_vector.duplicate()

    solver = PETSc.KSP().create()
    solver.setOperators(matrix)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setFromOptions()
    solver.solve(rhs_vector, solution)

    if solver.getConvergedReason() <= 0:
        raise RuntimeError(f"PETSc KSP failed to converge (reason={solver.getConvergedReason()})")

    return [float(value) for value in solution.getArray(readonly=True)]
