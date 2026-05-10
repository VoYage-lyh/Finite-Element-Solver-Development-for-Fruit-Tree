from __future__ import annotations

from typing import Any

from orchard_fem.numerics import require_petsc


def create_empty_aij_matrix_like(template_matrix: Any, size: int | None = None) -> Any:
    require_petsc()

    from petsc4py import PETSc

    matrix_size = template_matrix.getSize() if size is None else (size, size)
    created = PETSc.Mat().createAIJ(size=matrix_size, comm=template_matrix.getComm())
    created.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    created.setUp()
    return created


def allow_new_nonzero_allocation(matrix: Any) -> None:
    require_petsc()

    from petsc4py import PETSc

    matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)


def copy_owned_matrix_entries(source: Any, target: Any) -> None:
    require_petsc()

    from petsc4py import PETSc

    target_rows, target_columns = target.getSize()
    ownership_start, ownership_end = source.getOwnershipRange()
    for row_index in range(ownership_start, min(ownership_end, target_rows)):
        columns, values = source.getRow(row_index)
        try:
            filtered_columns = []
            filtered_values = []
            for column, value in zip(columns, values):
                if int(column) >= target_columns:
                    continue
                filtered_columns.append(int(column))
                filtered_values.append(float(value))
            if filtered_columns:
                target.setValues(
                    row_index,
                    filtered_columns,
                    filtered_values,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )
        finally:
            restore_row = getattr(source, "restoreRow", None)
            if restore_row is not None:
                restore_row(row_index, columns, values)


def copy_matrix_like(template_matrix: Any, source_matrix: Any) -> Any:
    if template_matrix.getSize() == source_matrix.getSize():
        copied = source_matrix.duplicate(copy=True)
        copied.assemble()
        return copied

    extended = create_empty_aij_matrix_like(template_matrix)
    copy_owned_matrix_entries(source_matrix, extended)
    extended.assemble()
    return extended


def add_matrix_copy(target: Any, increment: Any) -> Any:
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


def add_matrix_in_place(target: Any, increment: Any) -> None:
    require_petsc()

    from petsc4py import PETSc

    target.axpy(
        1.0,
        increment,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    target.assemble()


def accumulate_owned_matrix_value(
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
        int(row),
        int(column),
        float(value),
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def accumulate_owned_vector_value(
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
        int(index),
        float(value),
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def extend_vector_like(template_vector: Any, source_vector: Any) -> Any:
    from petsc4py import PETSc

    extended = template_vector.duplicate()
    extended.set(0.0)
    source_size = source_vector.getSize()
    ownership_start, ownership_end = extended.getOwnershipRange()
    for global_index in range(ownership_start, min(ownership_end, source_size)):
        value = source_vector.getValues([global_index])[0]
        extended.setValue(
            global_index,
            float(value),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    extended.assemblyBegin()
    extended.assemblyEnd()
    return extended
