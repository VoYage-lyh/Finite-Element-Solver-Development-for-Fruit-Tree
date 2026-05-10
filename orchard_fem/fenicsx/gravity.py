from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Any

from orchard_fem.discretization.beam.element_matrices import transform_to_global
from orchard_fem.discretization.beam.local_matrices import build_local_geometric_stiffness_matrix
from orchard_fem.discretization.beam.transforms import build_transformation_matrix
from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.beam_forms import EmbeddedBeamCellData
from orchard_fem.fenicsx.branch_dofs import BranchNodeDofMap, resolve_branch_node_dofs
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle
from orchard_fem.fenicsx.fruits import fruit_component_index
from orchard_fem.fenicsx.mpc import apply_mpc_to_vector_in_place
from orchard_fem.fenicsx.operator_bundle import EmbeddedBeamOperatorBundle
from orchard_fem.fenicsx.petsc_ops import (
    accumulate_owned_matrix_value,
    accumulate_owned_vector_value,
    add_matrix_copy,
    create_empty_aij_matrix_like,
)
from orchard_fem.topology import Vec3


def _normalize_gravity_direction(direction: tuple[float, float, float]) -> Vec3:
    magnitude = sqrt((direction[0] ** 2) + (direction[1] ** 2) + (direction[2] ** 2))
    if magnitude <= 1.0e-14:
        raise ValueError("gravity_direction must have non-zero magnitude")
    return Vec3(
        direction[0] / magnitude,
        direction[1] / magnitude,
        direction[2] / magnitude,
    )


def _build_gravity_load_vector(
    model: OrchardModel,
    operator_bundle: EmbeddedBeamOperatorBundle,
    cell_data: EmbeddedBeamCellData,
    mesh_spec: EmbeddedLineMeshSpec,
    branch_node_dofs: BranchNodeDofMap,
) -> Any:
    gravity_direction = _normalize_gravity_direction(model.analysis.gravity_direction)
    gravity_scale = 9.81
    load_vector = operator_bundle.stiffness_matrix.createVecRight()
    load_vector.set(0.0)
    owned_rows = load_vector.getOwnershipRange()

    for branch in model.branches:
        cell_indices = mesh_spec.cells_for_branch(branch.branch_id)
        node_dofs = branch_node_dofs[branch.branch_id]
        num_elements = max(branch.discretization.num_elements, 1)
        for local_index, cell_index in enumerate(cell_indices):
            start_point = branch.path.point_at(local_index / num_elements)
            end_point = branch.path.point_at((local_index + 1) / num_elements)
            length = sqrt(
                ((end_point.x - start_point.x) ** 2)
                + ((end_point.y - start_point.y) ** 2)
                + ((end_point.z - start_point.z) ** 2)
            )
            nodal_scale = (
                0.5 * cell_data.mass_per_length[cell_index] * gravity_scale * length
            )
            nodal_force = (
                gravity_direction.x * nodal_scale,
                gravity_direction.y * nodal_scale,
                gravity_direction.z * nodal_scale,
            )
            for node_index in (local_index, local_index + 1):
                dofs = node_dofs[node_index]
                accumulate_owned_vector_value(load_vector, owned_rows, dofs[0], nodal_force[0])
                accumulate_owned_vector_value(load_vector, owned_rows, dofs[1], nodal_force[1])
                accumulate_owned_vector_value(load_vector, owned_rows, dofs[2], nodal_force[2])

    gravity_components = (
        gravity_direction.x,
        gravity_direction.y,
        gravity_direction.z,
        0.0,
        0.0,
        0.0,
    )
    for fruit in model.fruits:
        fruit_dof = operator_bundle.fruit_dofs.get(fruit.fruit_id)
        if fruit_dof is None:
            continue
        component_index = fruit_component_index(fruit)
        accumulate_owned_vector_value(
            load_vector,
            owned_rows,
            fruit_dof,
            max(fruit.mass, 0.0) * gravity_scale * gravity_components[component_index],
        )

    load_vector.assemblyBegin()
    load_vector.assemblyEnd()
    return load_vector


def _zero_clamped_gravity_loads(
    model: OrchardModel,
    gravity_load_vector: Any,
    branch_node_dofs: BranchNodeDofMap,
) -> None:
    from petsc4py import PETSc

    owned_rows = gravity_load_vector.getOwnershipRange()
    for clamp in model.clamps:
        for dof in branch_node_dofs[clamp.branch_id][0]:
            ownership_start, ownership_end = owned_rows
            if not (ownership_start <= dof < ownership_end):
                continue
            gravity_load_vector.setValue(
                dof,
                0.0,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
    gravity_load_vector.assemblyBegin()
    gravity_load_vector.assemblyEnd()


def _solve_static_prestress_displacement(
    stiffness_matrix: Any,
    gravity_load_vector: Any,
) -> Any:
    from petsc4py import PETSc

    solution = gravity_load_vector.duplicate()
    solver = PETSc.KSP().create(stiffness_matrix.getComm())
    solver.setOperators(stiffness_matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setFromOptions()
    solver.solve(gravity_load_vector, solution)

    if solver.getConvergedReason() <= 0:
        raise RuntimeError(
            f"PETSc KSP failed to converge for FEniCSx gravity prestress "
            f"(reason={solver.getConvergedReason()})."
        )
    return solution


def _build_prestress_axial_forces(
    model: OrchardModel,
    cell_data: EmbeddedBeamCellData,
    mesh_spec: EmbeddedLineMeshSpec,
    branch_node_dofs: BranchNodeDofMap,
    static_displacement: Any,
) -> dict[str, list[float]]:
    displacement_values = static_displacement.getArray(readonly=True)
    axial_forces: dict[str, list[float]] = {}

    for branch in model.branches:
        cell_indices = mesh_spec.cells_for_branch(branch.branch_id)
        node_dofs = branch_node_dofs[branch.branch_id]
        num_elements = max(branch.discretization.num_elements, 1)
        branch_forces: list[float] = []
        for local_index, cell_index in enumerate(cell_indices):
            start_point = branch.path.point_at(local_index / num_elements)
            end_point = branch.path.point_at((local_index + 1) / num_elements)
            tangent = Vec3(
                end_point.x - start_point.x,
                end_point.y - start_point.y,
                end_point.z - start_point.z,
            )
            length = sqrt(
                (tangent.x**2) + (tangent.y**2) + (tangent.z**2)
            )
            tangent = Vec3(tangent.x / length, tangent.y / length, tangent.z / length)

            first_dofs = node_dofs[local_index]
            second_dofs = node_dofs[local_index + 1]
            first_axial = (
                displacement_values[first_dofs[0]] * tangent.x
                + displacement_values[first_dofs[1]] * tangent.y
                + displacement_values[first_dofs[2]] * tangent.z
            )
            second_axial = (
                displacement_values[second_dofs[0]] * tangent.x
                + displacement_values[second_dofs[1]] * tangent.y
                + displacement_values[second_dofs[2]] * tangent.z
            )
            branch_forces.append(
                cell_data.axial_rigidity[cell_index]
                * ((second_axial - first_axial) / length)
            )
        axial_forces[branch.branch_id] = branch_forces
    return axial_forces


def _build_geometric_stiffness_matrix(
    model: OrchardModel,
    operator_bundle: EmbeddedBeamOperatorBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    branch_node_dofs: BranchNodeDofMap,
    axial_forces: dict[str, list[float]],
) -> Any:
    geometric_matrix = create_empty_aij_matrix_like(
        operator_bundle.stiffness_matrix,
        operator_bundle.stiffness_matrix.getSize()[0],
    )
    owned_rows = geometric_matrix.getOwnershipRange()

    for branch in model.branches:
        node_dofs = branch_node_dofs[branch.branch_id]
        num_elements = max(branch.discretization.num_elements, 1)
        for local_index, axial_force in enumerate(axial_forces[branch.branch_id]):
            start_point = branch.path.point_at(local_index / num_elements)
            end_point = branch.path.point_at((local_index + 1) / num_elements)
            length = sqrt(
                ((end_point.x - start_point.x) ** 2)
                + ((end_point.y - start_point.y) ** 2)
                + ((end_point.z - start_point.z) ** 2)
            )
            transformation = build_transformation_matrix(start_point, end_point)
            # FEniCSx static preload reports compression with the opposite sign
            # convention from the legacy 12x12 geometric-stiffness helper.
            local_geometric = build_local_geometric_stiffness_matrix(-axial_force, length)
            global_geometric = transform_to_global(local_geometric, transformation)
            element_dofs = list(node_dofs[local_index]) + list(node_dofs[local_index + 1])
            for local_row, global_row in enumerate(element_dofs):
                row_values = global_geometric[local_row]
                ownership_start, ownership_end = owned_rows
                if not (ownership_start <= global_row < ownership_end):
                    continue
                columns = []
                values = []
                for local_col, value in enumerate(row_values):
                    if abs(value) <= 1.0e-14:
                        continue
                    columns.append(element_dofs[local_col])
                    values.append(float(value))
                if columns:
                    geometric_matrix.setValues(global_row, columns, values)

    geometric_matrix.assemble()
    return geometric_matrix


def augment_operators_with_gravity_prestress(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    cell_data: EmbeddedBeamCellData,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    if not model.analysis.include_gravity_prestress:
        return operator_bundle

    branch_node_dofs = resolve_branch_node_dofs(model, space_bundle, mesh_spec)
    gravity_load_vector = _build_gravity_load_vector(
        model,
        operator_bundle,
        cell_data,
        mesh_spec,
        branch_node_dofs,
    )
    _zero_clamped_gravity_loads(model, gravity_load_vector, branch_node_dofs)
    apply_mpc_to_vector_in_place(operator_bundle.mpc, gravity_load_vector)
    static_displacement = _solve_static_prestress_displacement(
        operator_bundle.stiffness_matrix,
        gravity_load_vector,
    )
    if operator_bundle.mpc is not None:
        operator_bundle.mpc.backsubstitution(static_displacement)
    prestress_axial_forces = _build_prestress_axial_forces(
        model,
        cell_data,
        mesh_spec,
        branch_node_dofs,
        static_displacement,
    )
    geometric_stiffness_matrix = _build_geometric_stiffness_matrix(
        model,
        operator_bundle,
        mesh_spec,
        branch_node_dofs,
        prestress_axial_forces,
    )
    stiffness_matrix = add_matrix_copy(
        operator_bundle.stiffness_matrix,
        geometric_stiffness_matrix,
    )

    return replace(
        operator_bundle,
        stiffness_matrix=stiffness_matrix,
        gravity_load_vector=gravity_load_vector,
        geometric_stiffness_matrix=geometric_stiffness_matrix,
        prestress_axial_forces=prestress_axial_forces,
    )
