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
        swing = operator_bundle.fruit_dofs.get(fruit.fruit_id)
        if swing is None:
            continue
        # Fruit = 2-DOF horizontal pendulum (x-, y-swing). Gravity is vertical, so
        # the horizontal swing DOFs carry only the (near-zero) horizontal gravity
        # component; the mg/L pendulum restoring is already in the fruit stiffness.
        for fruit_dof, component_index in zip(swing, (0, 1)):
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


def _dot(left: Vec3, right: Vec3) -> float:
    return (left.x * right.x) + (left.y * right.y) + (left.z * right.z)


def _sum_vectors(vectors: list[Vec3]) -> Vec3:
    total = Vec3(0.0, 0.0, 0.0)
    for vector in vectors:
        total = total + vector
    return total


def _branch_station_of_point(branch, point: Vec3) -> float:
    points = branch.path.points()
    total_length = branch.path.length()
    if total_length <= 1.0e-12:
        return 0.0

    best_distance_squared = float("inf")
    best_length = 0.0
    traversed = 0.0
    for index in range(len(points) - 1):
        first = points[index]
        second = points[index + 1]
        segment = second - first
        segment_length_squared = _dot(segment, segment)
        if segment_length_squared <= 1.0e-24:
            continue
        alpha = _dot(point - first, segment) / segment_length_squared
        alpha = max(0.0, min(1.0, alpha))
        closest = first + segment.scale(alpha)
        offset = point - closest
        distance_squared = _dot(offset, offset)
        if distance_squared < best_distance_squared:
            best_distance_squared = distance_squared
            best_length = traversed + sqrt(segment_length_squared) * alpha
        traversed += sqrt(segment_length_squared)

    return max(0.0, min(1.0, best_length / total_length))


def _element_tangent_and_length(branch, local_index: int, num_elements: int) -> tuple[Vec3, float]:
    start_point = branch.path.point_at(local_index / num_elements)
    end_point = branch.path.point_at((local_index + 1) / num_elements)
    tangent = Vec3(
        end_point.x - start_point.x,
        end_point.y - start_point.y,
        end_point.z - start_point.z,
    )
    length = sqrt((tangent.x**2) + (tangent.y**2) + (tangent.z**2))
    if length <= 1.0e-12:
        raise ValueError(
            f"Degenerate branch element on '{branch.branch_id}' at index {local_index}"
        )
    return Vec3(tangent.x / length, tangent.y / length, tangent.z / length), length


def _build_prestress_axial_forces(
    model: OrchardModel,
    cell_data: EmbeddedBeamCellData,
    mesh_spec: EmbeddedLineMeshSpec,
) -> dict[str, list[float]]:
    gravity_direction = _normalize_gravity_direction(model.analysis.gravity_direction)
    gravity_force_per_mass = gravity_direction.scale(9.81)

    def mass_force(mass: float) -> Vec3:
        return gravity_force_per_mass.scale(max(float(mass), 0.0))

    children_by_parent: dict[str, list[str]] = {}
    for branch in model.branches:
        if branch.parent_branch_id is not None:
            children_by_parent.setdefault(branch.parent_branch_id, []).append(branch.branch_id)

    element_forces: dict[str, list[Vec3]] = {}
    for branch in model.branches:
        cell_indices = mesh_spec.cells_for_branch(branch.branch_id)
        num_elements = max(branch.discretization.num_elements, 1)
        forces: list[Vec3] = []
        for local_index, cell_index in enumerate(cell_indices):
            _, length = _element_tangent_and_length(branch, local_index, num_elements)
            forces.append(mass_force(cell_data.mass_per_length[cell_index] * length))
        element_forces[branch.branch_id] = forces

    subtree_force_cache: dict[str, Vec3] = {}

    def subtree_force(branch_id: str) -> Vec3:
        cached = subtree_force_cache.get(branch_id)
        if cached is not None:
            return cached
        total = _sum_vectors(element_forces[branch_id])
        for child_branch_id in children_by_parent.get(branch_id, []):
            total = total + subtree_force(child_branch_id)
        subtree_force_cache[branch_id] = total
        return total

    child_attachment_station: dict[tuple[str, str], float] = {}
    for branch in model.branches:
        if branch.parent_branch_id is None:
            continue
        parent = model.require_branch(branch.parent_branch_id)
        child_attachment_station[(parent.branch_id, branch.branch_id)] = _branch_station_of_point(
            parent,
            branch.path.point_at(0.0),
        )

    axial_forces: dict[str, list[float]] = {}

    for branch in model.branches:
        num_elements = max(branch.discretization.num_elements, 1)
        branch_forces: list[float] = []
        for local_index in range(num_elements):
            midpoint_station = (local_index + 0.5) / num_elements
            tangent, _ = _element_tangent_and_length(branch, local_index, num_elements)
            distal_load = element_forces[branch.branch_id][local_index].scale(0.5)
            distal_load = distal_load + _sum_vectors(
                element_forces[branch.branch_id][local_index + 1 :]
            )
            for child_branch_id in children_by_parent.get(branch.branch_id, []):
                station = child_attachment_station[(branch.branch_id, child_branch_id)]
                if station >= midpoint_station:
                    distal_load = distal_load + subtree_force(child_branch_id)
            branch_forces.append(_dot(distal_load, tangent))
        axial_forces[branch.branch_id] = branch_forces
    return axial_forces


def _build_geometric_stiffness_matrix(
    model: OrchardModel,
    operator_bundle: EmbeddedBeamOperatorBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    branch_node_dofs: BranchNodeDofMap,
    axial_forces: dict[str, list[float]],
) -> Any:
    fixed_dofs: set[int] = set()
    for clamp in model.clamps:
        fixed_dofs.update(int(dof) for dof in branch_node_dofs[clamp.branch_id][0])

    mpc = operator_bundle.mpc
    if mpc is None:
        slave_dofs: frozenset[int] = frozenset()
        mpc_coefficients = None
        mpc_offsets = None
    else:
        slave_dofs = frozenset(int(slave) for slave in mpc.slaves)
        mpc_coefficients, mpc_offsets = mpc.coefficients()

    def constrained_terms(dof: int) -> tuple[tuple[int, float], ...]:
        if int(dof) in fixed_dofs:
            return ()
        if mpc is None or dof not in slave_dofs:
            return ((int(dof), 1.0),)

        assert mpc_coefficients is not None
        assert mpc_offsets is not None
        masters = mpc.masters.links(int(dof))
        coefficients = mpc_coefficients[mpc_offsets[dof] : mpc_offsets[dof + 1]]
        return tuple(
            (int(master), float(coefficient))
            for master, coefficient in zip(masters, coefficients)
        )

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
            local_geometric = build_local_geometric_stiffness_matrix(axial_force, length)
            global_geometric = transform_to_global(local_geometric, transformation)
            element_dofs = list(node_dofs[local_index]) + list(node_dofs[local_index + 1])
            for local_row, global_row in enumerate(element_dofs):
                row_values = global_geometric[local_row]
                row_terms = constrained_terms(global_row)
                for local_col, value in enumerate(row_values):
                    if abs(value) <= 1.0e-14:
                        continue
                    column_terms = constrained_terms(element_dofs[local_col])
                    for resolved_row, row_coefficient in row_terms:
                        for resolved_column, column_coefficient in column_terms:
                            accumulate_owned_matrix_value(
                                geometric_matrix,
                                owned_rows,
                                resolved_row,
                                resolved_column,
                                float(value) * row_coefficient * column_coefficient,
                            )

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
        gravity_static_displacement=static_displacement,
        geometric_stiffness_matrix=geometric_stiffness_matrix,
        prestress_axial_forces=prestress_axial_forces,
    )
