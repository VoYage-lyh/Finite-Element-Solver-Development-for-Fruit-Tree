from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Sequence

from orchard_fem.discretization.beam.element_matrices import transform_to_global
from orchard_fem.discretization.beam.local_matrices import build_local_geometric_stiffness_matrix
from orchard_fem.discretization.beam.transforms import build_transformation_matrix
from orchard_fem.discretization.types import (
    COMPONENT_LABELS,
    NonlinearLinkDefinition,
    NonlinearLinkKind,
)
from orchard_fem.domain import JointDefinition, JointLawKind, OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.boundary_conditions import build_model_clamp_boundary_conditions
from orchard_fem.fenicsx.beam_forms import (
    EmbeddedBeamCellData,
    EmbeddedBeamCoefficientFunctions,
    EmbeddedBeamFormBundle,
    build_embedded_beam_cell_data,
    build_embedded_timoshenko_forms,
    create_embedded_beam_coefficient_functions,
)
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec, build_embedded_line_mesh_spec
from orchard_fem.fenicsx.dofs import resolve_embedded_beam_component_dof
from orchard_fem.fenicsx.fields import (
    EmbeddedBeamFunctionSpaceBundle,
    create_embedded_beam_function_space,
)
from orchard_fem.topology import Vec3

CONSTRAINT_PENALTY = 1.0e12


@dataclass(frozen=True)
class EmbeddedBeamOperatorBundle:
    compiled_stiffness_form: Any
    compiled_mass_form: Any
    stiffness_matrix: Any
    mass_matrix: Any
    attachment_damping_matrix: Any | None = None
    gravity_load_vector: Any | None = None
    geometric_stiffness_matrix: Any | None = None
    prestress_axial_forces: dict[str, list[float]] = field(default_factory=dict)
    fruit_dofs: dict[str, int] = field(default_factory=dict)
    nonlinear_links: list[NonlinearLinkDefinition] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddedBeamExperimentBundle:
    mesh_spec: EmbeddedLineMeshSpec
    space_bundle: EmbeddedBeamFunctionSpaceBundle
    cell_data: EmbeddedBeamCellData
    coefficient_functions: EmbeddedBeamCoefficientFunctions
    form_bundle: EmbeddedBeamFormBundle
    operator_bundle: EmbeddedBeamOperatorBundle
    fruit_dofs: dict[str, int]


def _create_empty_aij_matrix_like(matrix: Any, size: int) -> Any:
    from petsc4py import PETSc

    created = PETSc.Mat().createAIJ(size=(size, size), comm=matrix.getComm())
    created.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    created.setUp()
    return created


def _allow_new_nonzero_allocation(matrix: Any) -> None:
    from petsc4py import PETSc

    matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)


def _copy_matrix_entries(source: Any, target: Any) -> None:
    from petsc4py import PETSc

    ownership_start, ownership_end = source.getOwnershipRange()
    for row_index in range(ownership_start, ownership_end):
        columns, values = source.getRow(row_index)
        try:
            if len(columns) > 0:
                target.setValues(
                    row_index,
                    columns,
                    values,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )
        finally:
            restore_row = getattr(source, "restoreRow", None)
            if restore_row is not None:
                restore_row(row_index, columns, values)


def _accumulate_owned_matrix_value(
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
        row,
        column,
        value,
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def _nearest_fruit_anchor_point(model: OrchardModel, fruit) -> tuple[float, float, float]:
    branch = model.require_branch(fruit.branch_id)
    num_elements = max(branch.discretization.num_elements, 1)
    node_index = min(
        range(num_elements + 1),
        key=lambda candidate: abs((candidate / num_elements) - fruit.location_s),
    )
    station = node_index / num_elements
    point = branch.path.point_at(station)
    return point.x, point.y, point.z


def _nearest_parent_node_index(model: OrchardModel, child_branch_id: str, parent_branch_id: str) -> int:
    child_branch = model.require_branch(child_branch_id)
    parent_branch = model.require_branch(parent_branch_id)
    parent_elements = max(parent_branch.discretization.num_elements, 1)
    child_root = child_branch.path.point_at(0.0)

    return min(
        range(parent_elements + 1),
        key=lambda candidate: (
            (parent_branch.path.point_at(candidate / parent_elements).x - child_root.x) ** 2
            + (parent_branch.path.point_at(candidate / parent_elements).y - child_root.y) ** 2
            + (parent_branch.path.point_at(candidate / parent_elements).z - child_root.z) ** 2
        ),
    )


def _joint_component_penalty(
    component_index: int,
    joint: JointDefinition | None,
) -> float:
    penalty = CONSTRAINT_PENALTY
    if joint is None:
        return penalty

    penalty *= max(joint.linear_stiffness_scale, 1.0e-6)
    if component_index >= 3 and joint.law.kind != JointLawKind.NONE:
        penalty *= max(joint.law.linear_scale, 1.0e-6)
    return penalty


def _append_joint_nonlinear_links(
    nonlinear_links: list[NonlinearLinkDefinition],
    joint: JointDefinition | None,
    child_root: tuple[int, int, int, int, int, int],
    nearest_parent: tuple[int, int, int, int, int, int],
) -> None:
    if joint is None or joint.law.kind == JointLawKind.NONE:
        return

    rotational_linear_stiffness = _joint_component_penalty(3, joint)
    rotational_open_stiffness = (
        CONSTRAINT_PENALTY
        * max(joint.linear_stiffness_scale, 1.0e-6)
        * max(joint.law.open_scale, 0.0)
    )

    for component_index in range(3, 6):
        component = COMPONENT_LABELS[component_index]
        label = f"joint:{joint.joint_id}:{component}"

        if joint.law.kind == JointLawKind.POLYNOMIAL:
            if abs(joint.law.cubic_scale) <= 0.0:
                continue
            nonlinear_links.append(
                NonlinearLinkDefinition(
                    label=label,
                    first_dof=child_root[component_index],
                    second_dof=nearest_parent[component_index],
                    kind=NonlinearLinkKind.CUBIC_SPRING,
                    cubic_stiffness=joint.law.cubic_scale,
                )
            )
            continue

        if joint.law.kind == JointLawKind.GAP_FRICTION:
            nonlinear_links.append(
                NonlinearLinkDefinition(
                    label=label,
                    first_dof=child_root[component_index],
                    second_dof=nearest_parent[component_index],
                    kind=NonlinearLinkKind.GAP_SPRING,
                    linear_stiffness=rotational_linear_stiffness,
                    open_stiffness=rotational_open_stiffness,
                    gap_threshold=max(joint.law.gap_threshold, 0.0),
                )
            )
            continue

        raise ValueError(f"Unsupported joint law kind: {joint.law.kind}")


def _normalize_gravity_direction(direction: tuple[float, float, float]) -> Vec3:
    magnitude = sqrt((direction[0] ** 2) + (direction[1] ** 2) + (direction[2] ** 2))
    if magnitude <= 1.0e-14:
        raise ValueError("gravity_direction must have non-zero magnitude")
    return Vec3(
        direction[0] / magnitude,
        direction[1] / magnitude,
        direction[2] / magnitude,
    )


def _resolve_branch_node_dofs(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
) -> dict[str, list[tuple[int, int, int, int, int, int]]]:
    components = ("ux", "uy", "uz", "rx", "ry", "rz")
    branch_node_dofs: dict[str, list[tuple[int, int, int, int, int, int]]] = {}

    for branch in model.branches:
        num_elements = max(branch.discretization.num_elements, 1)
        nodes: list[tuple[int, int, int, int, int, int]] = []
        for node_index in range(num_elements + 1):
            station = node_index / num_elements
            point = branch.path.point_at(station)
            point_tuple = (point.x, point.y, point.z)
            dofs = tuple(
                resolve_embedded_beam_component_dof(
                    space_bundle,
                    point_tuple,
                    component,
                )
                for component in components
            )
            nodes.append(dofs)  # type: ignore[arg-type]
        branch_node_dofs[branch.branch_id] = nodes
    return branch_node_dofs


def _accumulate_owned_vector_value(
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
        index,
        value,
        addv=PETSc.InsertMode.ADD_VALUES,
    )


def _build_gravity_load_vector(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    operator_bundle: EmbeddedBeamOperatorBundle,
    cell_data: EmbeddedBeamCellData,
    mesh_spec: EmbeddedLineMeshSpec,
    branch_node_dofs: dict[str, list[tuple[int, int, int, int, int, int]]],
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
                _accumulate_owned_vector_value(load_vector, owned_rows, dofs[0], nodal_force[0])
                _accumulate_owned_vector_value(load_vector, owned_rows, dofs[1], nodal_force[1])
                _accumulate_owned_vector_value(load_vector, owned_rows, dofs[2], nodal_force[2])

    load_vector.assemblyBegin()
    load_vector.assemblyEnd()
    return load_vector


def _zero_clamped_gravity_loads(
    model: OrchardModel,
    gravity_load_vector: Any,
    branch_node_dofs: dict[str, list[tuple[int, int, int, int, int, int]]],
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


def _solve_static_prestress_displacement(stiffness_matrix: Any, gravity_load_vector: Any) -> Any:
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
    branch_node_dofs: dict[str, list[tuple[int, int, int, int, int, int]]],
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
    branch_node_dofs: dict[str, list[tuple[int, int, int, int, int, int]]],
    axial_forces: dict[str, list[float]],
) -> Any:
    geometric_matrix = _create_empty_aij_matrix_like(
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


def _augment_operators_with_joints(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    branch_node_dofs = _resolve_branch_node_dofs(model, space_bundle)
    stiffness_matrix = operator_bundle.stiffness_matrix.duplicate(copy=True)
    _allow_new_nonzero_allocation(stiffness_matrix)
    owned_rows = stiffness_matrix.getOwnershipRange()
    nonlinear_links = list(operator_bundle.nonlinear_links)
    has_branch_connections = False

    for branch in model.branches:
        if branch.parent_branch_id is None:
            continue
        has_branch_connections = True

        child_nodes = branch_node_dofs[branch.branch_id]
        child_root = child_nodes[0]
        parent_node_index = _nearest_parent_node_index(
            model,
            branch.branch_id,
            branch.parent_branch_id,
        )
        parent_root = branch_node_dofs[branch.parent_branch_id][parent_node_index]
        joint = model.find_joint_for_child(branch.branch_id)

        for component_index in range(6):
            penalty = _joint_component_penalty(component_index, joint)
            first_dof = child_root[component_index]
            second_dof = parent_root[component_index]
            _accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                first_dof,
                first_dof,
                penalty,
            )
            _accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                second_dof,
                second_dof,
                penalty,
            )
            _accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                first_dof,
                second_dof,
                -penalty,
            )
            _accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                second_dof,
                first_dof,
                -penalty,
            )
        _append_joint_nonlinear_links(
            nonlinear_links,
            joint,
            child_root,
            parent_root,
        )

    if not has_branch_connections:
        return operator_bundle

    stiffness_matrix.assemble()
    return EmbeddedBeamOperatorBundle(
        compiled_stiffness_form=operator_bundle.compiled_stiffness_form,
        compiled_mass_form=operator_bundle.compiled_mass_form,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=operator_bundle.mass_matrix,
        attachment_damping_matrix=operator_bundle.attachment_damping_matrix,
        gravity_load_vector=operator_bundle.gravity_load_vector,
        geometric_stiffness_matrix=operator_bundle.geometric_stiffness_matrix,
        prestress_axial_forces=operator_bundle.prestress_axial_forces,
        fruit_dofs=operator_bundle.fruit_dofs,
        nonlinear_links=nonlinear_links,
    )


def _augment_operators_with_auto_nonlinear_links(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    auto_nonlinear_levels = set(model.analysis.auto_nonlinear_levels)
    if not auto_nonlinear_levels:
        return operator_bundle

    branch_node_dofs = _resolve_branch_node_dofs(model, space_bundle)
    explicit_joint_children = {joint.child_branch_id for joint in model.joints}
    nonlinear_links = list(operator_bundle.nonlinear_links)

    for branch in model.branches:
        if branch.parent_branch_id is None:
            continue
        if branch.level not in auto_nonlinear_levels:
            continue
        if branch.branch_id in explicit_joint_children:
            continue

        parent_node_index = _nearest_parent_node_index(
            model,
            branch.branch_id,
            branch.parent_branch_id,
        )
        nonlinear_links.append(
            NonlinearLinkDefinition(
                label=f"auto_joint:{branch.branch_id}",
                first_dof=branch_node_dofs[branch.branch_id][0][0],
                second_dof=branch_node_dofs[branch.parent_branch_id][parent_node_index][0],
                kind=NonlinearLinkKind.CUBIC_SPRING,
                cubic_stiffness=model.analysis.auto_nonlinear_cubic_scale,
            )
        )

    return EmbeddedBeamOperatorBundle(
        compiled_stiffness_form=operator_bundle.compiled_stiffness_form,
        compiled_mass_form=operator_bundle.compiled_mass_form,
        stiffness_matrix=operator_bundle.stiffness_matrix,
        mass_matrix=operator_bundle.mass_matrix,
        attachment_damping_matrix=operator_bundle.attachment_damping_matrix,
        gravity_load_vector=operator_bundle.gravity_load_vector,
        geometric_stiffness_matrix=operator_bundle.geometric_stiffness_matrix,
        prestress_axial_forces=operator_bundle.prestress_axial_forces,
        fruit_dofs=operator_bundle.fruit_dofs,
        nonlinear_links=nonlinear_links,
    )


def _augment_operators_with_nonlinear_clamps(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    nonlinear_clamps = [
        clamp for clamp in model.clamps if abs(clamp.cubic_stiffness) > 1.0e-14
    ]
    if not nonlinear_clamps:
        return operator_bundle

    branch_node_dofs = _resolve_branch_node_dofs(model, space_bundle)
    stiffness_matrix = operator_bundle.stiffness_matrix.duplicate(copy=True)
    _allow_new_nonzero_allocation(stiffness_matrix)
    owned_rows = stiffness_matrix.getOwnershipRange()
    nonlinear_links = list(operator_bundle.nonlinear_links)

    for clamp in nonlinear_clamps:
        root_dofs = branch_node_dofs[clamp.branch_id][0]
        for dof in root_dofs:
            _accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                dof,
                dof,
                CONSTRAINT_PENALTY,
            )
        nonlinear_links.append(
            NonlinearLinkDefinition(
                label=f"clamp:{clamp.branch_id}",
                first_dof=root_dofs[0],
                second_dof=-1,
                kind=NonlinearLinkKind.CUBIC_SPRING,
                linear_stiffness=CONSTRAINT_PENALTY,
                cubic_stiffness=clamp.cubic_stiffness,
            )
        )

    stiffness_matrix.assemble()
    return EmbeddedBeamOperatorBundle(
        compiled_stiffness_form=operator_bundle.compiled_stiffness_form,
        compiled_mass_form=operator_bundle.compiled_mass_form,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=operator_bundle.mass_matrix,
        attachment_damping_matrix=operator_bundle.attachment_damping_matrix,
        gravity_load_vector=operator_bundle.gravity_load_vector,
        geometric_stiffness_matrix=operator_bundle.geometric_stiffness_matrix,
        prestress_axial_forces=operator_bundle.prestress_axial_forces,
        fruit_dofs=operator_bundle.fruit_dofs,
        nonlinear_links=nonlinear_links,
    )


def _add_matrix_in_place(target: Any, increment: Any) -> Any:
    from petsc4py import PETSc

    updated = target.duplicate(copy=True)
    updated.axpy(
        1.0,
        increment,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    updated.assemble()
    return updated


def _augment_operators_with_gravity_prestress(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    cell_data: EmbeddedBeamCellData,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    if not model.analysis.include_gravity_prestress:
        return operator_bundle

    branch_node_dofs = _resolve_branch_node_dofs(model, space_bundle)
    gravity_load_vector = _build_gravity_load_vector(
        model,
        space_bundle,
        operator_bundle,
        cell_data,
        mesh_spec,
        branch_node_dofs,
    )
    _zero_clamped_gravity_loads(model, gravity_load_vector, branch_node_dofs)
    static_displacement = _solve_static_prestress_displacement(
        operator_bundle.stiffness_matrix,
        gravity_load_vector,
    )
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
    stiffness_matrix = _add_matrix_in_place(
        operator_bundle.stiffness_matrix,
        geometric_stiffness_matrix,
    )

    return EmbeddedBeamOperatorBundle(
        compiled_stiffness_form=operator_bundle.compiled_stiffness_form,
        compiled_mass_form=operator_bundle.compiled_mass_form,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=operator_bundle.mass_matrix,
        attachment_damping_matrix=operator_bundle.attachment_damping_matrix,
        gravity_load_vector=gravity_load_vector,
        geometric_stiffness_matrix=geometric_stiffness_matrix,
        prestress_axial_forces=prestress_axial_forces,
        fruit_dofs=operator_bundle.fruit_dofs,
        nonlinear_links=operator_bundle.nonlinear_links,
    )


def _augment_operators_with_fruits(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    if not model.fruits:
        return operator_bundle

    base_size = operator_bundle.stiffness_matrix.getSize()[0]
    total_size = base_size + len(model.fruits)
    stiffness_matrix = _create_empty_aij_matrix_like(
        operator_bundle.stiffness_matrix,
        total_size,
    )
    mass_matrix = _create_empty_aij_matrix_like(
        operator_bundle.mass_matrix,
        total_size,
    )
    damping_matrix = _create_empty_aij_matrix_like(
        operator_bundle.stiffness_matrix,
        total_size,
    )

    _copy_matrix_entries(operator_bundle.stiffness_matrix, stiffness_matrix)
    _copy_matrix_entries(operator_bundle.mass_matrix, mass_matrix)

    stiffness_owned_rows = stiffness_matrix.getOwnershipRange()
    mass_owned_rows = mass_matrix.getOwnershipRange()
    damping_owned_rows = damping_matrix.getOwnershipRange()

    fruit_dofs: dict[str, int] = {}
    for fruit_index, fruit in enumerate(model.fruits):
        fruit_dof = base_size + fruit_index
        fruit_dofs[fruit.fruit_id] = fruit_dof

        anchor_point = _nearest_fruit_anchor_point(model, fruit)
        coupled_branch_dof = resolve_embedded_beam_component_dof(
            space_bundle,
            anchor_point,
            "ux",
        )

        _accumulate_owned_matrix_value(
            mass_matrix,
            mass_owned_rows,
            fruit_dof,
            fruit_dof,
            max(fruit.mass, 1.0e-9),
        )

        stiffness_value = max(fruit.stiffness, 1.0e-6)
        _accumulate_owned_matrix_value(
            stiffness_matrix,
            stiffness_owned_rows,
            fruit_dof,
            fruit_dof,
            stiffness_value,
        )
        _accumulate_owned_matrix_value(
            stiffness_matrix,
            stiffness_owned_rows,
            fruit_dof,
            coupled_branch_dof,
            -stiffness_value,
        )
        _accumulate_owned_matrix_value(
            stiffness_matrix,
            stiffness_owned_rows,
            coupled_branch_dof,
            fruit_dof,
            -stiffness_value,
        )
        _accumulate_owned_matrix_value(
            stiffness_matrix,
            stiffness_owned_rows,
            coupled_branch_dof,
            coupled_branch_dof,
            stiffness_value,
        )

        damping_value = max(fruit.damping, 0.0)
        if damping_value > 0.0:
            _accumulate_owned_matrix_value(
                damping_matrix,
                damping_owned_rows,
                fruit_dof,
                fruit_dof,
                damping_value,
            )
            _accumulate_owned_matrix_value(
                damping_matrix,
                damping_owned_rows,
                fruit_dof,
                coupled_branch_dof,
                -damping_value,
            )
            _accumulate_owned_matrix_value(
                damping_matrix,
                damping_owned_rows,
                coupled_branch_dof,
                fruit_dof,
                -damping_value,
            )
            _accumulate_owned_matrix_value(
                damping_matrix,
                damping_owned_rows,
                coupled_branch_dof,
                coupled_branch_dof,
                damping_value,
            )

    stiffness_matrix.assemble()
    mass_matrix.assemble()
    damping_matrix.assemble()

    return EmbeddedBeamOperatorBundle(
        compiled_stiffness_form=operator_bundle.compiled_stiffness_form,
        compiled_mass_form=operator_bundle.compiled_mass_form,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=mass_matrix,
        attachment_damping_matrix=damping_matrix,
        fruit_dofs=fruit_dofs,
        nonlinear_links=operator_bundle.nonlinear_links,
    )


def assemble_embedded_beam_operators(
    form_bundle: EmbeddedBeamFormBundle,
    *,
    bcs: Sequence[Any] = (),
    diag: float = 1.0,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    from dolfinx import fem as dolfinx_fem
    from dolfinx.fem import petsc as dolfinx_fem_petsc

    compiled_stiffness_form = dolfinx_fem.form(form_bundle.stiffness_form)
    compiled_mass_form = dolfinx_fem.form(form_bundle.mass_form)

    stiffness_matrix = dolfinx_fem_petsc.assemble_matrix(
        compiled_stiffness_form,
        bcs=bcs,
        diag=diag,
    )
    stiffness_matrix.assemble()

    mass_matrix = dolfinx_fem_petsc.assemble_matrix(
        compiled_mass_form,
        bcs=bcs,
        diag=0.0,
    )
    mass_matrix.assemble()

    return EmbeddedBeamOperatorBundle(
        compiled_stiffness_form=compiled_stiffness_form,
        compiled_mass_form=compiled_mass_form,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=mass_matrix,
    )


def build_embedded_timoshenko_experiment(
    model: OrchardModel,
    *,
    polynomial_degree: int = 1,
    spec: EmbeddedLineMeshSpec | None = None,
    shear_correction: float = 0.4,
    comm: object | None = None,
    partitioner: object | None = None,
    max_facet_to_cell_links: int = 2,
    bcs: Sequence[Any] | None = None,
    use_model_clamps: bool = False,
    clamp_tolerance: float = 1.0e-8,
    diag: float = 1.0,
) -> EmbeddedBeamExperimentBundle:
    resolved_spec = spec or build_embedded_line_mesh_spec(model)
    space_bundle = create_embedded_beam_function_space(
        resolved_spec,
        polynomial_degree=polynomial_degree,
        comm=comm,
        partitioner=partitioner,
        max_facet_to_cell_links=max_facet_to_cell_links,
    )
    cell_data = build_embedded_beam_cell_data(
        model,
        spec=resolved_spec,
        shear_correction=shear_correction,
    )
    coefficient_functions = create_embedded_beam_coefficient_functions(
        space_bundle.mesh,
        cell_data,
    )
    form_bundle = build_embedded_timoshenko_forms(
        space_bundle,
        coefficient_functions,
    )
    effective_bcs: Sequence[Any]
    if bcs is not None:
        effective_bcs = bcs
    elif use_model_clamps:
        effective_bcs = build_model_clamp_boundary_conditions(
            model,
            space_bundle,
            atol=clamp_tolerance,
            include_nonlinear_clamps=False,
        )
    else:
        effective_bcs = ()
    operator_bundle = assemble_embedded_beam_operators(
        form_bundle,
        bcs=effective_bcs,
        diag=diag,
    )
    operator_bundle = _augment_operators_with_joints(
        model,
        space_bundle,
        operator_bundle,
    )
    operator_bundle = _augment_operators_with_auto_nonlinear_links(
        model,
        space_bundle,
        operator_bundle,
    )
    operator_bundle = _augment_operators_with_nonlinear_clamps(
        model,
        space_bundle,
        operator_bundle,
    )
    operator_bundle = _augment_operators_with_fruits(
        model,
        space_bundle,
        operator_bundle,
    )
    operator_bundle = _augment_operators_with_gravity_prestress(
        model,
        space_bundle,
        resolved_spec,
        cell_data,
        operator_bundle,
    )
    return EmbeddedBeamExperimentBundle(
        mesh_spec=resolved_spec,
        space_bundle=space_bundle,
        cell_data=cell_data,
        coefficient_functions=coefficient_functions,
        form_bundle=form_bundle,
        operator_bundle=operator_bundle,
        fruit_dofs=operator_bundle.fruit_dofs,
    )
