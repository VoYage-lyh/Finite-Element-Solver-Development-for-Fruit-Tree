from __future__ import annotations

from math import fabs

from orchard_fem.discretization.beam_element import (
    BeamElementProperties,
    build_global_element_matrices,
    build_local_geometric_stiffness_matrix,
    build_transformation_matrix,
    transform_to_global,
)
from orchard_fem.discretization.damping import apply_rayleigh_damping
from orchard_fem.discretization.dofs import DOFManager, branch_dof, resolve_node_index
from orchard_fem.discretization.matrix_ops import add_pair_penalty, scatter, zero_matrix
from orchard_fem.discretization.types import (
    COMPONENT_LABELS,
    CONSTRAINT_PENALTY,
    BranchElementState,
    BranchNodeState,
    LinearDynamicAssemblyResult,
    NonlinearLinkDefinition,
    NonlinearLinkKind,
)
from orchard_fem.domain import JointDefinition, JointLawKind, OrchardModel
from orchard_fem.materials.base import build_material_lookup, evaluate_branch_section_state
from orchard_fem.topology import Vec3, distance, normalize


def _nearest_parent_node(
    branch_nodes: dict[str, list[BranchNodeState]],
    child_branch_id: str,
    parent_branch_id: str,
) -> BranchNodeState:
    child_root = branch_nodes[child_branch_id][0]
    return min(
        branch_nodes[parent_branch_id],
        key=lambda node: distance(child_root.position, node.position),
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
    child_root: BranchNodeState,
    nearest_parent: BranchNodeState,
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
                    first_dof=child_root.dofs[component_index],
                    second_dof=nearest_parent.dofs[component_index],
                    kind=NonlinearLinkKind.CUBIC_SPRING,
                    cubic_stiffness=joint.law.cubic_scale,
                )
            )
            continue

        if joint.law.kind == JointLawKind.GAP_FRICTION:
            nonlinear_links.append(
                NonlinearLinkDefinition(
                    label=label,
                    first_dof=child_root.dofs[component_index],
                    second_dof=nearest_parent.dofs[component_index],
                    kind=NonlinearLinkKind.GAP_SPRING,
                    linear_stiffness=rotational_linear_stiffness,
                    open_stiffness=rotational_open_stiffness,
                    gap_threshold=max(joint.law.gap_threshold, 0.0),
                )
            )
            continue

        raise ValueError(f"Unsupported joint law kind: {joint.law.kind}")


class OrchardSystemAssembler:
    def assemble(self, model: OrchardModel) -> LinearDynamicAssemblyResult:
        dof_manager = DOFManager()
        branch_nodes: dict[str, list[BranchNodeState]] = {}
        branch_elements: dict[str, list[BranchElementState]] = {}
        fruit_dofs: dict[str, int] = {}
        nonlinear_links: list[NonlinearLinkDefinition] = []

        for branch in model.branches:
            num_elements = max(branch.discretization.num_elements, 1)
            nodes: list[BranchNodeState] = []
            for node_index in range(num_elements + 1):
                station = node_index / num_elements
                label_prefix = f"branch:{branch.branch_id}:n{node_index}"
                dofs = tuple(
                    dof_manager.register_label(f"{label_prefix}:{component}")
                    for component in COMPONENT_LABELS
                )
                nodes.append(
                    BranchNodeState(
                        label_prefix=label_prefix,
                        position=branch.path.point_at(station),
                        station=station,
                        dofs=dofs,
                    )
                )
            branch_nodes[branch.branch_id] = nodes

        for fruit in model.fruits:
            fruit_dofs[fruit.fruit_id] = dof_manager.register_label(f"fruit:{fruit.fruit_id}")

        stiffness = zero_matrix(dof_manager.size())
        mass = zero_matrix(dof_manager.size())
        damping = zero_matrix(dof_manager.size())
        gravity_load = [0.0 for _ in range(dof_manager.size())]
        material_lookup = build_material_lookup(model.materials)
        gravity_scale = 9.81
        apply_gravity_prestress = model.analysis.include_gravity_prestress
        gravity_direction = (
            normalize(Vec3(*model.analysis.gravity_direction))
            if apply_gravity_prestress
            else Vec3(0.0, 0.0, -1.0)
        )

        for branch in model.branches:
            nodes = branch_nodes[branch.branch_id]
            branch_elements[branch.branch_id] = []
            for element_index in range(len(nodes) - 1):
                first = nodes[element_index]
                second = nodes[element_index + 1]
                first_state = evaluate_branch_section_state(branch, material_lookup, first.station)
                second_state = evaluate_branch_section_state(branch, material_lookup, second.station)

                area = 0.5 * (first_state.area + second_state.area)
                iy = 0.5 * (first_state.ix + second_state.ix)
                iz = 0.5 * (first_state.iy + second_state.iy)
                polar_moment = max(
                    0.5 * (first_state.polar_moment + second_state.polar_moment),
                    iy + iz,
                )
                mass_per_length = 0.5 * (
                    first_state.mass_per_length + second_state.mass_per_length
                )
                element_length = distance(first.position, second.position)

                properties = BeamElementProperties(
                    youngs_modulus=0.5
                    * (
                        first_state.effective_youngs_modulus
                        + second_state.effective_youngs_modulus
                    ),
                    shear_modulus=0.5
                    * (
                        first_state.effective_shear_modulus
                        + second_state.effective_shear_modulus
                    ),
                    area=area,
                    iy=iy,
                    iz=iz,
                    torsion_constant=polar_moment,
                    density=mass_per_length / area if area > 0.0 else 0.0,
                    length=element_length,
                )
                transformation = build_transformation_matrix(first.position, second.position)
                global_stiffness, global_mass = build_global_element_matrices(
                    properties=properties,
                    start=first.position,
                    end=second.position,
                )
                element_dofs = list(first.dofs) + list(second.dofs)
                scatter(stiffness, global_stiffness, element_dofs)
                scatter(mass, global_mass, element_dofs)
                branch_elements[branch.branch_id].append(
                    BranchElementState(
                        branch_id=branch.branch_id,
                        element_index=element_index,
                        dofs=tuple(element_dofs),
                        transformation_matrix=transformation,
                        length=element_length,
                        axial_rigidity=properties.youngs_modulus * properties.area,
                    )
                )

                if apply_gravity_prestress:
                    nodal_scale = 0.5 * mass_per_length * gravity_scale * element_length
                    nodal_force = Vec3(
                        gravity_direction.x * nodal_scale,
                        gravity_direction.y * nodal_scale,
                        gravity_direction.z * nodal_scale,
                    )
                    for node in (first, second):
                        gravity_load[node.dofs[0]] += nodal_force.x
                        gravity_load[node.dofs[1]] += nodal_force.y
                        gravity_load[node.dofs[2]] += nodal_force.z

        for branch in model.branches:
            if branch.parent_branch_id is None:
                continue

            child_nodes = branch_nodes[branch.branch_id]
            child_root = child_nodes[0]
            nearest_parent = _nearest_parent_node(
                branch_nodes,
                branch.branch_id,
                branch.parent_branch_id,
            )
            matching_joint = model.find_joint_for_child(branch.branch_id)

            for component_index in range(6):
                add_pair_penalty(
                    stiffness,
                    child_root.dofs[component_index],
                    nearest_parent.dofs[component_index],
                    _joint_component_penalty(component_index, matching_joint),
                )
            _append_joint_nonlinear_links(
                nonlinear_links,
                matching_joint,
                child_root,
                nearest_parent,
            )

        auto_nonlinear_levels = set(model.analysis.auto_nonlinear_levels)
        explicit_joint_children = {joint.child_branch_id for joint in model.joints}
        if auto_nonlinear_levels:
            for branch in model.branches:
                if branch.parent_branch_id is None:
                    continue
                if branch.level not in auto_nonlinear_levels:
                    continue
                if branch.branch_id in explicit_joint_children:
                    continue

                child_nodes = branch_nodes[branch.branch_id]
                child_root = child_nodes[0]
                nearest_parent = _nearest_parent_node(
                    branch_nodes,
                    branch.branch_id,
                    branch.parent_branch_id,
                )
                nonlinear_links.append(
                    NonlinearLinkDefinition(
                        label=f"auto_joint:{branch.branch_id}",
                        first_dof=child_root.dofs[0],
                        second_dof=nearest_parent.dofs[0],
                        kind=NonlinearLinkKind.CUBIC_SPRING,
                        cubic_stiffness=model.analysis.auto_nonlinear_cubic_scale,
                    )
                )

        for clamp in model.clamps:
            root = branch_nodes[clamp.branch_id][0]
            for dof in root.dofs:
                stiffness[dof][dof] += CONSTRAINT_PENALTY

            if abs(clamp.cubic_stiffness) > 0.0:
                nonlinear_links.append(
                    NonlinearLinkDefinition(
                        label=f"clamp:{clamp.branch_id}",
                        first_dof=root.dofs[0],
                        second_dof=-1,
                        kind=NonlinearLinkKind.CUBIC_SPRING,
                        linear_stiffness=CONSTRAINT_PENALTY,
                        cubic_stiffness=clamp.cubic_stiffness,
                    )
                )

        for fruit in model.fruits:
            fruit_dof = fruit_dofs[fruit.fruit_id]
            nodes = branch_nodes[fruit.branch_id]
            nearest_node = min(nodes, key=lambda node: fabs(node.station - fruit.location_s))
            coupled_branch_dof = nearest_node.dofs[0]

            mass[fruit_dof][fruit_dof] += max(fruit.mass, 1.0e-9)

            stiffness_value = max(fruit.stiffness, 1.0e-6)
            stiffness[fruit_dof][fruit_dof] += stiffness_value
            stiffness[fruit_dof][coupled_branch_dof] -= stiffness_value
            stiffness[coupled_branch_dof][fruit_dof] -= stiffness_value
            stiffness[coupled_branch_dof][coupled_branch_dof] += stiffness_value

            damping_value = max(fruit.damping, 0.0)
            damping[fruit_dof][fruit_dof] += damping_value
            damping[fruit_dof][coupled_branch_dof] -= damping_value
            damping[coupled_branch_dof][fruit_dof] -= damping_value
            damping[coupled_branch_dof][coupled_branch_dof] += damping_value

        if apply_gravity_prestress:
            from orchard_fem.solver_core import compute_gravity_axial_forces

            axial_forces = compute_gravity_axial_forces(
                LinearDynamicAssemblyResult(
                    stiffness_matrix=stiffness,
                    mass_matrix=mass,
                    damping_matrix=damping,
                    gravity_load=gravity_load,
                    dof_labels=dof_manager.labels,
                    branch_nodes=branch_nodes,
                    branch_elements=branch_elements,
                    fruit_dofs=fruit_dofs,
                    nonlinear_links=[],
                    excitation_dof=-1,
                    observation_names=[],
                    observation_dofs=[],
                ),
                gravity_load,
            )
            for branch in model.branches:
                for element, axial_force in zip(
                    branch_elements[branch.branch_id],
                    axial_forces[branch.branch_id],
                ):
                    local_geometric = build_local_geometric_stiffness_matrix(axial_force, element.length)
                    global_geometric = transform_to_global(
                        local_geometric,
                        element.transformation_matrix,
                    )
                    scatter(stiffness, global_geometric, list(element.dofs))

        apply_rayleigh_damping(model, mass, stiffness, damping, material_lookup)

        excitation_nodes = branch_nodes[model.excitation.target_branch_id]
        excitation_dof = branch_dof(
            excitation_nodes,
            resolve_node_index(excitation_nodes, model.excitation.target_node),
            model.excitation.target_component,
        )

        observation_names: list[str] = []
        observation_dofs: list[int] = []
        for observation in model.observations:
            if observation.target_type == "branch":
                nodes = branch_nodes[observation.target_id]
                node_index = resolve_node_index(nodes, observation.target_node)
                if len(observation.target_components) == 1:
                    observation_names.append(observation.observation_id)
                    observation_dofs.append(
                        branch_dof(
                            nodes,
                            node_index,
                            observation.target_components[0],
                        )
                    )
                else:
                    for component in observation.target_components:
                        observation_names.append(f"{observation.observation_id}_{component}")
                        observation_dofs.append(
                            branch_dof(
                                nodes,
                                node_index,
                                component,
                            )
                        )
            elif observation.target_type == "fruit":
                observation_names.append(observation.observation_id)
                observation_dofs.append(fruit_dofs[observation.target_id])
            else:
                raise ValueError(
                    f"Unsupported observation target type: {observation.target_type}"
                )

        if not observation_dofs:
            observation_names.append("excitation_branch")
            observation_dofs.append(excitation_dof)

        return LinearDynamicAssemblyResult(
            stiffness_matrix=stiffness,
            mass_matrix=mass,
            damping_matrix=damping,
            gravity_load=gravity_load,
            dof_labels=dof_manager.labels,
            branch_nodes=branch_nodes,
            branch_elements=branch_elements,
            fruit_dofs=fruit_dofs,
            nonlinear_links=nonlinear_links,
            excitation_dof=excitation_dof,
            observation_names=observation_names,
            observation_dofs=observation_dofs,
        )


class OrchardModalAssembler(OrchardSystemAssembler):
    pass
