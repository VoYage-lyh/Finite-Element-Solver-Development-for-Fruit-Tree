from __future__ import annotations

from dataclasses import replace

from orchard_fem.discretization.types import (
    COMPONENT_LABELS,
    NonlinearLinkDefinition,
    NonlinearLinkKind,
)
from orchard_fem.domain import JointDefinition, JointLawKind, OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.branch_dofs import (
    BranchNodeDofs,
    nearest_parent_node_index,
    resolve_branch_node_dofs,
)
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle
from orchard_fem.fenicsx.operator_bundle import EmbeddedBeamOperatorBundle
from orchard_fem.fenicsx.petsc_ops import (
    accumulate_owned_matrix_value,
    allow_new_nonzero_allocation,
)

CONSTRAINT_PENALTY = 1.0e12


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
    child_root: BranchNodeDofs,
    nearest_parent: BranchNodeDofs,
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
            if child_root[component_index] == nearest_parent[component_index]:
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
            if child_root[component_index] == nearest_parent[component_index]:
                continue
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


def augment_operators_with_joints(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    branch_node_dofs = resolve_branch_node_dofs(model, space_bundle, mesh_spec)
    stiffness_matrix = operator_bundle.stiffness_matrix.duplicate(copy=True)
    allow_new_nonzero_allocation(stiffness_matrix)
    owned_rows = stiffness_matrix.getOwnershipRange()
    nonlinear_links = list(operator_bundle.nonlinear_links)
    has_branch_connections = False

    for branch in model.branches:
        if branch.parent_branch_id is None:
            continue
        if branch.branch_id in operator_bundle.mpc_constrained_branch_ids:
            continue
        has_branch_connections = True

        child_nodes = branch_node_dofs[branch.branch_id]
        child_root = child_nodes[0]
        parent_node_index = nearest_parent_node_index(
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
            if first_dof == second_dof:
                continue
            accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                first_dof,
                first_dof,
                penalty,
            )
            accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                second_dof,
                second_dof,
                penalty,
            )
            accumulate_owned_matrix_value(
                stiffness_matrix,
                owned_rows,
                first_dof,
                second_dof,
                -penalty,
            )
            accumulate_owned_matrix_value(
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
    return replace(
        operator_bundle,
        stiffness_matrix=stiffness_matrix,
        nonlinear_links=nonlinear_links,
    )


def augment_operators_with_auto_nonlinear_links(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    auto_nonlinear_levels = set(model.analysis.auto_nonlinear_levels)
    if not auto_nonlinear_levels:
        return operator_bundle

    branch_node_dofs = resolve_branch_node_dofs(model, space_bundle, mesh_spec)
    explicit_joint_children = {joint.child_branch_id for joint in model.joints}
    nonlinear_links = list(operator_bundle.nonlinear_links)

    for branch in model.branches:
        if branch.parent_branch_id is None:
            continue
        if branch.level not in auto_nonlinear_levels:
            continue
        if branch.branch_id in explicit_joint_children:
            continue

        parent_node_index = nearest_parent_node_index(
            model,
            branch.branch_id,
            branch.parent_branch_id,
        )
        first_dof = branch_node_dofs[branch.branch_id][0][0]
        second_dof = branch_node_dofs[branch.parent_branch_id][parent_node_index][0]
        if first_dof == second_dof:
            continue
        nonlinear_links.append(
            NonlinearLinkDefinition(
                label=f"auto_joint:{branch.branch_id}",
                first_dof=first_dof,
                second_dof=second_dof,
                kind=NonlinearLinkKind.CUBIC_SPRING,
                cubic_stiffness=model.analysis.auto_nonlinear_cubic_scale,
            )
        )

    return replace(
        operator_bundle,
        nonlinear_links=nonlinear_links,
    )


def augment_operators_with_nonlinear_clamps(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    nonlinear_clamps = [
        clamp for clamp in model.clamps if abs(clamp.cubic_stiffness) > 1.0e-14
    ]
    if not nonlinear_clamps:
        return operator_bundle

    branch_node_dofs = resolve_branch_node_dofs(model, space_bundle, mesh_spec)
    stiffness_matrix = operator_bundle.stiffness_matrix.duplicate(copy=True)
    allow_new_nonzero_allocation(stiffness_matrix)
    owned_rows = stiffness_matrix.getOwnershipRange()
    nonlinear_links = list(operator_bundle.nonlinear_links)

    for clamp in nonlinear_clamps:
        root_dofs = branch_node_dofs[clamp.branch_id][0]
        for dof in root_dofs:
            accumulate_owned_matrix_value(
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
    return replace(
        operator_bundle,
        stiffness_matrix=stiffness_matrix,
        nonlinear_links=nonlinear_links,
    )
