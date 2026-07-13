from __future__ import annotations

import random
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


# Duffing-type parameter ranges adapted from the manuscript (Liu et al., Table 3).
# The paper's k3 and c2 are calibrated against a single-degree-of-freedom local
# element with relative displacement at the mm scale; converted to SI those
# would be 1.8e9-3.3e9 N/m^3 and 1.2e4-4.3e4 N·s^2/m^2.
#
# We attach the nonlinear link between the branch Tip and Root (a Tip-Root
# beam in our reduced-order tree), so the link's Δu is the full bending
# amplitude of the branch — several mm at resonance.  At that scale the
# unscaled manuscript ranges produce O(100) N restoring force per link, which
# overwhelms the harmonic-balance Newton.  We scale both k3 and c2 down by two
# orders of magnitude so the qualitative hardening/softening picture is
# preserved while the FRF Newton stays well-conditioned.
_K3_DOWNSCALE = 0.01
_C2_DOWNSCALE = 0.1
_HARDENING_K3_RANGE_NPM3 = (1.8e9 * _K3_DOWNSCALE, 3.3e9 * _K3_DOWNSCALE)
_SOFTENING_K3_RANGE_NPM3 = (-2.1e9 * _K3_DOWNSCALE, -0.9e9 * _K3_DOWNSCALE)
_HARDENING_C2_RANGE_NSPM2 = (1.2e4 * _C2_DOWNSCALE, 2.5e4 * _C2_DOWNSCALE)
_SOFTENING_C2_RANGE_NSPM2 = (2.6e4 * _C2_DOWNSCALE, 4.3e4 * _C2_DOWNSCALE)
# Per-level probability of softening response, derived from Fig. 6 distribution
# of nonlinear-unit acceptance at the auto-injected joint position (child Root
# = parent Tip area for that branch level).
_SOFTENING_PROBABILITY_BY_LEVEL = {2: 0.25, 3: 0.55}
_DEFAULT_SOFTENING_PROBABILITY = 0.5

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

    randomize = bool(model.analysis.auto_nonlinear_randomize)
    rng = (
        random.Random(int(model.analysis.auto_nonlinear_seed)) if randomize else None
    )

    eligible_branches = sorted(
        (
            branch
            for branch in model.branches
            if branch.parent_branch_id is not None
            and branch.level in auto_nonlinear_levels
            and branch.branch_id not in explicit_joint_children
        ),
        key=lambda branch: branch.branch_id,
    )

    for branch in eligible_branches:
        # Attach the nonlinear element across the branch itself (Tip ux vs
        # Root ux), not across the parent-child junction.  The junction DOFs
        # are penalty-pinned by ``augment_operators_with_joints`` so their
        # relative displacement is forced to ~0 — both k3 Δu^3 and
        # c2 |Δv| Δv would then contribute essentially nothing.  Across the
        # same branch the bending mode is free, so the relative displacement
        # is mm-scale at resonance and the Duffing correction actually
        # modifies the response.  This matches the manuscript's reduced-order
        # interpretation of f_nl,e as a *local element* equivalent restoring
        # force superposed on the linear baseline (Liu et al., Eq. (4)).
        nodes = branch_node_dofs[branch.branch_id]
        if len(nodes) < 2:
            continue
        tip_dofs = nodes[-1]
        root_dofs = nodes[0]
        first_dof = tip_dofs[0]    # branch tip, ux component
        second_dof = root_dofs[0]  # branch root, ux component
        if first_dof == second_dof:
            continue

        if rng is None:
            cubic_stiffness = model.analysis.auto_nonlinear_cubic_scale
            quadratic_damping = 0.0
        else:
            probability = _SOFTENING_PROBABILITY_BY_LEVEL.get(
                branch.level, _DEFAULT_SOFTENING_PROBABILITY
            )
            if rng.random() < probability:
                cubic_stiffness = rng.uniform(*_SOFTENING_K3_RANGE_NPM3)
                quadratic_damping = rng.uniform(*_SOFTENING_C2_RANGE_NSPM2)
            else:
                cubic_stiffness = rng.uniform(*_HARDENING_K3_RANGE_NPM3)
                quadratic_damping = rng.uniform(*_HARDENING_C2_RANGE_NSPM2)

        nonlinear_links.append(
            NonlinearLinkDefinition(
                label=f"auto_branch:{branch.branch_id}",
                first_dof=first_dof,
                second_dof=second_dof,
                kind=NonlinearLinkKind.CUBIC_SPRING,
                cubic_stiffness=cubic_stiffness,
                quadratic_damping=quadratic_damping,
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
        clamp for clamp in model.clamps
        if abs(clamp.cubic_stiffness) > 1.0e-14
        or abs(clamp.quadratic_damping) > 1.0e-14
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
                quadratic_damping=clamp.quadratic_damping,
            )
        )

    stiffness_matrix.assemble()
    return replace(
        operator_bundle,
        stiffness_matrix=stiffness_matrix,
        nonlinear_links=nonlinear_links,
    )
