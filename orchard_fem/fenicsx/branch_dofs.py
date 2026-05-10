from __future__ import annotations

from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.dofs import resolve_embedded_beam_component_dof_by_vertex
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle

BRANCH_COMPONENTS = ("ux", "uy", "uz", "rx", "ry", "rz")
BranchNodeDofs = tuple[int, int, int, int, int, int]
BranchNodeDofMap = dict[str, list[BranchNodeDofs]]


def nearest_parent_node_index(
    model: OrchardModel,
    child_branch_id: str,
    parent_branch_id: str,
) -> int:
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


def resolve_branch_node_dofs(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
) -> BranchNodeDofMap:
    branch_node_dofs: BranchNodeDofMap = {}

    for branch in model.branches:
        cell_indices = mesh_spec.cells_for_branch(branch.branch_id)
        if not cell_indices:
            raise ValueError(f"Branch '{branch.branch_id}' has no mesh cells")
        vertex_indices = [mesh_spec.cells[cell_indices[0]][0]]
        vertex_indices.extend(mesh_spec.cells[cell_index][1] for cell_index in cell_indices)
        nodes: list[BranchNodeDofs] = []
        for vertex_index in vertex_indices:
            dofs = tuple(
                resolve_embedded_beam_component_dof_by_vertex(
                    space_bundle,
                    vertex_index,
                    component,
                )
                for component in BRANCH_COMPONENTS
            )
            nodes.append(dofs)  # type: ignore[arg-type]
        branch_node_dofs[branch.branch_id] = nodes
    return branch_node_dofs
