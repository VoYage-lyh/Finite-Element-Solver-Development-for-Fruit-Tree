from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.dofs import resolve_embedded_beam_component_dof_by_vertex
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle


@dataclass(frozen=True)
class BranchConnectionMPC:
    constraint: Any | None
    constrained_branch_ids: frozenset[str]

    @property
    def enabled(self) -> bool:
        return self.constraint is not None and bool(self.constrained_branch_ids)


def _nearest_parent_vertex(
    model: OrchardModel,
    mesh_spec: EmbeddedLineMeshSpec,
    *,
    child_branch_id: str,
    parent_branch_id: str,
) -> int:
    child_cell_indices = mesh_spec.cells_for_branch(child_branch_id)
    parent_cell_indices = mesh_spec.cells_for_branch(parent_branch_id)
    if not child_cell_indices or not parent_cell_indices:
        raise ValueError(
            f"Cannot build branch MPC for '{child_branch_id}' because one branch has no cells."
        )

    child_root_vertex = mesh_spec.cells[child_cell_indices[0]][0]
    child_root = np.asarray(mesh_spec.points[child_root_vertex], dtype=np.float64)
    parent_vertices = [mesh_spec.cells[parent_cell_indices[0]][0]]
    parent_vertices.extend(mesh_spec.cells[cell_index][1] for cell_index in parent_cell_indices)
    return min(
        parent_vertices,
        key=lambda vertex: float(
            np.sum((np.asarray(mesh_spec.points[vertex], dtype=np.float64) - child_root) ** 2)
        ),
    )


def _local_dof_to_global_owner(function_space: Any, dof: int) -> tuple[int, int]:
    index_map = function_space.dofmap.index_map
    block_size = function_space.dofmap.index_map_bs
    block = int(dof) // block_size
    remainder = int(dof) % block_size

    if block < index_map.size_local:
        global_block = int(index_map.local_to_global(np.asarray([block], dtype=np.int32))[0])
        owner = int(function_space.mesh.comm.rank)
    else:
        ghost_index = block - index_map.size_local
        global_block = int(index_map.ghosts[ghost_index])
        owner = int(index_map.owners[ghost_index])

    return (global_block * block_size) + remainder, owner


def _branch_root_vertex(mesh_spec: EmbeddedLineMeshSpec, branch_id: str) -> int:
    cell_indices = mesh_spec.cells_for_branch(branch_id)
    if not cell_indices:
        raise ValueError(f"Branch '{branch_id}' has no mesh cells")
    return mesh_spec.cells[cell_indices[0]][0]


def build_linear_branch_connection_mpc(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
) -> BranchConnectionMPC:
    require_dolfinx()

    explicit_joint_children = {joint.child_branch_id for joint in model.joints}
    auto_nonlinear_levels = set(model.analysis.auto_nonlinear_levels)
    candidate_branches = [
        branch
        for branch in model.branches
        if branch.parent_branch_id is not None
        and branch.branch_id not in explicit_joint_children
        and branch.level not in auto_nonlinear_levels
    ]
    if not candidate_branches:
        return BranchConnectionMPC(None, frozenset())

    import dolfinx_mpc

    components = ("ux", "uy", "uz", "rx", "ry", "rz")
    slaves: list[int] = []
    masters: list[int] = []
    coeffs: list[float] = []
    owners: list[int] = []
    offsets: list[int] = [0]
    constrained_branch_ids: set[str] = set()

    for branch in candidate_branches:
        assert branch.parent_branch_id is not None
        child_vertex = _branch_root_vertex(mesh_spec, branch.branch_id)
        parent_vertex = _nearest_parent_vertex(
            model,
            mesh_spec,
            child_branch_id=branch.branch_id,
            parent_branch_id=branch.parent_branch_id,
        )

        branch_slave_count = 0
        for component in components:
            slave_dof = resolve_embedded_beam_component_dof_by_vertex(
                space_bundle,
                child_vertex,
                component,
            )
            master_dof = resolve_embedded_beam_component_dof_by_vertex(
                space_bundle,
                parent_vertex,
                component,
            )
            if slave_dof == master_dof:
                continue

            master_global_dof, master_owner = _local_dof_to_global_owner(
                space_bundle.mixed_space,
                master_dof,
            )
            slaves.append(int(slave_dof))
            masters.append(master_global_dof)
            owners.append(master_owner)
            coeffs.append(1.0)
            offsets.append(len(masters))
            branch_slave_count += 1

        if branch_slave_count:
            constrained_branch_ids.add(branch.branch_id)

    if not slaves:
        return BranchConnectionMPC(None, frozenset())

    constraint = dolfinx_mpc.MultiPointConstraint(space_bundle.mixed_space)
    constraint.add_constraint(
        space_bundle.mixed_space,
        np.asarray(slaves, dtype=np.int32),
        np.asarray(masters, dtype=np.int64),
        np.asarray(coeffs, dtype=np.float64),
        np.asarray(owners, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
    )
    constraint.finalize()
    return BranchConnectionMPC(
        constraint,
        frozenset(constrained_branch_ids),
    )


def backsubstitute_mpc_vector(mpc: Any | None, vector: Any) -> Any:
    if mpc is not None:
        mpc.backsubstitution(vector)
    return vector


def apply_mpc_to_vector_in_place(mpc: Any | None, vector: Any) -> Any:
    if mpc is None:
        return vector

    from petsc4py import PETSc

    coeffs, offsets = mpc.coefficients()
    owned_start, owned_end = vector.getOwnershipRange()
    for slave in mpc.slaves:
        slave_index = int(slave)
        if not (owned_start <= slave_index < owned_end):
            continue
        slave_value = float(vector.getValues([slave_index])[0])
        vector.setValue(
            slave_index,
            0.0,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        if abs(slave_value) <= 0.0:
            continue
        master_dofs = mpc.masters.links(slave_index)
        master_coeffs = coeffs[offsets[slave_index] : offsets[slave_index + 1]]
        for master_dof, coefficient in zip(master_dofs, master_coeffs):
            master_index = int(master_dof)
            if not (owned_start <= master_index < owned_end):
                continue
            vector.setValue(
                master_index,
                float(coefficient) * slave_value,
                addv=PETSc.InsertMode.ADD_VALUES,
            )
    vector.assemblyBegin()
    vector.assemblyEnd()
    return vector


def apply_mpc_to_real_block_vector_in_place(
    mpc: Any | None,
    block_vector: Any,
    *,
    system_size: int,
    template_matrix: Any,
) -> Any:
    if mpc is None:
        return block_vector

    real_part = template_matrix.createVecRight()
    imaginary_part = template_matrix.createVecRight()
    indices = list(range(system_size))
    real_part.setValues(indices, block_vector.getValues(indices))
    imaginary_part.setValues(
        indices,
        block_vector.getValues([system_size + index for index in indices]),
    )
    real_part.assemblyBegin()
    real_part.assemblyEnd()
    imaginary_part.assemblyBegin()
    imaginary_part.assemblyEnd()

    apply_mpc_to_vector_in_place(mpc, real_part)
    apply_mpc_to_vector_in_place(mpc, imaginary_part)

    block_vector.setValues(indices, real_part.getValues(indices))
    block_vector.setValues(
        [system_size + index for index in indices],
        imaginary_part.getValues(indices),
    )
    block_vector.assemblyBegin()
    block_vector.assemblyEnd()
    return block_vector


def backsubstitute_mpc_real_block_vector(
    mpc: Any | None,
    block_vector: Any,
    *,
    system_size: int,
    template_matrix: Any,
) -> Any:
    if mpc is None:
        return block_vector

    real_part = template_matrix.createVecRight()
    imaginary_part = template_matrix.createVecRight()
    indices = list(range(system_size))
    real_part.setValues(indices, block_vector.getValues(indices))
    imaginary_part.setValues(
        indices,
        block_vector.getValues([system_size + index for index in indices]),
    )
    real_part.assemblyBegin()
    real_part.assemblyEnd()
    imaginary_part.assemblyBegin()
    imaginary_part.assemblyEnd()

    mpc.backsubstitution(real_part)
    mpc.backsubstitution(imaginary_part)

    updated = block_vector.duplicate()
    block_vector.copy(updated)
    updated.setValues(indices, real_part.getValues(indices))
    updated.setValues(
        [system_size + index for index in indices],
        imaginary_part.getValues(indices),
    )
    updated.assemblyBegin()
    updated.assemblyEnd()
    return updated
