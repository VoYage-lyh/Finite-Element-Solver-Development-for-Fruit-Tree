from __future__ import annotations

from dataclasses import replace

from orchard_fem.discretization.types import COMPONENT_INDEX
from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.branch_dofs import resolve_branch_node_dofs
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle
from orchard_fem.fenicsx.operator_bundle import EmbeddedBeamOperatorBundle
from orchard_fem.fenicsx.petsc_ops import (
    accumulate_owned_matrix_value,
    copy_owned_matrix_entries,
    create_empty_aij_matrix_like,
)


def fruit_component_index(fruit) -> int:
    try:
        component_index = COMPONENT_INDEX[fruit.target_component]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported fruit target_component '{fruit.target_component}' "
            f"for fruit '{fruit.fruit_id}'"
        ) from exc
    if component_index >= 3:
        raise ValueError(
            f"Fruit '{fruit.fruit_id}' must target a translational component "
            "(ux, uy, or uz)."
        )
    return component_index


def augment_operators_with_fruits(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    mesh_spec: EmbeddedLineMeshSpec,
    operator_bundle: EmbeddedBeamOperatorBundle,
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    if not model.fruits:
        return operator_bundle

    base_size = operator_bundle.stiffness_matrix.getSize()[0]
    total_size = base_size + len(model.fruits)
    stiffness_matrix = create_empty_aij_matrix_like(
        operator_bundle.stiffness_matrix,
        total_size,
    )
    mass_matrix = create_empty_aij_matrix_like(
        operator_bundle.mass_matrix,
        total_size,
    )
    damping_matrix = create_empty_aij_matrix_like(
        operator_bundle.stiffness_matrix,
        total_size,
    )

    copy_owned_matrix_entries(operator_bundle.stiffness_matrix, stiffness_matrix)
    copy_owned_matrix_entries(operator_bundle.mass_matrix, mass_matrix)

    stiffness_owned_rows = stiffness_matrix.getOwnershipRange()
    mass_owned_rows = mass_matrix.getOwnershipRange()
    damping_owned_rows = damping_matrix.getOwnershipRange()
    branch_node_dofs = resolve_branch_node_dofs(model, space_bundle, mesh_spec)

    fruit_dofs: dict[str, int] = {}
    for fruit_index, fruit in enumerate(model.fruits):
        fruit_dof = base_size + fruit_index
        fruit_dofs[fruit.fruit_id] = fruit_dof

        branch = model.require_branch(fruit.branch_id)
        num_elements = max(branch.discretization.num_elements, 1)
        node_index = min(
            range(num_elements + 1),
            key=lambda candidate: abs((candidate / num_elements) - fruit.location_s),
        )
        coupled_branch_dof = branch_node_dofs[fruit.branch_id][node_index][
            fruit_component_index(fruit)
        ]

        accumulate_owned_matrix_value(
            mass_matrix,
            mass_owned_rows,
            fruit_dof,
            fruit_dof,
            max(fruit.mass, 1.0e-9),
        )

        stiffness_value = max(fruit.stiffness, 1.0e-6)
        for row, column, sign in (
            (fruit_dof, fruit_dof, 1.0),
            (fruit_dof, coupled_branch_dof, -1.0),
            (coupled_branch_dof, fruit_dof, -1.0),
            (coupled_branch_dof, coupled_branch_dof, 1.0),
        ):
            accumulate_owned_matrix_value(
                stiffness_matrix,
                stiffness_owned_rows,
                row,
                column,
                sign * stiffness_value,
            )

        damping_value = max(fruit.damping, 0.0)
        if damping_value > 0.0:
            for row, column, sign in (
                (fruit_dof, fruit_dof, 1.0),
                (fruit_dof, coupled_branch_dof, -1.0),
                (coupled_branch_dof, fruit_dof, -1.0),
                (coupled_branch_dof, coupled_branch_dof, 1.0),
            ):
                accumulate_owned_matrix_value(
                    damping_matrix,
                    damping_owned_rows,
                    row,
                    column,
                    sign * damping_value,
                )

    stiffness_matrix.assemble()
    mass_matrix.assemble()
    damping_matrix.assemble()

    return replace(
        operator_bundle,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=mass_matrix,
        attachment_damping_matrix=damping_matrix,
        fruit_dofs=fruit_dofs,
    )
