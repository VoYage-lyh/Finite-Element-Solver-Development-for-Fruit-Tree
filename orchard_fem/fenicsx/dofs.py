from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orchard_fem.discretization import resolve_node_index
from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle

_DISPLACEMENT_COMPONENTS = {"ux": 0, "uy": 1, "uz": 2}
_ROTATION_COMPONENTS = {"rx": 0, "ry": 1, "rz": 2}


@dataclass(frozen=True)
class EmbeddedBeamResponseMapping:
    excitation_dof: int
    observation_names: list[str]
    observation_dofs: list[int]


def _point_marker(point: tuple[float, float, float], atol: float):
    def marker(x) -> np.ndarray:
        return np.logical_and.reduce(
            [
                np.isclose(x[0], point[0], atol=atol),
                np.isclose(x[1], point[1], atol=atol),
                np.isclose(x[2], point[2], atol=atol),
            ]
        )

    return marker


def _target_point(branch, target_node: str) -> tuple[float, float, float]:
    num_elements = max(branch.discretization.num_elements, 1)
    node_index = resolve_node_index([None] * (num_elements + 1), target_node)
    station = node_index / num_elements
    point = branch.path.point_at(station)
    return point.x, point.y, point.z


def _extract_parent_dofs(located_dofs: Any) -> np.ndarray:
    if isinstance(located_dofs, list | tuple):
        if not located_dofs:
            return np.asarray([], dtype=np.int64)
        return np.asarray(located_dofs[0], dtype=np.int64)

    dof_array = np.asarray(located_dofs, dtype=np.int64)
    if dof_array.ndim == 2:
        return dof_array[:, 0]
    return dof_array


def _ensure_vertex_connectivity(mesh: Any) -> None:
    topology = mesh.topology
    topology.create_connectivity(0, topology.dim)
    topology.create_connectivity(topology.dim, 0)


def resolve_embedded_beam_component_dof(
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    point: tuple[float, float, float],
    component: str,
    *,
    atol: float = 1.0e-8,
) -> int:
    require_dolfinx()

    from dolfinx import fem as dolfinx_fem

    if component in _DISPLACEMENT_COMPONENTS:
        parent_subspace = space_bundle.displacement_space
        component_index = _DISPLACEMENT_COMPONENTS[component]
    elif component in _ROTATION_COMPONENTS:
        parent_subspace = space_bundle.rotation_space
        component_index = _ROTATION_COMPONENTS[component]
    else:
        raise ValueError(f"Unsupported embedded beam component: {component}")

    component_space = parent_subspace.sub(component_index)
    collapsed_space, _ = component_space.collapse()
    _ensure_vertex_connectivity(space_bundle.mesh)
    _ensure_vertex_connectivity(collapsed_space.mesh)
    dof_pairs = dolfinx_fem.locate_dofs_geometrical(
        (component_space, collapsed_space),
        _point_marker(point, atol),
    )
    dofs = np.unique(_extract_parent_dofs(dof_pairs))

    if dofs.size == 0:
        raise ValueError(
            f"No DOF found at point {point} for component {component}."
        )

    if len(dofs) != 1:
        raise ValueError(
            f"Expected exactly one DOF at point {point} for component {component}, found {len(dofs)}."
        )
    return int(dofs[0])


def resolve_embedded_beam_component_dof_by_vertex(
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    vertex_index: int,
    component: str,
) -> int:
    require_dolfinx()

    from dolfinx import fem as dolfinx_fem

    if component in _DISPLACEMENT_COMPONENTS:
        parent_subspace = space_bundle.displacement_space
        component_index = _DISPLACEMENT_COMPONENTS[component]
    elif component in _ROTATION_COMPONENTS:
        parent_subspace = space_bundle.rotation_space
        component_index = _ROTATION_COMPONENTS[component]
    else:
        raise ValueError(f"Unsupported embedded beam component: {component}")

    component_space = parent_subspace.sub(component_index)
    collapsed_space, _ = component_space.collapse()
    _ensure_vertex_connectivity(space_bundle.mesh)
    _ensure_vertex_connectivity(collapsed_space.mesh)
    dof_pairs = dolfinx_fem.locate_dofs_topological(
        (component_space, collapsed_space),
        0,
        np.asarray([int(vertex_index)], dtype=np.int32),
    )
    dofs = np.unique(_extract_parent_dofs(dof_pairs))
    if dofs.size == 0:
        raise ValueError(
            f"No DOF found at vertex {vertex_index} for component {component}."
        )
    if len(dofs) != 1:
        raise ValueError(
            f"Expected exactly one DOF at vertex {vertex_index} for component "
            f"{component}, found {len(dofs)}."
        )
    return int(dofs[0])


def resolve_embedded_beam_response_mapping(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    *,
    fruit_dofs: dict[str, int] | None = None,
    atol: float = 1.0e-8,
) -> EmbeddedBeamResponseMapping:
    excitation_branch = model.require_branch(model.excitation.target_branch_id)
    excitation_point = _target_point(excitation_branch, model.excitation.target_node)
    excitation_dof = resolve_embedded_beam_component_dof(
        space_bundle,
        excitation_point,
        model.excitation.target_component,
        atol=atol,
    )

    observation_names: list[str] = []
    observation_dofs: list[int] = []
    for observation in model.observations:
        if observation.target_type == "branch":
            branch = model.require_branch(observation.target_id)
            point = _target_point(branch, observation.target_node)
            if len(observation.target_components) == 1:
                observation_names.append(observation.observation_id)
                observation_dofs.append(
                    resolve_embedded_beam_component_dof(
                        space_bundle,
                        point,
                        observation.target_components[0],
                        atol=atol,
                    )
                )
            else:
                for component in observation.target_components:
                    observation_names.append(f"{observation.observation_id}_{component}")
                    observation_dofs.append(
                        resolve_embedded_beam_component_dof(
                            space_bundle,
                            point,
                            component,
                            atol=atol,
                        )
                    )
        elif observation.target_type == "fruit":
            if fruit_dofs is None or observation.target_id not in fruit_dofs:
                raise NotImplementedError(
                    "The experimental FEniCSx branch cannot resolve fruit observations without fruit-DOF augmentation."
                )
            observation_names.append(observation.observation_id)
            observation_dofs.append(fruit_dofs[observation.target_id])
        else:
            raise NotImplementedError(
                "The experimental FEniCSx branch currently supports only branch and fruit observations."
            )
    if not observation_dofs:
        observation_names.append("excitation_branch")
        observation_dofs.append(excitation_dof)

    return EmbeddedBeamResponseMapping(
        excitation_dof=excitation_dof,
        observation_names=observation_names,
        observation_dofs=observation_dofs,
    )
