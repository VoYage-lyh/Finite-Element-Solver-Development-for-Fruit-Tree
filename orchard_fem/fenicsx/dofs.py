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
    # For DIRECTIONAL excitation (excitation.target_direction set): the node's
    # translational DOFs with their unit-direction weights, [(dof, weight), …].
    # ``None`` → single-component excitation on ``excitation_dof`` (default).
    excitation_direction_dofs: list[tuple[int, float]] | None = None


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
        # Multiple branches can share an endpoint (e.g. several level-2 children
        # attached at the same node of a level-1 parent). Each branch's embedded
        # line contributes its own vertex at that location, so the geometric
        # lookup returns one DOF per incident branch. After the joint operator
        # is assembled (coincident DOFs, MPC, or penalty), the values at these
        # DOFs coincide to within solver tolerance, so any representative works
        # for observation purposes. Pick the smallest for determinism.
        import warnings

        warnings.warn(
            f"Multiple DOFs ({len(dofs)}) found at point {point} for component "
            f"{component}; using the smallest. Verify joint coupling at this "
            "shared node.",
            stacklevel=2,
        )
        return int(min(dofs))
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
        import warnings

        warnings.warn(
            f"Multiple DOFs ({len(dofs)}) found at vertex {vertex_index} for "
            f"component {component}; using the smallest.",
            stacklevel=2,
        )
        return int(min(dofs))
    return int(dofs[0])


def resolve_embedded_beam_response_mapping(
    model: OrchardModel,
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    *,
    fruit_dofs: dict[str, tuple[int, int]] | None = None,
    atol: float = 1.0e-8,
) -> EmbeddedBeamResponseMapping:
    excitation_branch = model.require_branch(model.excitation.target_branch_id)
    excitation_point = _target_point(excitation_branch, model.excitation.target_node)

    excitation_direction_dofs: list[tuple[int, float]] | None = None
    direction = model.excitation.target_direction
    if direction is not None:
        # DIRECTIONAL excitation: drive the node's ux/uy/uz translational DOFs
        # weighted by the unit direction (⊥ the branch axis / any 3-D direction).
        norm = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
        if norm <= 1.0e-14:
            raise ValueError("excitation.target_direction must be non-zero.")
        excitation_direction_dofs = []
        for component, weight in zip(("ux", "uy", "uz"), direction):
            w = float(weight) / norm
            if abs(w) < 1.0e-12:
                continue
            dof = resolve_embedded_beam_component_dof(
                space_bundle, excitation_point, component, atol=atol,
            )
            excitation_direction_dofs.append((dof, w))
        # The primary DOF (largest |weight|) reports the driven amplitude.
        excitation_dof = max(excitation_direction_dofs, key=lambda dw: abs(dw[1]))[0]
    else:
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
            # Fruit is a 2-DOF horizontal pendulum (x-swing, y-swing). The
            # observation component picks the axis: uy→y-swing, else x-swing.
            swing = fruit_dofs[observation.target_id]
            comp = (observation.target_components or ["ux"])[0]
            observation_names.append(observation.observation_id)
            observation_dofs.append(swing[1] if comp == "uy" else swing[0])
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
        excitation_direction_dofs=excitation_direction_dofs,
    )
