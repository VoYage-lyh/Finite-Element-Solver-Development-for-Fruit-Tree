from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import fabs

from orchard_fem.elements.beam_formulation import (
    BeamElementProperties,
    build_local_geometric_stiffness_matrix,
    build_global_element_matrices,
    build_transformation_matrix,
    transform_to_global,
)
from orchard_fem.materials.base import (
    BranchSectionState,
    build_material_lookup,
    evaluate_branch_section_state,
)
from orchard_fem.model import OrchardModel
from orchard_fem.topology.tree import Vec3, distance, normalize


Matrix = list[list[float]]
CONSTRAINT_PENALTY = 1.0e10
COMPONENT_LABELS = ("ux", "uy", "uz", "rx", "ry", "rz")
COMPONENT_INDEX = {label: index for index, label in enumerate(COMPONENT_LABELS)}


@dataclass(frozen=True)
class BranchNodeState:
    label_prefix: str
    position: object
    station: float
    dofs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class BranchElementState:
    branch_id: str
    element_index: int
    dofs: tuple[int, ...]
    transformation_matrix: Matrix
    length: float
    axial_rigidity: float


@dataclass(frozen=True)
class LinearDynamicAssemblyResult:
    stiffness_matrix: Matrix
    mass_matrix: Matrix
    damping_matrix: Matrix
    gravity_load: list[float]
    dof_labels: list[str]
    branch_nodes: dict[str, list[BranchNodeState]]
    branch_elements: dict[str, list[BranchElementState]]
    fruit_dofs: dict[str, int]
    nonlinear_links: list["NonlinearLinkDefinition"]
    excitation_dof: int
    observation_names: list[str]
    observation_dofs: list[int]


ModalAssemblyResult = LinearDynamicAssemblyResult


class NonlinearLinkKind(str, Enum):
    CUBIC_SPRING = "cubic_spring"
    GAP_SPRING = "gap_spring"


@dataclass(frozen=True)
class NonlinearLinkDefinition:
    label: str
    first_dof: int
    second_dof: int
    kind: NonlinearLinkKind
    linear_stiffness: float = 0.0
    cubic_stiffness: float = 0.0
    open_stiffness: float = 0.0
    gap_threshold: float = 0.0


class _DOFManager:
    def __init__(self) -> None:
        self._label_to_index: dict[str, int] = {}
        self._labels: list[str] = []

    def register_label(self, label: str) -> int:
        if label in self._label_to_index:
            return self._label_to_index[label]
        index = len(self._labels)
        self._labels.append(label)
        self._label_to_index[label] = index
        return index

    @property
    def labels(self) -> list[str]:
        return self._labels

    def size(self) -> int:
        return len(self._labels)


def _zero_matrix(size: int) -> Matrix:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _scatter(global_matrix: Matrix, local_matrix: Matrix, dofs: list[int]) -> None:
    for row in range(len(dofs)):
        for col in range(len(dofs)):
            global_matrix[dofs[row]][dofs[col]] += local_matrix[row][col]


def _add_pair_penalty(matrix: Matrix, first_dof: int, second_dof: int, penalty: float) -> None:
    matrix[first_dof][first_dof] += penalty
    matrix[first_dof][second_dof] -= penalty
    matrix[second_dof][first_dof] -= penalty
    matrix[second_dof][second_dof] += penalty


def _resolve_node_index(nodes: list[BranchNodeState], target_node: str) -> int:
    if not nodes:
        raise ValueError("Branch has no discretized nodes")
    if target_node == "root":
        return 0
    if target_node == "tip":
        return len(nodes) - 1

    node_index = int(target_node)
    if node_index < 0 or node_index >= len(nodes):
        raise ValueError(f"Requested node index is out of range: {target_node}")
    return node_index


def _branch_dof(nodes: list[BranchNodeState], node_index: int, component: str) -> int:
    try:
        component_index = COMPONENT_INDEX[component]
    except KeyError as exc:
        raise ValueError(f"Unsupported DOF component: {component}") from exc
    return nodes[node_index].dofs[component_index]


def _trapezoidal_average(states: list[BranchSectionState], getter) -> float:
    if not states:
        return 0.0
    if len(states) == 1:
        return getter(states[0])

    total = 0.0
    total_span = states[-1].station - states[0].station
    if total_span <= 1.0e-12:
        return getter(states[0])

    for left, right in zip(states, states[1:]):
        span = right.station - left.station
        total += 0.5 * (getter(left) + getter(right)) * span
    return total / total_span


def _compute_default_damping_ratio(model: OrchardModel, material_lookup: dict[str, object]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0

    for branch in model.branches:
        profile_states = [
            evaluate_branch_section_state(branch, material_lookup, profile.station)
            for profile in branch.section_series.profiles
        ]
        average_mass = _trapezoidal_average(profile_states, lambda state: state.mass_per_length)
        average_damping = _trapezoidal_average(profile_states, lambda state: state.damping_ratio)
        branch_mass = average_mass * max(branch.path.length(), 1.0e-6)
        weighted_sum += branch_mass * average_damping
        total_weight += branch_mass

    return weighted_sum / total_weight if total_weight > 0.0 else 0.0


def _apply_rayleigh_damping(
    model: OrchardModel,
    mass: Matrix,
    stiffness: Matrix,
    damping: Matrix,
    material_lookup: dict[str, object],
) -> None:
    alpha = model.analysis.rayleigh_alpha
    beta = model.analysis.rayleigh_beta

    if abs(alpha) < 1.0e-14 and abs(beta) < 1.0e-14:
        zeta = _compute_default_damping_ratio(model, material_lookup)
        omega_ref = 2.0 * 3.14159265358979323846 * max(model.analysis.frequency_start_hz, 0.1)
        beta = (2.0 * zeta / omega_ref) if omega_ref > 0.0 else 0.0

    for row_index in range(len(mass)):
        for column_index in range(len(mass[row_index])):
            damping[row_index][column_index] += (
                alpha * mass[row_index][column_index]
            ) + (
                beta * stiffness[row_index][column_index]
            )


class OrchardSystemAssembler:
    def assemble(self, model: OrchardModel) -> LinearDynamicAssemblyResult:
        dof_manager = _DOFManager()
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

        stiffness = _zero_matrix(dof_manager.size())
        mass = _zero_matrix(dof_manager.size())
        damping = _zero_matrix(dof_manager.size())
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
                _scatter(stiffness, global_stiffness, element_dofs)
                _scatter(mass, global_mass, element_dofs)
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
            parent_nodes = branch_nodes[branch.parent_branch_id]
            child_root = child_nodes[0]
            nearest_parent = min(
                parent_nodes,
                key=lambda node: distance(child_root.position, node.position),
            )

            penalty = CONSTRAINT_PENALTY
            matching_joint = next(
                (joint for joint in model.joints if joint.child_branch_id == branch.branch_id),
                None,
            )
            if matching_joint is not None:
                penalty *= max(matching_joint.linear_stiffness_scale, 1.0e-6)

            for component_index in range(6):
                _add_pair_penalty(
                    stiffness,
                    child_root.dofs[component_index],
                    nearest_parent.dofs[component_index],
                    penalty,
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
                parent_nodes = branch_nodes[branch.parent_branch_id]
                child_root = child_nodes[0]
                nearest_parent = min(
                    parent_nodes,
                    key=lambda node: distance(child_root.position, node.position),
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
            branch_dof = nearest_node.dofs[0]

            mass[fruit_dof][fruit_dof] += max(fruit.mass, 1.0e-9)

            stiffness_value = max(fruit.stiffness, 1.0e-6)
            stiffness[fruit_dof][fruit_dof] += stiffness_value
            stiffness[fruit_dof][branch_dof] -= stiffness_value
            stiffness[branch_dof][fruit_dof] -= stiffness_value
            stiffness[branch_dof][branch_dof] += stiffness_value

            damping_value = max(fruit.damping, 0.0)
            damping[fruit_dof][fruit_dof] += damping_value
            damping[fruit_dof][branch_dof] -= damping_value
            damping[branch_dof][fruit_dof] -= damping_value
            damping[branch_dof][branch_dof] += damping_value

        if apply_gravity_prestress:
            from orchard_fem.solvers.static_preload import compute_gravity_axial_forces

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
                for element, axial_force in zip(branch_elements[branch.branch_id], axial_forces[branch.branch_id]):
                    local_geometric = build_local_geometric_stiffness_matrix(axial_force, element.length)
                    global_geometric = transform_to_global(
                        local_geometric,
                        element.transformation_matrix,
                    )
                    _scatter(stiffness, global_geometric, list(element.dofs))

        _apply_rayleigh_damping(model, mass, stiffness, damping, material_lookup)

        excitation_nodes = branch_nodes[model.excitation.target_branch_id]
        excitation_dof = _branch_dof(
            excitation_nodes,
            _resolve_node_index(excitation_nodes, model.excitation.target_node),
            model.excitation.target_component,
        )

        observation_names: list[str] = []
        observation_dofs: list[int] = []
        for observation in model.observations:
            if observation.target_type == "branch":
                nodes = branch_nodes[observation.target_id]
                node_index = _resolve_node_index(nodes, observation.target_node)
                if len(observation.target_components) == 1:
                    observation_names.append(observation.observation_id)
                    observation_dofs.append(
                        _branch_dof(
                            nodes,
                            node_index,
                            observation.target_components[0],
                        )
                    )
                else:
                    for component in observation.target_components:
                        observation_names.append(f"{observation.observation_id}_{component}")
                        observation_dofs.append(
                            _branch_dof(
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
