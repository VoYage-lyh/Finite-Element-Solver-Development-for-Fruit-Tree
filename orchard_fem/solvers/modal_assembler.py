from __future__ import annotations

from dataclasses import dataclass
from math import fabs

from orchard_fem.elements.beam_formulation import BeamElementProperties, build_global_element_matrices
from orchard_fem.materials.base import build_material_lookup, evaluate_branch_section_state
from orchard_fem.model import OrchardModel
from orchard_fem.topology.tree import distance


Matrix = list[list[float]]
CONSTRAINT_PENALTY = 1.0e10
COMPONENT_LABELS = ("ux", "uy", "uz", "rx", "ry", "rz")


@dataclass(frozen=True)
class BranchNodeState:
    label_prefix: str
    position: object
    station: float
    dofs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ModalAssemblyResult:
    stiffness_matrix: Matrix
    mass_matrix: Matrix
    dof_labels: list[str]
    branch_nodes: dict[str, list[BranchNodeState]]
    fruit_dofs: dict[str, int]


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


class OrchardModalAssembler:
    def assemble(self, model: OrchardModel) -> ModalAssemblyResult:
        dof_manager = _DOFManager()
        branch_nodes: dict[str, list[BranchNodeState]] = {}
        fruit_dofs: dict[str, int] = {}

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
        material_lookup = build_material_lookup(model.materials)

        for branch in model.branches:
            nodes = branch_nodes[branch.branch_id]
            for element_index in range(len(nodes) - 1):
                first = nodes[element_index]
                second = nodes[element_index + 1]
                first_state = evaluate_branch_section_state(branch, material_lookup, first.station)
                second_state = evaluate_branch_section_state(branch, material_lookup, second.station)

                area = 0.5 * (first_state.area + second_state.area)
                iy = 0.5 * (first_state.ix + second_state.ix)
                iz = 0.5 * (first_state.iy + second_state.iy)
                polar_moment = max(0.5 * (first_state.polar_moment + second_state.polar_moment), iy + iz)
                mass_per_length = 0.5 * (first_state.mass_per_length + second_state.mass_per_length)
                element_length = distance(first.position, second.position)

                properties = BeamElementProperties(
                    youngs_modulus=0.5 * (first_state.effective_youngs_modulus + second_state.effective_youngs_modulus),
                    shear_modulus=0.5 * (first_state.effective_shear_modulus + second_state.effective_shear_modulus),
                    area=area,
                    iy=iy,
                    iz=iz,
                    torsion_constant=polar_moment,
                    density=mass_per_length / area if area > 0.0 else 0.0,
                    length=element_length,
                )

                global_stiffness, global_mass = build_global_element_matrices(
                    properties=properties,
                    start=first.position,
                    end=second.position,
                )
                element_dofs = list(first.dofs) + list(second.dofs)
                _scatter(stiffness, global_stiffness, element_dofs)
                _scatter(mass, global_mass, element_dofs)

        for branch in model.branches:
            if branch.parent_branch_id is None:
                continue

            child_nodes = branch_nodes[branch.branch_id]
            parent_nodes = branch_nodes[branch.parent_branch_id]
            child_root = child_nodes[0]
            nearest_parent = min(parent_nodes, key=lambda node: distance(child_root.position, node.position))

            penalty = CONSTRAINT_PENALTY
            matching_joint = next((joint for joint in model.joints if joint.child_branch_id == branch.branch_id), None)
            if matching_joint is not None:
                penalty *= max(matching_joint.linear_stiffness_scale, 1.0e-6)

            for component_index in range(6):
                _add_pair_penalty(
                    stiffness,
                    child_root.dofs[component_index],
                    nearest_parent.dofs[component_index],
                    penalty,
                )

        for clamp in model.clamps:
            root = branch_nodes[clamp.branch_id][0]
            for dof in root.dofs:
                stiffness[dof][dof] += CONSTRAINT_PENALTY

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

        return ModalAssemblyResult(
            stiffness_matrix=stiffness,
            mass_matrix=mass,
            dof_labels=dof_manager.labels,
            branch_nodes=branch_nodes,
            fruit_dofs=fruit_dofs,
        )
