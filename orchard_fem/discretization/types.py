from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
