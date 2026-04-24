from __future__ import annotations

from orchard_fem.discretization.types import COMPONENT_INDEX


class DOFManager:
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


def resolve_node_index(nodes, target_node: str) -> int:
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


def branch_dof(nodes, node_index: int, component: str) -> int:
    try:
        component_index = COMPONENT_INDEX[component]
    except KeyError as exc:
        raise ValueError(f"Unsupported DOF component: {component}") from exc
    return nodes[node_index].dofs[component_index]


__all__ = ["DOFManager", "branch_dof", "resolve_node_index"]
