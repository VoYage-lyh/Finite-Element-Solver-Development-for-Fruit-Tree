from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Optional

from orchard_fem.topology.paths import BranchPath


@dataclass
class TopologyNode:
    branch_id: str
    parent_branch_id: Optional[str]
    level: int
    path: BranchPath
    child_branch_ids: list[str] = field(default_factory=list)


class TreeTopology:
    def __init__(self) -> None:
        self._nodes: Dict[str, TopologyNode] = {}

    @property
    def nodes(self) -> Dict[str, TopologyNode]:
        return self._nodes

    def add_branch(
        self,
        branch_id: str,
        parent_branch_id: Optional[str],
        level: int,
        path: BranchPath,
    ) -> None:
        self._nodes[branch_id] = TopologyNode(
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            level=level,
            path=path,
        )

    def rebuild_child_links(self) -> None:
        for node in self._nodes.values():
            node.child_branch_ids.clear()

        for branch_id, node in self._nodes.items():
            if node.parent_branch_id is None:
                continue
            parent = self._nodes.get(node.parent_branch_id)
            if parent is not None:
                parent.child_branch_ids.append(branch_id)

    def contains(self, branch_id: str) -> bool:
        return branch_id in self._nodes

    def require_node(self, branch_id: str) -> TopologyNode:
        try:
            return self._nodes[branch_id]
        except KeyError as exc:
            raise KeyError(f"Topology node not found for branch: {branch_id}") from exc

    def roots(self) -> list[str]:
        return [
            branch_id
            for branch_id, node in self._nodes.items()
            if node.parent_branch_id is None
        ]

    def traversal_order(self) -> list[str]:
        queue: Deque[str] = deque(self.roots())
        visited: set[str] = set()
        order: list[str] = []

        while queue:
            branch_id = queue.popleft()
            if branch_id in visited:
                continue

            visited.add(branch_id)
            order.append(branch_id)
            queue.extend(self.require_node(branch_id).child_branch_ids)

        return order

    def validate(self) -> tuple[bool, str]:
        if not self._nodes:
            return False, "Tree topology is empty"

        roots = self.roots()
        if not roots:
            return False, "Tree topology has no root branch"

        for branch_id, node in self._nodes.items():
            if node.parent_branch_id is None:
                continue
            if node.parent_branch_id == branch_id:
                return False, f"Branch cannot be its own parent: {branch_id}"
            if node.parent_branch_id not in self._nodes:
                return False, (
                    f"Missing parent branch '{node.parent_branch_id}' for branch '{branch_id}'"
                )

        states = {branch_id: "unvisited" for branch_id in self._nodes}

        def visit(branch_id: str) -> tuple[bool, str]:
            states[branch_id] = "visiting"
            for child_branch_id in self.require_node(branch_id).child_branch_ids:
                if states[child_branch_id] == "visiting":
                    return False, f"Cycle detected around branch: {child_branch_id}"
                if states[child_branch_id] == "unvisited":
                    ok, message = visit(child_branch_id)
                    if not ok:
                        return False, message
            states[branch_id] = "visited"
            return True, ""

        for root in roots:
            if states[root] == "unvisited":
                ok, message = visit(root)
                if not ok:
                    return False, message

        return True, ""

    @classmethod
    def from_branch_records(cls, records: Iterable[TopologyNode]) -> "TreeTopology":
        topology = cls()
        for record in records:
            topology.add_branch(
                record.branch_id,
                record.parent_branch_id,
                record.level,
                record.path,
            )
        topology.rebuild_child_links()
        return topology
