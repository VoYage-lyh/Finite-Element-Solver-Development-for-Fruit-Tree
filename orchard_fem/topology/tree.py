from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Deque, Dict, Iterable, Optional


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Vec3":
        return Vec3(self.x * factor, self.y * factor, self.z * factor)


def dot(left: Vec3, right: Vec3) -> float:
    return (left.x * right.x) + (left.y * right.y) + (left.z * right.z)


def norm(value: Vec3) -> float:
    return sqrt(dot(value, value))


def distance(left: Vec3, right: Vec3) -> float:
    return norm(left - right)


def normalize(value: Vec3) -> Vec3:
    magnitude = norm(value)
    if magnitude <= 1.0e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return value.scale(1.0 / magnitude)


def lerp(left: Vec3, right: Vec3, alpha: float) -> Vec3:
    return left.scale(1.0 - alpha) + right.scale(alpha)


@dataclass(frozen=True)
class BranchPath:
    start: Vec3
    end: Vec3

    def length(self) -> float:
        return distance(self.start, self.end)

    def point_at(self, station: float) -> Vec3:
        return lerp(self.start, self.end, max(0.0, min(1.0, station)))

    def direction(self) -> Vec3:
        return normalize(self.end - self.start)


@dataclass(frozen=True)
class ObservationPoint:
    observation_id: str
    target_type: str
    target_id: str
    target_node: str = "tip"
    target_component: str = "ux"


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
        return [branch_id for branch_id, node in self._nodes.items() if node.parent_branch_id is None]

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
            topology.add_branch(record.branch_id, record.parent_branch_id, record.level, record.path)
        topology.rebuild_child_links()
        return topology
