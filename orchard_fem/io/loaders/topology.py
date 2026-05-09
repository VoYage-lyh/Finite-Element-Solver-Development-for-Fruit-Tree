from __future__ import annotations

from typing import Any

from orchard_fem.topology import BranchPath, TreeTopology, Vec3


def parse_branch_path_payload(branch: dict[str, Any]) -> BranchPath:
    if "points" in branch:
        points = [
            Vec3(*[float(value) for value in point])
            for point in branch["points"]
        ]
        if len(points) < 2:
            raise ValueError(f"Branch '{branch['id']}' points must contain at least two points")
        return BranchPath(
            start=points[0],
            end=points[-1],
            control_points=tuple(points[1:-1]),
        )

    return BranchPath(
        start=Vec3(*[float(value) for value in branch["start"]]),
        end=Vec3(*[float(value) for value in branch["end"]]),
    )


def build_topology_from_model_payload(payload: dict[str, Any]) -> TreeTopology:
    topology = TreeTopology()
    for branch in payload.get("branches", []):
        topology.add_branch(
            branch_id=str(branch["id"]),
            parent_branch_id=branch.get("parent_branch_id"),
            level=int(branch["level"]),
            path=parse_branch_path_payload(branch),
        )

    topology.rebuild_child_links()
    valid, message = topology.validate()
    if not valid:
        raise ValueError(f"Invalid topology extracted from orchard model: {message}")
    return topology
