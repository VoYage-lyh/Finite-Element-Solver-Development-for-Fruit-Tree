from __future__ import annotations

from typing import Any

from orchard_fem.topology import BranchPath, TreeTopology, Vec3


def build_topology_from_model_payload(payload: dict[str, Any]) -> TreeTopology:
    topology = TreeTopology()
    for branch in payload.get("branches", []):
        topology.add_branch(
            branch_id=str(branch["id"]),
            parent_branch_id=branch.get("parent_branch_id"),
            level=int(branch["level"]),
            path=BranchPath(
                start=Vec3(*[float(value) for value in branch["start"]]),
                end=Vec3(*[float(value) for value in branch["end"]]),
            ),
        )

    topology.rebuild_child_links()
    valid, message = topology.validate()
    if not valid:
        raise ValueError(f"Invalid topology extracted from orchard model: {message}")
    return topology
