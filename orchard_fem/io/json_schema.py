from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchard_fem.topology.tree import BranchPath, TreeTopology, Vec3


REQUIRED_TOP_LEVEL_KEYS = (
    "materials",
    "branches",
    "excitation",
    "analysis",
)


def load_legacy_model(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Legacy orchard model is missing required keys: {', '.join(missing)}")

    return payload


def build_topology_from_legacy_model(payload: dict[str, Any]) -> TreeTopology:
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
        raise ValueError(f"Invalid topology extracted from legacy model: {message}")
    return topology
