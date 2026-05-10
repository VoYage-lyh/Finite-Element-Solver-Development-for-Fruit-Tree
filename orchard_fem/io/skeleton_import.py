from __future__ import annotations

import json
from math import ceil, sqrt
from pathlib import Path
from typing import Any


DEFAULT_MATERIALS = [
    {
        "id": "pith_default",
        "tissue": "pith",
        "model": "linear",
        "density": 550.0,
        "youngs_modulus": 1.0e9,
        "poisson_ratio": 0.35,
        "damping_ratio": 0.02,
    },
    {
        "id": "xylem_default",
        "tissue": "xylem",
        "model": "linear",
        "density": 750.0,
        "youngs_modulus": 1.0e10,
        "poisson_ratio": 0.30,
        "damping_ratio": 0.01,
    },
    {
        "id": "phloem_default",
        "tissue": "phloem",
        "model": "linear",
        "density": 620.0,
        "youngs_modulus": 2.0e9,
        "poisson_ratio": 0.35,
        "damping_ratio": 0.02,
    },
]


def _point(value: Any, *, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must be a 3-value coordinate list")
    return [float(component) for component in value]


def _polyline_length(points: list[list[float]]) -> float:
    total = 0.0
    for index in range(len(points) - 1):
        total += sqrt(
            sum((points[index + 1][axis] - points[index][axis]) ** 2 for axis in range(3))
        )
    return total


def _branch_points(branch: dict[str, Any]) -> list[list[float]]:
    if "points" in branch:
        points = [
            _point(point, field_name=f"branches[{branch.get('id', '?')}].points[]")
            for point in branch["points"]
        ]
        if len(points) < 2:
            raise ValueError(f"Branch '{branch.get('id', '?')}' points must contain at least two points")
        return points
    return [
        _point(branch["start"], field_name=f"branches[{branch.get('id', '?')}].start"),
        _point(branch["end"], field_name=f"branches[{branch.get('id', '?')}].end"),
    ]


def _discretization(branch: dict[str, Any], points: list[list[float]]) -> dict[str, Any]:
    if "discretization" in branch:
        return dict(branch["discretization"])
    if "num_elements" in branch:
        return {"num_elements": max(int(branch["num_elements"]), 1), "hotspot": False}
    target_element_length = branch.get("target_element_length")
    if target_element_length is not None:
        length = _polyline_length(points)
        return {
            "num_elements": max(int(ceil(length / max(float(target_element_length), 1.0e-12))), 1),
            "hotspot": False,
        }
    return {"num_elements": 4, "hotspot": False}


def _stations(branch: dict[str, Any]) -> list[dict[str, Any]]:
    if "stations" in branch:
        return [dict(station) for station in branch["stations"]]

    if "outer_radius" in branch:
        radius_root = float(branch["outer_radius"])
        radius_tip = radius_root
    else:
        root_value = branch.get("outer_radius_root", branch.get("radius_root"))
        tip_value = branch.get("outer_radius_tip", branch.get("radius_tip"))
        if root_value is None or tip_value is None:
            raise ValueError(
                f"Branch '{branch.get('id', '?')}' must define stations, outer_radius, "
                "or root/tip radius values"
            )
        radius_root = float(root_value)
        radius_tip = float(tip_value)

    return [
        {"s": 0.0, "shorthand": "circular", "outer_radius": radius_root},
        {"s": 1.0, "shorthand": "circular", "outer_radius": radius_tip},
    ]


def _convert_branch(branch: dict[str, Any]) -> dict[str, Any]:
    points = _branch_points(branch)
    converted = {
        "id": str(branch["id"]),
        "parent_branch_id": branch.get("parent_branch_id"),
        "level": int(branch.get("level", 0)),
        "start": points[0],
        "end": points[-1],
        "discretization": _discretization(branch, points),
        "stations": _stations(branch),
    }
    if len(points) > 2:
        converted["points"] = points
    return converted


def _infer_terminal_branch_ids(branches: list[dict[str, Any]]) -> set[str]:
    """Return branch ids that have no children (leaf nodes in the branch tree)."""
    parent_ids = {
        str(branch["parent_branch_id"])
        for branch in branches
        if branch.get("parent_branch_id") is not None
    }
    return {str(branch["id"]) for branch in branches} - parent_ids


def convert_skeleton_payload_to_orchard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "branches" not in payload:
        raise ValueError("Skeleton payload must contain a branches array")
    if "analysis" not in payload:
        raise ValueError("Skeleton payload must contain an analysis block")
    if "excitation" not in payload:
        raise ValueError("Skeleton payload must contain an excitation block")

    raw_branches = payload["branches"]
    terminal_ids = _infer_terminal_branch_ids(raw_branches)
    converted_branches = []
    for branch in raw_branches:
        converted = _convert_branch(branch)
        converted["is_terminal"] = str(branch["id"]) in terminal_ids
        converted_branches.append(converted)

    result: dict[str, Any] = {
        "metadata": dict(payload.get("metadata", {})),
        "materials": [dict(material) for material in payload.get("materials", DEFAULT_MATERIALS)],
        "branches": converted_branches,
        "joints": [dict(joint) for joint in payload.get("joints", [])],
        "fruits": [dict(fruit) for fruit in payload.get("fruits", [])],
        "clamps": [dict(clamp) for clamp in payload.get("clamps", [])],
        "excitation": dict(payload["excitation"]),
        "analysis": dict(payload["analysis"]),
        "observations": [dict(observation) for observation in payload.get("observations", [])],
    }

    if "fruit_distribution_policy" in payload:
        result["fruit_distribution_policy"] = dict(payload["fruit_distribution_policy"])

    return result


def convert_skeleton_file(input_path: Path, output_path: Path) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    converted = convert_skeleton_payload_to_orchard_payload(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
