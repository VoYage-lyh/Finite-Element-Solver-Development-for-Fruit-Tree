from __future__ import annotations

import json
from math import ceil, sqrt
from pathlib import Path
from typing import Any


DEFAULT_ANALYSIS = {
    "mode": "frequency_response",
    "solver_backend": "fenicsx",
    "frequency_start_hz": 1.0,
    "frequency_end_hz": 25.0,
    "frequency_steps": 25,
    "rayleigh_alpha": 0.0,
    "rayleigh_beta": 1.0e-4,
    "include_gravity_prestress": True,
    "output_csv": "frequency_response.csv",
}

DEFAULT_FRUIT_POLICY = {
    "total_fruit_count": 20,
    "seed": 2026,
    "include_terminal_primary": False,
    "detachment_displacement_m": 0.010,
    "attachment_damping_ratio": 0.05,
    "attachment_component": "uz",
}

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
    parent_ids = {
        str(branch["parent_branch_id"])
        for branch in branches
        if branch.get("parent_branch_id") is not None
    }
    return {str(branch["id"]) for branch in branches} - parent_ids


def _root_branch_id(branches: list[dict[str, Any]]) -> str:
    for branch in branches:
        if branch.get("parent_branch_id") is None:
            return str(branch["id"])
    return str(branches[0]["id"])


def _auto_joints(_branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Branch-to-branch connections are handled automatically by the FEniCSx assembly:
    - Coincident endpoints → mesh point deduplication (shared DOFs, rigid)
    - Non-coincident endpoints → dolfinx_mpc rigid constraint (auto-detected)
    No explicit joint entries are needed unless the user wants a spring / nonlinear law.
    """
    return []


def _auto_clamps(root_id: str) -> list[dict[str, Any]]:
    return [{
        "branch_id": root_id,
        "support_stiffness": 40000.0,
        "support_damping": 35.0,
    }]


def _auto_excitation(root_id: str) -> dict[str, Any]:
    return {
        "kind": "harmonic_force",
        "target_branch_id": root_id,
        "target_node": "tip",
        "target_component": "ux",
        "amplitude": 10.0,
        "phase_degrees": 0.0,
        "driving_frequency_hz": 5.0,
    }


def _auto_observations(converted_branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Root / mid / tip observation at every branch — covers full structural response."""
    observations = []
    for branch in converted_branches:
        for position in ("root", "mid", "tip"):
            observations.append({
                "id": f"obs_{branch['id']}_{position}",
                "target_type": "branch",
                "target_id": branch["id"],
                "target_node": position,
                "target_components": ["ux", "uy", "uz"],
            })
    return observations


def convert_skeleton_payload_to_orchard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "branches" not in payload:
        raise ValueError("Skeleton payload must contain a branches array")

    raw_branches = payload["branches"]
    terminal_ids = _infer_terminal_branch_ids(raw_branches)
    root_id = _root_branch_id(raw_branches)

    converted_branches = []
    for branch in raw_branches:
        converted = _convert_branch(branch)
        converted["is_terminal"] = str(branch["id"]) in terminal_ids
        converted_branches.append(converted)

    # joints: use provided or auto-generate from topology
    joints = payload.get("joints")
    if joints is None:
        joints = _auto_joints(raw_branches)
    else:
        joints = [dict(j) for j in joints]

    # clamps: use provided or auto-generate for root branch
    clamps = payload.get("clamps")
    if clamps is None:
        clamps = _auto_clamps(root_id)
    else:
        clamps = [dict(c) for c in clamps]

    # excitation: use provided or auto-generate at root tip
    excitation = payload.get("excitation")
    if excitation is None:
        excitation = _auto_excitation(root_id)
    else:
        excitation = dict(excitation)

    # analysis: use provided or use defaults
    analysis = payload.get("analysis")
    if analysis is None:
        analysis = dict(DEFAULT_ANALYSIS)
    else:
        # fill missing keys with defaults
        merged = dict(DEFAULT_ANALYSIS)
        merged.update(analysis)
        analysis = merged

    # observations: use provided or auto-generate at all branch tips
    observations = payload.get("observations")
    if observations is None:
        observations = _auto_observations(converted_branches)
    else:
        observations = [dict(o) for o in observations]

    # fruits / fruit_distribution_policy
    fruits = [dict(f) for f in payload.get("fruits", [])]
    fruit_policy = payload.get("fruit_distribution_policy")
    if not fruits and fruit_policy is None:
        # Only add default policy when fruit-bearing branches exist (level >= 2,
        # or level == 1 for terminal primary branches).
        max_level = max((int(b.get("level", 0)) for b in raw_branches), default=0)
        if max_level >= 2:
            fruit_policy = dict(DEFAULT_FRUIT_POLICY)
        elif max_level >= 1:
            policy = dict(DEFAULT_FRUIT_POLICY)
            policy["include_terminal_primary"] = True
            fruit_policy = policy

    result: dict[str, Any] = {
        "metadata": dict(payload.get("metadata", {})),
        "materials": [dict(m) for m in payload.get("materials", DEFAULT_MATERIALS)],
        "branches": converted_branches,
        "joints": joints,
        "fruits": fruits,
        "clamps": clamps,
        "excitation": excitation,
        "analysis": analysis,
        "observations": observations,
    }
    if fruit_policy is not None:
        result["fruit_distribution_policy"] = dict(fruit_policy)

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
