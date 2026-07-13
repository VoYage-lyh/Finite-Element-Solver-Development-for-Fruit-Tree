"""TreeQSM / point-cloud skeleton adapter.

This module defines the interface that upstream 3D-scan processing pipelines
must satisfy to produce an Orchard FEM skeleton JSON.

Upstream responsibility (image recognition / point-cloud code):
    - Extract branch centerline polylines from the 3D scan.
    - Assign parent_branch_id relationships (tree graph).
    - Measure outer radius at root and tip of each branch.
    - Assign branch levels (0 = trunk, 1 = primary scaffold, 2 = secondary, …).

This module:
    - Defines the expected input schema (`TreeQSMBranch`, `TreeQSMPayload`).
    - Converts a TreeQSM-style dict payload to the skeleton JSON format accepted
      by `convert_skeleton_payload_to_orchard_payload`.
    - Provides a direct file-to-file converter `convert_treeqsm_file`.

Expected TreeQSM JSON schema
----------------------------
{
  "metadata": {"name": "...", "cultivar": "..."},        // optional
  "branches": [
    {
      "id":               "trunk",        // unique string identifier
      "parent_branch_id": null,           // null for trunk, string otherwise
      "level":            0,              // 0=trunk, 1=primary, 2=secondary, …
      "points": [                         // ≥2 3D points along centerline [m]
        [x0, y0, z0],
        [x1, y1, z1],
        ...
      ],
      "outer_radius_root": 0.04,          // [m] radius at branch root
      "outer_radius_tip":  0.02,          // [m] radius at branch tip
      "target_element_length": 0.15       // [m] optional, controls FEM mesh density
    },
    ...
  ],
  // Optional: override default materials
  "materials": [ ... ],
  // Optional: place fruit distribution policy (see FruitDistributionPolicy)
  "fruit_distribution_policy": {
    "total_fruit_count": 200,
    "seed": 2026
  },
  // Required: excitation definition
  "excitation": {
    "kind": "harmonic_force",
    "target_branch_id": "trunk",
    "target_node": "tip",
    "target_component": "ux",
    "amplitude": 1.0,
    "phase_degrees": 0.0,
    "driving_frequency_hz": 5.0
  },
  // Required: analysis settings
  "analysis": {
    "mode": "frequency_response",
    "solver_backend": "fenicsx",
    "frequency_start_hz": 0.5,
    "frequency_end_hz": 30.0,
    "frequency_steps": 60,
    "include_gravity_prestress": true,
    "output_csv": "build/freq_response.csv"
  },
  // Optional: observation points
  "observations": [ ... ],
  // Optional: ground clamps
  "clamps": [ ... ]
}

Branch level assignment algorithm (reference implementation for upstream code)
-------------------------------------------------------------------------------
Given the branch parent-child graph rooted at the trunk:

    def assign_levels(branches):
        id_to_branch = {b["id"]: b for b in branches}
        root = next(b for b in branches if b["parent_branch_id"] is None)
        root["level"] = 0
        queue = [root["id"]]
        while queue:
            current_id = queue.pop(0)
            current_level = id_to_branch[current_id]["level"]
            for b in branches:
                if b["parent_branch_id"] == current_id:
                    b["level"] = current_level + 1
                    queue.append(b["id"])

This is provided for reference only; the upstream pipeline is responsible for
producing correctly assigned levels before calling this adapter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchard_fem.io.skeleton_import import convert_skeleton_payload_to_orchard_payload


def convert_treeqsm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a TreeQSM-format payload to Orchard skeleton JSON.

    The TreeQSM format is treated as identical to the skeleton format accepted
    by `convert_skeleton_payload_to_orchard_payload`, so this function is a
    thin pass-through that validates the required keys and delegates.

    Raises
    ------
    ValueError
        If required keys are missing or branch geometry is invalid.
    """
    required = ("branches", "excitation", "analysis")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"TreeQSM payload is missing required keys: {missing}"
        )

    for branch in payload["branches"]:
        if "id" not in branch:
            raise ValueError("Every branch entry must have an 'id' field.")
        if "points" not in branch and not ("start" in branch and "end" in branch):
            raise ValueError(
                f"Branch '{branch.get('id')}' must have either 'points' or "
                "'start'/'end' coordinates."
            )
        if "outer_radius_root" not in branch and "outer_radius" not in branch:
            raise ValueError(
                f"Branch '{branch.get('id')}' must have 'outer_radius_root' "
                "and 'outer_radius_tip' (or 'outer_radius' for uniform sections)."
            )

    return convert_skeleton_payload_to_orchard_payload(payload)


def convert_treeqsm_file(input_path: Path, output_path: Path) -> Path:
    """Read a TreeQSM JSON file, convert, and write Orchard skeleton JSON."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    converted = convert_treeqsm_payload(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
