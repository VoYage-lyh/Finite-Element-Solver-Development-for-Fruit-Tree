"""Topology-based tree input.

Define a tree by branch lengths, diameters, attachment positions and
orientation angles — without writing any 3D coordinates.  Supports curved
branches (3-segment polylines or more) and absolute-distance attachment.

Geometry vs FEM mesh
--------------------
The number of polyline segments (``n_curve_segments``) is *purely geometric*:
it controls the shape of the branch but NOT the FE mesh.  Use
``target_element_length`` or ``num_elements`` for FE discretization,
independent of geometric curvature.

Minimal input::

    {
      "trunk": {"length": 1.2, "diameter_root": 0.08, "diameter_tip": 0.05},
      "branches": [
        {
          "id": "primary_left",
          "parent": "trunk",
          "level": 1,
          "location_distance": 1.14,
          "length": 0.9,
          "diameter_root": 0.04,
          "diameter_tip": 0.02,
          "inclination_deg": 55,
          "azimuth_deg": 180,
          "curve_deg": 18,
          "curve_direction_deg": 0,
          "n_curve_segments": 3
        }
      ]
    }

Conventions
-----------
- ``location_distance`` [m]: arc-length distance from parent branch root to
  child attachment.  Preferred over ``location_s`` (which is the normalized
  0..1 fraction).  If both are given, ``location_distance`` wins.
- ``curve_deg``: total bending angle of the branch (tip direction vs root
  direction).  0 → straight branch.
- ``curve_direction_deg``: direction the branch bends, in the plane
  perpendicular to its initial tangent.  Convention:
    * ``0``   → bend toward −Z (drooping downward — typical for fruit-laden branches)
    * ``180`` → bend toward +Z (upward)
    * ``90`` / ``270`` → bend horizontally (sideways)
- ``n_curve_segments``: number of straight segments approximating the curve
  (default 1 = straight).  More segments → smoother shape, but does NOT
  change the FE mesh density.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# ── Vector helpers ───────────────────────────────────────────────────────────

def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1.0e-12:
        raise ValueError(f"Cannot normalize zero-length vector: {vec}")
    return [x / norm for x in vec]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _scale(vec: list[float], k: float) -> list[float]:
    return [vec[0] * k, vec[1] * k, vec[2] * k]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _rotate_around_axis(vec: list[float], axis: list[float], angle_rad: float) -> list[float]:
    """Rodrigues' rotation formula: rotate ``vec`` around unit ``axis``."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    cross_v = _cross(axis, vec)
    dot_av = _dot(axis, vec)
    return [
        vec[0] * c + cross_v[0] * s + axis[0] * dot_av * one_minus_c,
        vec[1] * c + cross_v[1] * s + axis[1] * dot_av * one_minus_c,
        vec[2] * c + cross_v[2] * s + axis[2] * dot_av * one_minus_c,
    ]


# ── Local frames ─────────────────────────────────────────────────────────────

def _local_perp_frame(tangent: list[float]) -> tuple[list[float], list[float]]:
    """Two unit vectors perpendicular to ``tangent``."""
    world_up = [0.0, 0.0, 1.0]
    if abs(_dot(tangent, world_up)) > 0.99:
        world_up = [1.0, 0.0, 0.0]
    perp1 = _normalize(_cross(tangent, world_up))
    perp2 = _normalize(_cross(tangent, perp1))
    return perp1, perp2


def _branch_initial_direction(
    parent_tangent: list[float],
    inclination_deg: float,
    azimuth_deg: float,
) -> list[float]:
    inc = math.radians(inclination_deg)
    az = math.radians(azimuth_deg)
    perp1, perp2 = _local_perp_frame(parent_tangent)
    perp = _add(_scale(perp1, math.cos(az)), _scale(perp2, math.sin(az)))
    direction = _add(_scale(parent_tangent, math.cos(inc)), _scale(perp, math.sin(inc)))
    return _normalize(direction)


def _bend_axis(tangent: list[float], orientation_rad: float) -> list[float]:
    """Rotation axis for bending a branch in a given direction.

    orientation_rad = 0 → bends the tangent toward −Z (drooping under gravity).
    """
    world_up = [0.0, 0.0, 1.0]
    if abs(_dot(tangent, world_up)) > 0.99:
        primary = [1.0, 0.0, 0.0]
    else:
        primary = _normalize(_cross(world_up, tangent))
    if abs(orientation_rad) > 1.0e-12:
        primary = _rotate_around_axis(primary, tangent, orientation_rad)
    return _normalize(primary)


# ── Polyline geometry ────────────────────────────────────────────────────────

def _polyline_points(
    start: list[float],
    initial_direction: list[float],
    length: float,
    n_segments: int,
    curve_total_rad: float,
    curve_orientation_rad: float,
) -> list[list[float]]:
    """Generate (n_segments + 1) points along a curved branch.

    The branch starts at ``start`` with direction ``initial_direction``.  Each
    segment is rotated by a uniform increment around a bend axis perpendicular
    to the initial tangent, so the tip direction is ``curve_total_rad`` off
    from the root direction.
    """
    if n_segments < 1:
        raise ValueError("n_curve_segments must be >= 1")

    if n_segments == 1 or abs(curve_total_rad) < 1.0e-9:
        return [list(start), _add(start, _scale(initial_direction, length))]

    axis = _bend_axis(initial_direction, curve_orientation_rad)
    seg_length = length / n_segments
    # Segment k (1-indexed) direction is initial_direction rotated by
    # (k − 0.5) * curve_total / n_segments — midpoint-rule integration gives a
    # tip direction that matches curve_total_rad after n_segments steps.
    step = curve_total_rad / n_segments

    points = [list(start)]
    current_pos = list(start)
    for k in range(n_segments):
        # Direction of segment k+1 (1-indexed): rotate by k*step at the start
        # of the segment.  Using start-of-segment angles makes the first
        # segment unrotated (k=0) and the tip direction equals (n-1)*step
        # off from start.  To get total bend = curve_total_rad we use
        # angles[k] = k * curve_total_rad / (n_segments − 1)).
        if n_segments > 1:
            angle = k * curve_total_rad / (n_segments - 1)
        else:
            angle = 0.0
        seg_dir = _rotate_around_axis(initial_direction, axis, angle)
        current_pos = _add(current_pos, _scale(seg_dir, seg_length))
        points.append(list(current_pos))
    return points


def _polyline_length(points: list[list[float]]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        d = _sub(points[i + 1], points[i])
        total += math.sqrt(_dot(d, d))
    return total


def _polyline_position_and_tangent(
    points: list[list[float]],
    distance: float,
) -> tuple[list[float], list[float]]:
    """Find (position, tangent) on a polyline at arc-length ``distance`` from start."""
    if distance <= 0.0:
        tangent = _normalize(_sub(points[1], points[0]))
        return list(points[0]), tangent

    cumulative = 0.0
    for i in range(len(points) - 1):
        seg = _sub(points[i + 1], points[i])
        seg_len = math.sqrt(_dot(seg, seg))
        if cumulative + seg_len >= distance - 1.0e-12:
            t = (distance - cumulative) / seg_len if seg_len > 0 else 0.0
            pos = _add(points[i], _scale(seg, t))
            tangent = _normalize(seg)
            return pos, tangent
        cumulative += seg_len

    # Distance past end → clamp to tip
    tangent = _normalize(_sub(points[-1], points[-2]))
    return list(points[-1]), tangent


# ── Branch-level helpers ─────────────────────────────────────────────────────

def _resolve_attachment_distance(
    branch: dict[str, Any],
    parent_arc_length: float,
) -> float:
    """Return the arc-length distance on the parent where the branch attaches.

    Prefers ``location_distance`` (absolute, in meters); falls back to
    ``location_s`` (normalized 0..1).
    """
    if "location_distance" in branch:
        return float(branch["location_distance"])
    if "location_s" in branch:
        s = float(branch["location_s"])
        if not 0.0 <= s <= 1.0:
            raise ValueError(
                f"Branch '{branch.get('id', '?')}' location_s must be in [0, 1], got {s}"
            )
        return s * parent_arc_length
    raise ValueError(
        f"Branch '{branch.get('id', '?')}' must define location_distance or location_s"
    )


def _build_branch_polyline(
    branch: dict[str, Any],
    attachment: list[float],
    parent_tangent: list[float],
) -> list[list[float]]:
    direction = _branch_initial_direction(
        parent_tangent,
        inclination_deg=float(branch["inclination_deg"]),
        azimuth_deg=float(branch["azimuth_deg"]),
    )
    length = float(branch["length"])
    n_segments = int(branch.get("n_curve_segments", 1))
    curve_total_rad = math.radians(float(branch.get("curve_deg", 0.0)))
    curve_orientation_rad = math.radians(float(branch.get("curve_direction_deg", 0.0)))
    return _polyline_points(
        attachment,
        direction,
        length,
        n_segments,
        curve_total_rad,
        curve_orientation_rad,
    )


# ── Main conversion ──────────────────────────────────────────────────────────

def convert_topology_payload_to_skeleton_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "trunk" not in payload:
        raise ValueError("Topology payload must contain a 'trunk' object")

    trunk = payload["trunk"]
    trunk_origin = list(map(float, trunk.get("origin", [0.0, 0.0, 0.0])))
    trunk_dir0 = _normalize(list(map(float, trunk.get("direction", [0.0, 0.0, 1.0]))))
    trunk_length = float(trunk["length"])
    trunk_n_segments = int(trunk.get("n_curve_segments", 1))
    trunk_curve = math.radians(float(trunk.get("curve_deg", 0.0)))
    trunk_curve_dir = math.radians(float(trunk.get("curve_direction_deg", 0.0)))
    trunk_points = _polyline_points(
        trunk_origin, trunk_dir0, trunk_length,
        trunk_n_segments, trunk_curve, trunk_curve_dir,
    )

    # branch_id → polyline (list of points)
    polylines: dict[str, list[list[float]]] = {}
    trunk_id = str(trunk.get("id", "trunk"))
    polylines[trunk_id] = trunk_points

    skeleton_branches: list[dict[str, Any]] = []
    trunk_entry: dict[str, Any] = {
        "id": trunk_id,
        "parent_branch_id": None,
        "level": 0,
        "points": trunk_points,
        "outer_radius_root": float(trunk["diameter_root"]) / 2.0,
        "outer_radius_tip": float(trunk["diameter_tip"]) / 2.0,
    }
    if "target_element_length" in trunk:
        trunk_entry["target_element_length"] = float(trunk["target_element_length"])
    elif "num_elements" in trunk:
        trunk_entry["num_elements"] = int(trunk["num_elements"])
    skeleton_branches.append(trunk_entry)

    # Topological sort: process a branch only after its parent has been processed
    processed = {trunk_id}
    remaining = list(payload.get("branches", []))

    while remaining:
        progress = False
        for branch in list(remaining):
            parent_id = str(branch["parent"])
            if parent_id not in processed:
                continue

            parent_polyline = polylines[parent_id]
            parent_arc_length = _polyline_length(parent_polyline)
            distance = _resolve_attachment_distance(branch, parent_arc_length)
            if distance < -1.0e-9 or distance > parent_arc_length + 1.0e-9:
                raise ValueError(
                    f"Branch '{branch['id']}' attachment distance {distance} m "
                    f"is outside parent '{parent_id}' length {parent_arc_length} m"
                )
            attachment, parent_tangent = _polyline_position_and_tangent(
                parent_polyline, distance
            )
            branch_points = _build_branch_polyline(branch, attachment, parent_tangent)

            branch_id = str(branch["id"])
            polylines[branch_id] = branch_points

            entry: dict[str, Any] = {
                "id": branch_id,
                "parent_branch_id": parent_id,
                "level": int(branch.get("level", 1)),
                "points": branch_points,
                "outer_radius_root": float(branch["diameter_root"]) / 2.0,
                "outer_radius_tip": float(branch["diameter_tip"]) / 2.0,
            }
            if "target_element_length" in branch:
                entry["target_element_length"] = float(branch["target_element_length"])
            elif "num_elements" in branch:
                entry["num_elements"] = int(branch["num_elements"])
            skeleton_branches.append(entry)

            processed.add(branch_id)
            remaining.remove(branch)
            progress = True

        if not progress:
            unresolved = [str(b.get("id", "?")) for b in remaining]
            raise ValueError(
                f"Could not resolve parent dependencies for branches: {unresolved}. "
                "Check that every 'parent' references an existing branch."
            )

    skeleton: dict[str, Any] = {
        "metadata": dict(payload.get("metadata", {})),
        "branches": skeleton_branches,
    }
    for key in (
        "materials",
        "fruits",
        "fruit_distribution_policy",
        "clamps",
        "joints",
        "excitation",
        "analysis",
        "observations",
    ):
        if key in payload:
            skeleton[key] = payload[key]
    return skeleton


def _resolve_external_refs(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Inline ``materials_ref`` (path to a JSON file with a ``materials`` array).

    Resolved relative to ``base_dir``.  If ``materials`` is already present in
    the payload, the reference is ignored.
    """
    if "materials_ref" in payload and "materials" not in payload:
        ref_path = Path(payload["materials_ref"])
        if not ref_path.is_absolute():
            ref_path = base_dir / ref_path
        ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
        materials = ref_data.get("materials", ref_data)
        payload = dict(payload)
        payload["materials"] = materials
    return payload


def convert_topology_file(input_path: Path, output_path: Path) -> Path:
    """Read topology JSON and write a skeleton JSON ready for ``import-skeleton``."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload = _resolve_external_refs(payload, input_path.parent)
    skeleton = convert_topology_payload_to_skeleton_payload(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def convert_topology_to_orchard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """One-shot: topology → skeleton → full orchard FEM model payload."""
    from orchard_fem.io.skeleton_import import convert_skeleton_payload_to_orchard_payload

    skeleton = convert_topology_payload_to_skeleton_payload(payload)
    return convert_skeleton_payload_to_orchard_payload(skeleton)


def convert_topology_to_orchard_file(input_path: Path, output_path: Path) -> Path:
    """Read topology JSON and write a complete orchard FEM model JSON."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload = _resolve_external_refs(payload, input_path.parent)
    orchard = convert_topology_to_orchard_payload(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(orchard, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
