"""Ordered branches → orchard skeleton-import JSON.

The emitted payload is the minimal schema accepted by
``python -m orchard_fem import-skeleton`` (see
:mod:`orchard_fem.io.skeleton_import`): a ``metadata`` block plus a ``branches``
list of ``{id, parent_branch_id, level, points, outer_radius_root/tip}``. The
importer fills in materials, joints, clamps, excitation, analysis and
observations, so this stage only has to produce clean geometry and topology.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from orchard_vision.lift_3d import MonocularPlanarLift
from orchard_vision.types import Branch

# Radius floor so distance-transform tips (~0.5 px) never export a zero section.
_MIN_RADIUS_M = 0.002


def _end_radius(radius_m: np.ndarray, *, at_root: bool, window_fraction: float = 0.15) -> float:
    """Robust root/tip radius = median over the end window.

    The single boundary pixel is unreliable: a branch end that sits on a junction
    inherits the inflated distance-transform radius where masks merge, so we take
    a median over a short window instead of the endpoint value.
    """
    window = max(2, int(round(len(radius_m) * window_fraction)))
    segment = radius_m[:window] if at_root else radius_m[-window:]
    return float(np.median(segment))


def _resample(points_xyz: np.ndarray, radius_m: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Arc-length downsample a polyline to ``max_points``, keeping both ends."""
    count = len(points_xyz)
    if count <= max_points:
        return points_xyz, radius_m
    segment = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    if arc[-1] <= 0:
        return points_xyz[[0, -1]], radius_m[[0, -1]]
    targets = np.linspace(0.0, arc[-1], max_points)
    indices = np.clip(np.searchsorted(arc, targets), 0, count - 1)
    indices = np.unique(np.concatenate([[0], indices, [count - 1]]))
    return points_xyz[indices], radius_m[indices]


def branch_to_dict(
    branch: Branch,
    lift: MonocularPlanarLift,
    *,
    max_points_per_branch: int,
) -> dict[str, Any]:
    radius_full = lift.branch_radius_m(branch)
    xyz, _ = _resample(lift.branch_xyz(branch), radius_full, max_points_per_branch)
    root_radius = max(_end_radius(radius_full, at_root=True), _MIN_RADIUS_M)
    # A branch never thickens toward its tip; this also damps junction inflation.
    tip_radius = max(min(_end_radius(radius_full, at_root=False), root_radius), _MIN_RADIUS_M)
    return {
        "id": branch.id,
        "parent_branch_id": branch.parent_id,
        "level": int(branch.level),
        "points": [[float(x), float(y), float(z)] for x, y, z in xyz],
        "outer_radius_root": root_radius,
        "outer_radius_tip": tip_radius,
    }


def branches_to_skeleton_payload(
    branches: list[Branch],
    lift: MonocularPlanarLift,
    *,
    model_name: str,
    max_points_per_branch: int = 16,
) -> dict[str, Any]:
    """Assemble the full skeleton-import payload from ordered branches."""
    out_branches = [
        branch_to_dict(branch, lift, max_points_per_branch=max_points_per_branch)
        for branch in branches
    ]
    _snap_roots_to_parents(out_branches)
    return {
        "metadata": {
            "name": model_name,
            "source": "orchard_vision.monocular",
            "notes": "Auto-extracted from a single RGB photo; radii/scale are approximate.",
        },
        "branches": out_branches,
    }


def _snap_roots_to_parents(out_branches: list[dict[str, Any]]) -> None:
    """Move each child's first point onto the nearest vertex of its parent's
    (independently down-sampled) polyline, so branches join exactly in the exported
    model instead of leaving a resampling gap the solver would bridge as a float."""
    by_id = {branch["id"]: branch for branch in out_branches}
    for branch in out_branches:
        parent = by_id.get(branch["parent_branch_id"])
        if parent is None:
            continue
        parent_points = np.array(parent["points"])
        root = np.array(branch["points"][0])
        nearest = int(np.argmin(np.linalg.norm(parent_points - root, axis=1)))
        branch["points"][0] = parent_points[nearest].tolist()
