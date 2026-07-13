"""Plain data containers shared across the orchard_vision pipeline stages.

All pixel coordinates are ``(row, col)`` integer arrays (image convention, row
grows downward). Metric conversion happens only in :mod:`orchard_vision.lift_3d`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _polyline_length_px(pixels: np.ndarray) -> float:
    """Arc length of an ``(N, 2)`` pixel polyline."""
    if len(pixels) < 2:
        return 0.0
    deltas = np.diff(pixels.astype(float), axis=0)
    return float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())


@dataclass
class GraphEdge:
    """One chain of the pixel skeleton connecting two graph nodes.

    ``pixels`` is ordered ``(row, col)`` and includes *both* end-node pixels, so
    consecutive edges concatenate into a continuous branch polyline.
    ``radius_px`` is the distance-transform half-width sampled at each pixel.
    """

    u: int
    v: int
    pixels: np.ndarray  # (N, 2) int, (row, col)
    radius_px: np.ndarray  # (N,) float

    @property
    def length_px(self) -> float:
        return _polyline_length_px(self.pixels)


@dataclass
class SkeletonGraph:
    """Undirected graph traced from a 1-pixel skeleton."""

    nodes: np.ndarray  # (M, 2) int, (row, col) of node pixels
    node_kind: list[str]  # 'endpoint' | 'junction', aligned with ``nodes``
    edges: list[GraphEdge]


@dataclass
class Branch:
    """A merged, order-labelled branch (trunk / primary / secondary / …).

    ``pixels`` runs from the attached (parent-side) end toward the free tip, so
    ``radius_px[0]`` is the root radius and ``radius_px[-1]`` the tip radius.
    ``level`` is the branch order: 0 = trunk, 1 = primary, 2 = secondary, …
    """

    id: str
    parent_id: str | None
    level: int
    pixels: np.ndarray  # (N, 2) int, (row, col), root-side → tip
    radius_px: np.ndarray  # (N,) float

    @property
    def length_px(self) -> float:
        return _polyline_length_px(self.pixels)
