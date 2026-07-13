"""Monocular 2D → 3D lift: pixel branch polylines → metric coordinates.

No depth sensor is assumed (the field rig uses an ordinary camera), so the tree
is placed in a single vertical plane matching the solver's axes:

* image-x (col)  → world-x  (horizontal)
* image-up (row) → world-z  (vertical, the solver's gravity/``uz`` axis)
* out-of-plane   → world-y = 0

Metric scale comes from one real-world reference: a **measured trunk base
diameter** (preferred — pins every radius, hence the section stiffness that sets
the resonance frequencies) or, failing that, an assumed tree height. The planar
(``y = 0``) assumption is the remaining fidelity gap; a depth map or multi-view
capture would recover the out-of-plane geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orchard_vision.types import Branch


@dataclass
class MonocularPlanarLift:
    """Pixel → metre mapping with the trunk base at the world origin."""

    meters_per_pixel: float
    base_row: int  # image row that maps to world z = 0
    base_col: int  # image col that maps to world x = 0

    @classmethod
    def from_tree_height(
        cls,
        branches: list[Branch],
        tree_height_m: float,
        base_rc: tuple[int, int],
    ) -> "MonocularPlanarLift":
        """Derive the scale from the pixel span of the whole tree."""
        rows = np.concatenate([branch.pixels[:, 0] for branch in branches])
        pixel_height = float(rows.max() - rows.min())
        meters_per_pixel = tree_height_m / max(pixel_height, 1.0)
        return cls(meters_per_pixel, int(base_rc[0]), int(base_rc[1]))

    @classmethod
    def from_trunk_diameter(
        cls,
        branches: list[Branch],
        trunk_diameter_m: float,
        base_rc: tuple[int, int],
    ) -> "MonocularPlanarLift":
        """Derive the scale from a **measured** trunk base diameter.

        Far more reliable than a guessed tree height: one calliper/tape reading at
        the base pins the pixel→metre scale (and thus every branch radius, so the
        section stiffness that drives the resonance frequencies is metric-correct).
        """
        trunk = next((b for b in branches if b.level == 0), branches[0])
        # Calibrate the *base* radius (what a tape measures, and what exports as
        # ``outer_radius_root``): median over the base 15% of the trunk centreline.
        window = max(2, round(len(trunk.radius_px) * 0.15))
        trunk_radius_px = float(np.median(trunk.radius_px[:window]))
        meters_per_pixel = trunk_diameter_m / (2.0 * max(trunk_radius_px, 1.0))
        return cls(meters_per_pixel, int(base_rc[0]), int(base_rc[1]))

    def branch_xyz(self, branch: Branch) -> np.ndarray:
        """Return the branch centreline as ``(N, 3)`` metric ``[x, y, z]``."""
        rows = branch.pixels[:, 0].astype(float)
        cols = branch.pixels[:, 1].astype(float)
        x = (cols - self.base_col) * self.meters_per_pixel
        z = (self.base_row - rows) * self.meters_per_pixel  # image-down → world-up
        y = np.zeros_like(x)
        return np.column_stack([x, y, z])

    def branch_radius_m(self, branch: Branch) -> np.ndarray:
        return branch.radius_px.astype(float) * self.meters_per_pixel
