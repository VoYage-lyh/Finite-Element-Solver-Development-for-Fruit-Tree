"""Per-branch instance masks from the wood mask + ordered skeletons.

The ordering stage gives each branch a *centreline*; this turns the binary wood
mask into one labelled region per branch by assigning every wood pixel to its
nearest branch centreline (a watershed seeded from the skeleton). The result is
instance segmentation: trunk, each primary, each secondary as its own region.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from orchard_vision.types import Branch


def branch_instances(mask: np.ndarray, branches: list[Branch]) -> np.ndarray:
    """Return a label image (0 = background, i = ``branches[i-1]``).

    Each ``True`` pixel of ``mask`` is labelled with the branch whose skeleton
    centreline is nearest (Euclidean), so the wood mask is partitioned into one
    region per branch.
    """
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not branches:
        return labels

    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, branch in enumerate(branches, start=1):
        rows = np.clip(branch.pixels[:, 0], 0, mask.shape[0] - 1)
        cols = np.clip(branch.pixels[:, 1], 0, mask.shape[1] - 1)
        markers[rows, cols] = index
    if not markers.any():
        return labels

    # For every pixel, the index of the nearest skeleton (marker) pixel.
    nearest = distance_transform_edt(markers == 0, return_indices=True)[1]
    propagated = markers[nearest[0], nearest[1]]
    return np.where(mask, propagated, 0).astype(np.int32)


def _march_into_mask(
    start: np.ndarray, direction: np.ndarray, mask: np.ndarray, max_steps: int
) -> list[tuple[int, int]]:
    """Step 1 px at a time from ``start`` along ``direction`` while inside ``mask``."""
    norm = float(np.hypot(direction[0], direction[1]))
    if norm == 0:
        return []
    step = direction.astype(float) / norm
    height, width = mask.shape
    position = start.astype(float)
    points: list[tuple[int, int]] = []
    for _ in range(max_steps):
        position = position + step
        row, col = int(round(position[0])), int(round(position[1]))
        if not (0 <= row < height and 0 <= col < width) or not mask[row, col]:
            break
        points.append((row, col))
    return points


def extend_branches_to_mask(
    branches: list[Branch],
    mask: np.ndarray,
    *,
    tangent_span: int = 6,
    max_steps: int = 400,
) -> None:
    """Grow each branch centreline along its end tangents out to the wood-mask edge.

    Skeletonisation ends a branch ~one radius short of its rounded mask tip, so the
    drawn centreline is shorter than the instance region. Marching each **free** end
    to the mask boundary makes the skeleton span its region. Only *terminal* tips
    (no child forks there) are extended — extending a trunk/primary tip that sits at
    a fork would drive it into the junction blob and up a neighbour (the bogus
    "trunk continues into the canopy" artifact). The trunk also extends its base to
    the mask bottom. Extension **holds the end radius constant** (the distance
    transform collapses near the mask edge). Mutates ``branches`` in place.
    """
    has_children = {branch.parent_id for branch in branches if branch.parent_id is not None}
    for branch in branches:
        pixels = branch.pixels
        if len(pixels) < 2:
            continue
        span = min(tangent_span, len(pixels) - 1)
        tip = (
            _march_into_mask(pixels[-1], pixels[-1] - pixels[-1 - span], mask, max_steps)
            if branch.id not in has_children  # only genuine free ends
            else []
        )
        base = (
            _march_into_mask(pixels[0], pixels[0] - pixels[span], mask, max_steps)[::-1]
            if branch.level == 0
            else []
        )
        base_arr = np.array(base, dtype=int).reshape(-1, 2)
        tip_arr = np.array(tip, dtype=int).reshape(-1, 2)
        if not len(base_arr) and not len(tip_arr):
            continue
        branch.pixels = np.vstack([base_arr, pixels, tip_arr])
        branch.radius_px = np.concatenate(
            [
                np.full(len(base_arr), branch.radius_px[0], dtype=branch.radius_px.dtype),
                branch.radius_px,
                np.full(len(tip_arr), branch.radius_px[-1], dtype=branch.radius_px.dtype),
            ]
        )


def instance_areas_m2(labels: np.ndarray, meters_per_pixel: float) -> dict[int, float]:
    """Projected area (m²) of each branch instance, keyed by label id."""
    counts = np.bincount(labels.ravel())
    scale = meters_per_pixel ** 2
    return {int(i): float(counts[i]) * scale for i in range(1, len(counts)) if counts[i] > 0}
