"""Pixel skeleton → undirected graph of branch chains.

:func:`skeletonize_mask` thins the branch mask to a 1-pixel medial axis and
records the local radius (Euclidean distance transform) at every skeleton pixel.
:func:`build_graph` walks that skeleton into nodes (endpoints + junctions) and
edges (the degree-2 chains between them); downstream ordering merges the edges
into branches.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.morphology import skeletonize

from orchard_vision.types import GraphEdge, SkeletonGraph

# 8-connectivity neighbour offsets, in (drow, dcol).
_NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def skeletonize_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(skeleton, radius_px)``.

    ``skeleton`` is a 1-pixel-wide boolean medial axis; ``radius_px`` is the
    distance of each pixel to the background (the local branch half-width), which
    later feeds the solver's cross-section radii.
    """
    skeleton = skeletonize(mask)
    radius = distance_transform_edt(mask)
    return skeleton, radius.astype(np.float32)


def _pixel_degree(skeleton: np.ndarray) -> np.ndarray:
    """Number of skeleton neighbours for each pixel (0 off the skeleton)."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    counts = convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    return counts * skeleton


def _walk_chain(
    start: tuple[int, int],
    second: tuple[int, int],
    skeleton_pixels: set[tuple[int, int]],
    node_pixels: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Trace a degree-2 chain from node ``start`` through ``second`` to the next node.

    Returns the ordered pixel list including both bounding node pixels. A local
    visited set guards against the short cycles that 8-connectivity can create.
    """
    chain = [start, second]
    if second in node_pixels:  # adjacent node → unit-length edge
        return chain
    visited = {start, second}
    previous, current = start, second
    while True:
        forward = [
            (current[0] + dr, current[1] + dc)
            for dr, dc in _NEIGHBOURS
            if (current[0] + dr, current[1] + dc) in skeleton_pixels
            and (current[0] + dr, current[1] + dc) != previous
        ]
        forward = [p for p in forward if p not in visited or p in node_pixels]
        if not forward:
            break
        nxt = forward[0]
        chain.append(nxt)
        if nxt in node_pixels:
            break
        visited.add(nxt)
        previous, current = current, nxt
    return chain


def build_graph(skeleton: np.ndarray, radius: np.ndarray) -> SkeletonGraph:
    """Trace the skeleton into a node/edge graph (8-connectivity).

    Nodes are skeleton pixels with degree != 2 (endpoints with degree 1 and
    junctions with degree >= 3). Edges are the simple chains between nodes; each
    keeps its ordered pixel polyline and the radius sampled along it.
    """
    degree = _pixel_degree(skeleton)
    node_rc = np.argwhere(skeleton & (degree != 2))
    node_index = {(int(r), int(c)): i for i, (r, c) in enumerate(node_rc)}
    node_pixels = set(node_index)
    node_kind = ["endpoint" if degree[r, c] == 1 else "junction" for r, c in node_rc]

    skeleton_pixels = {(int(r), int(c)) for r, c in np.argwhere(skeleton)}
    edges: list[GraphEdge] = []
    walked: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for node, node_id in node_index.items():
        for dr, dc in _NEIGHBOURS:
            neighbour = (node[0] + dr, node[1] + dc)
            if neighbour not in skeleton_pixels or (node, neighbour) in walked:
                continue
            chain = _walk_chain(node, neighbour, skeleton_pixels, node_pixels)
            end = chain[-1]
            if end not in node_index:  # dangling chain (no closing node) → skip
                continue
            walked.add((node, chain[1]))
            walked.add((end, chain[-2]))
            pixels = np.array(chain, dtype=int)
            edges.append(
                GraphEdge(
                    u=node_id,
                    v=node_index[end],
                    pixels=pixels,
                    radius_px=radius[pixels[:, 0], pixels[:, 1]],
                )
            )

    return SkeletonGraph(nodes=node_rc, node_kind=node_kind, edges=edges)
