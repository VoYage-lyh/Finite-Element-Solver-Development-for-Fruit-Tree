"""Trace order-labelled branches (trunk / primary / …) from the skeleton graph.

The skeleton is turned into a rooted tree at the base and each node is given a
**reach** = the longest downstream centreline below it. A branch is traced by
following, at every junction, the child with the greatest reach — so the trunk
tracks the main stem *globally* and cannot veer onto a short side-branch (the
failure mode of greedy angle/radius rules). A branch *terminates* at a co-dominant
fork, where a second child's reach is within ``reach_dominance`` of the best.
Children fork at a junction that lies on the parent path, so they are connected by
construction. The output is a parent-linked list of
:class:`~orchard_vision.types.Branch` with ``level`` 0 = trunk, 1 = primary, …
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from orchard_vision.types import Branch, GraphEdge, SkeletonGraph

_LEVEL_NAMES = {0: "trunk", 1: "primary", 2: "secondary", 3: "tertiary"}


def _level_name(level: int) -> str:
    return _LEVEL_NAMES.get(level, f"order{level}")


def _adjacency(graph: SkeletonGraph) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for index, edge in enumerate(graph.edges):
        adjacency[edge.u].append(index)
        adjacency[edge.v].append(index)
    return adjacency


def _other_node(graph: SkeletonGraph, edge_index: int, node: int) -> int:
    edge = graph.edges[edge_index]
    return edge.v if edge.u == node else edge.u


def _oriented(graph: SkeletonGraph, edge_index: int, from_node: int) -> tuple[np.ndarray, np.ndarray]:
    """Edge pixels and radii oriented to start at ``from_node``."""
    edge = graph.edges[edge_index]
    if edge.u == from_node:
        return edge.pixels, edge.radius_px
    return edge.pixels[::-1], edge.radius_px[::-1]


def _collapse_junctions(graph: SkeletonGraph, merge_px: float) -> SkeletonGraph:
    """Merge junction *clusters* into single branch points.

    Skeletonisation frays one real fork into several nearby junction pixels linked
    by 1–2 px edges, which scatters a single branch point into many and misplaces
    where the trunk/primary split sits. Short edges whose *both* ends are junctions
    are contracted (threshold scales with the local radius, so thick forks merge a
    wider cluster); leaf spurs are left for :func:`_prune_spurs`.
    """
    adjacency = _adjacency(graph)
    degree = {node: len(edges) for node, edges in adjacency.items()}
    parent = list(range(len(graph.nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in graph.edges:
        threshold = max(merge_px, 2.0 * float(edge.radius_px.mean()))
        if degree[edge.u] >= 3 and degree[edge.v] >= 3 and edge.length_px < threshold:
            parent[find(edge.u)] = find(edge.v)

    new_index: dict[int, int] = {}
    new_nodes: list[np.ndarray] = []
    for i in range(len(graph.nodes)):
        rep = find(i)
        if rep not in new_index:
            new_index[rep] = len(new_nodes)
            new_nodes.append(graph.nodes[rep])

    new_edges = [
        GraphEdge(u=new_index[find(e.u)], v=new_index[find(e.v)], pixels=e.pixels, radius_px=e.radius_px)
        for e in graph.edges
        if new_index[find(e.u)] != new_index[find(e.v)]  # drop contracted self-loops
    ]

    degree_new: dict[int, int] = defaultdict(int)
    for edge in new_edges:
        degree_new[edge.u] += 1
        degree_new[edge.v] += 1
    node_kind = ["endpoint" if degree_new[i] == 1 else "junction" for i in range(len(new_nodes))]
    return SkeletonGraph(nodes=np.array(new_nodes), node_kind=node_kind, edges=new_edges)


def _select_root(graph: SkeletonGraph, active_degree: dict[int, int]) -> int:
    """Trunk base = lowest-in-image node (largest row), preferring endpoints."""
    endpoints = [n for n, d in active_degree.items() if d == 1]
    candidates = endpoints or list(active_degree)
    return max(candidates, key=lambda n: graph.nodes[n][0])


def _prune_spurs(
    graph: SkeletonGraph,
    adjacency: dict[int, list[int]],
    root: int,
    min_length_px: float,
) -> set[int]:
    """Drop short leaf hairs (skeletonisation noise) hanging off junctions."""
    active = set(range(len(graph.edges)))

    def degree(node: int) -> int:
        return sum(1 for e in adjacency[node] if e in active)

    changed = True
    while changed:
        changed = False
        for node in list(adjacency):
            if node == root or degree(node) != 1:
                continue
            (edge_index,) = [e for e in adjacency[node] if e in active]
            edge = graph.edges[edge_index]
            other = edge.v if edge.u == node else edge.u
            if edge.length_px < min_length_px and degree(other) >= 3:
                active.discard(edge_index)
                changed = True
    return active


def _drop_tiny_terminals(branches: list[Branch], min_length_px: float) -> list[Branch]:
    """Remove terminal branches shorter than ``min_length_px`` (base-flare stubs,
    tiny twigs). A branch is terminal when no other branch calls it a parent."""
    parent_ids = {branch.parent_id for branch in branches}
    return [
        branch
        for branch in branches
        if branch.parent_id is None or branch.id in parent_ids or branch.length_px >= min_length_px
    ]


def drop_low_primaries(
    branches: list[Branch], base_row: int, min_height_px: float
) -> list[Branch]:
    """Drop **short** level-1 stubs attaching below ``min_height_px`` above the base
    — and their descendants. A base-flare artifact is both low *and* short; a real
    primary is long even where it forks low, so requiring ``length < min_height_px``
    keeps real limbs (and avoids nuking a whole tree when the scale is off)."""
    removed = {
        branch.id
        for branch in branches
        if branch.level == 1
        and (base_row - int(branch.pixels[0, 0])) < min_height_px
        and branch.length_px < min_height_px
    }
    changed = True
    while changed:  # cascade to descendants of a removed branch
        changed = False
        for branch in branches:
            if branch.id not in removed and branch.parent_id in removed:
                removed.add(branch.id)
                changed = True
    return [branch for branch in branches if branch.id not in removed]


def order_branches(
    graph: SkeletonGraph,
    *,
    min_spur_px: float = 12.0,
    junction_merge_px: float = 8.0,
    reach_dominance: float = 0.45,
    max_level: int | None = None,
) -> tuple[list[Branch], tuple[int, int]]:
    """Return ``(branches, root_pixel_rc)`` with branches labelled by order.

    The trunk is traced by **global reach**, not local greed: from the base, at
    every junction it follows the child leading to the longest downstream path (the
    main stem), so it can never veer off into a short side-branch. It *terminates*
    only at a co-dominant fork — where a second child's reach is at least
    ``reach_dominance`` × the best's — ending where the stem genuinely splits into
    scaffolds. Primaries/secondaries are traced the same way off their parent. A
    child always starts at a junction that lies on its parent's path, so branches
    are connected by construction. ``max_level`` caps the traced order.
    """
    if not graph.edges:
        return [], (0, 0)

    graph = _collapse_junctions(graph, junction_merge_px)
    adjacency = _adjacency(graph)
    full_degree = {node: len(edges) for node, edges in adjacency.items()}
    root = _select_root(graph, full_degree)
    active = _prune_spurs(graph, adjacency, root, min_spur_px)

    # Rooted spanning tree from the base (any cycle back-edge is dropped).
    children: dict[int, list[tuple[int, int]]] = defaultdict(list)
    visited = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for edge_index in adjacency[node]:
            if edge_index not in active:
                continue
            other = _other_node(graph, edge_index, node)
            if other not in visited:
                visited.add(other)
                children[node].append((edge_index, other))
                stack.append(other)

    # reach[node] = longest downstream centreline length below it (post-order).
    reach: dict[int, float] = {}
    order_stack: list[tuple[int, bool]] = [(root, False)]
    while order_stack:
        node, done = order_stack.pop()
        if done:
            reach[node] = max(
                (graph.edges[e].length_px + reach[c] for e, c in children[node]), default=0.0
            )
        else:
            order_stack.append((node, True))
            order_stack.extend((child, False) for _, child in children[node])

    def score(child: tuple[int, int]) -> float:
        edge_index, node = child
        return graph.edges[edge_index].length_px + reach[node]

    def trace(entry_node: int, first_child: tuple[int, int]):
        """Follow max-reach continuations until a co-dominant fork or a tip.
        Returns (edges as (edge_index, from_node), spawned (junction, child))."""
        taken = [(first_child[0], entry_node)]
        node = first_child[1]
        spawned: list[tuple[int, tuple[int, int]]] = []
        while True:
            forward = children[node]
            if not forward:
                break
            ranked = sorted(forward, key=score, reverse=True)
            best = ranked[0]
            if len(ranked) > 1 and score(ranked[1]) >= reach_dominance * score(best):
                spawned.extend((node, child) for child in ranked)  # co-dominant fork
                break
            spawned.extend((node, child) for child in ranked[1:])
            taken.append((best[0], node))
            node = best[1]
        return taken, spawned

    root_children = children[root]
    if not root_children:
        return [], (int(graph.nodes[root][0]), int(graph.nodes[root][1]))

    branches: list[Branch] = []
    counters: dict[int, int] = defaultdict(int)
    trunk_first = max(root_children, key=score)
    other_base_stems = [child for child in root_children if child != trunk_first]
    queue: deque[tuple[int, tuple[int, int], str | None, int]] = deque()
    queue.append((root, trunk_first, None, 0))

    while queue:
        entry_node, first_child, parent_id, level = queue.popleft()
        taken, spawned = trace(entry_node, first_child)
        if parent_id is None:  # the trunk also parents any other stem off the base
            spawned = [(root, child) for child in other_base_stems] + spawned

        pixel_chunks: list[np.ndarray] = []
        radius_chunks: list[np.ndarray] = []
        for index, (edge_index, from_node) in enumerate(taken):
            pixels, radii = _oriented(graph, edge_index, from_node)
            if index > 0:  # drop the shared junction pixel between edges
                pixels, radii = pixels[1:], radii[1:]
            pixel_chunks.append(pixels)
            radius_chunks.append(radii)

        counters[level] += 1
        branch_id = "trunk" if level == 0 else f"{_level_name(level)}_{counters[level]}"
        branches.append(
            Branch(
                id=branch_id,
                parent_id=parent_id,
                level=level,
                pixels=np.vstack(pixel_chunks),
                radius_px=np.concatenate(radius_chunks),
            )
        )
        if max_level is not None and level + 1 > max_level:
            continue
        for junction_node, child in spawned:
            queue.append((junction_node, child, branch_id, level + 1))

    branches = _drop_tiny_terminals(branches, max(min_spur_px, junction_merge_px))
    root_rc = (int(graph.nodes[root][0]), int(graph.nodes[root][1]))
    return branches, root_rc
