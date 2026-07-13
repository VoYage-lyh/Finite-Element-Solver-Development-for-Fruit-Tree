from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from orchard_fem.visualization.dependencies import require_plotting_dependencies
from orchard_fem.visualization.model_scene import (
    LEVEL_PALETTE,
    branch_polyline_points,
    build_branch_lookup,
    hierarchical_labels,
    resolve_branch_point,
    resolve_excitation_point,
    resolve_observation_point,
)


def _apply_publication_style(plt) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


def _compact_label(label: str) -> str:
    replacements = {
        "primary": "P",
        "secondary": "S",
        "scaffold": "Scaf",
        "leader": "Lead",
        "trunk": "Trunk",
        "inner": "in",
        "outer": "out",
        "center": "C",
        "central": "C",
        "left": "L",
        "right": "R",
        "lower": "low",
        "upper": "up",
        "upright": "up",
        "terminal": "term",
        "fruiting": "fruit",
        "axis": "axis",
        "arch": "arch",
        "spray": "spray",
        "top": "top",
        "mid": "mid",
        "root": "root",
        "tip": "tip",
    }
    return " ".join(replacements.get(part, part) for part in label.split("_"))


def _extract_outer_radius(station: dict) -> float:
    if station.get("shorthand") == "circular":
        return float(station["outer_radius"])
    regions = station.get("regions", [])
    if regions:
        return float(regions[0].get("outer_radius", 0.01))
    return 0.01


def _radius_at_s(branch: dict, s: float) -> float:
    stations = branch.get("stations", [])
    if not stations:
        return 0.01
    sorted_stations = sorted(stations, key=lambda st: float(st.get("s", 0.0)))
    if s <= float(sorted_stations[0].get("s", 0.0)):
        return _extract_outer_radius(sorted_stations[0])
    if s >= float(sorted_stations[-1].get("s", 1.0)):
        return _extract_outer_radius(sorted_stations[-1])
    for i in range(len(sorted_stations) - 1):
        s0 = float(sorted_stations[i].get("s", 0.0))
        s1 = float(sorted_stations[i + 1].get("s", 1.0))
        if s0 <= s <= s1:
            if s1 - s0 < 1e-12:
                return _extract_outer_radius(sorted_stations[i])
            alpha = (s - s0) / (s1 - s0)
            r0 = _extract_outer_radius(sorted_stations[i])
            r1 = _extract_outer_radius(sorted_stations[i + 1])
            return r0 + alpha * (r1 - r0)
    return _extract_outer_radius(sorted_stations[-1])


def _branch_arc_s(polyline: list[list[float]]) -> list[float]:
    cumulative = [0.0]
    for i in range(len(polyline) - 1):
        p0, p1 = polyline[i], polyline[i + 1]
        seg = math.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3)))
        cumulative.append(cumulative[-1] + seg)
    total = cumulative[-1]
    if total < 1e-12:
        n = len(polyline)
        return [float(i) / max(1, n - 1) for i in range(n)]
    return [v / total for v in cumulative]


def _tube_surface(p0, p1, r0: float, r1: float, np_mod, n_theta: int = 14):
    p0 = np_mod.asarray(p0, dtype=float)
    p1 = np_mod.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np_mod.linalg.norm(direction))
    if length < 1e-12:
        return None
    e_z = direction / length
    e_tmp = np_mod.array([1.0, 0.0, 0.0]) if abs(e_z[0]) < 0.9 else np_mod.array([0.0, 1.0, 0.0])
    e_x = np_mod.cross(e_z, e_tmp)
    e_x /= np_mod.linalg.norm(e_x)
    e_y = np_mod.cross(e_z, e_x)

    theta = np_mod.linspace(0.0, 2.0 * math.pi, n_theta + 1)
    cos_t = np_mod.cos(theta)
    sin_t = np_mod.sin(theta)

    X = np_mod.zeros((2, n_theta + 1))
    Y = np_mod.zeros((2, n_theta + 1))
    Z = np_mod.zeros((2, n_theta + 1))
    for ring, (center, radius) in enumerate([(p0, r0), (p1, r1)]):
        pts = center[:, None] + radius * (
            np_mod.outer(e_x, cos_t) + np_mod.outer(e_y, sin_t)
        )
        X[ring] = pts[0]
        Y[ring] = pts[1]
        Z[ring] = pts[2]
    return X, Y, Z


def plot_tree_3d(
    model_data: dict,
    *,
    show: bool = True,
    output_path: str | Path | None = None,
    show_io_markers: bool = True,
) -> Any:
    """Render the fruit-tree structure as an interactive 3D figure.

    One figure with tapered-tube branches, fruit spheres, and excitation marker.
    Returns the matplotlib Figure.
    """
    plt, np = require_plotting_dependencies(show)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _apply_publication_style(plt)

    branches = model_data.get("branches", [])
    if not branches:
        raise ValueError("Model JSON contains no branches")

    max_level = max(int(b.get("level", 0)) for b in branches) or 1
    branch_polylines = {
        branch["id"]: branch_polyline_points(branch)
        for branch in branches
    }
    all_pts = [
        point
        for polyline in branch_polylines.values()
        for point in polyline
    ]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    z_range = max(z_max - z_min, 0.1)
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    # Tight padding around actual tree extent — no forced cube.
    x_pad = max(x_range * 0.08, 0.08)
    y_pad = max(y_range * 0.30, 0.10)
    z_pad = max(z_range * 0.05, 0.08)
    label_offset = max(max(x_range, z_range) * 0.04, 0.05)

    level_colors = {
        level: LEVEL_PALETTE[min(level, len(LEVEL_PALETTE) - 1)]
        for level in range(max_level + 1)
    }

    fig = plt.figure(figsize=(14, 13))
    ax = fig.add_subplot(111, projection="3d")

    branch_lookup = build_branch_lookup(model_data)
    labels = hierarchical_labels(branches)

    for branch in branches:
        level = int(branch.get("level", 0))
        color = level_colors[level]
        polyline = branch_polylines[branch["id"]]
        s_vals = _branch_arc_s(polyline)

        for seg in range(len(polyline) - 1):
            p0, p1 = polyline[seg], polyline[seg + 1]
            r0 = _radius_at_s(branch, s_vals[seg])
            r1 = _radius_at_s(branch, s_vals[seg + 1])
            result = _tube_surface(p0, p1, r0, r1, np)
            if result is None:
                continue
            X, Y, Z = result
            ax.plot_surface(X, Y, Z, color=color, alpha=0.92, linewidth=0, antialiased=True)

        label_anchor = np.asarray(resolve_branch_point(branch, 0.58), dtype=float)
        radial = np.array([label_anchor[0] - mid_x, label_anchor[1] - mid_y, 0.0])
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm > 1.0e-12:
            radial /= radial_norm
        label_pos = label_anchor + radial * label_offset + np.array(
            [0.0, 0.0, label_offset * 1.15]
        )
        ax.plot(
            [label_anchor[0], label_pos[0]],
            [label_anchor[1], label_pos[1]],
            [label_anchor[2], label_pos[2]],
            color="#666666",
            linewidth=0.7,
            alpha=0.55,
            zorder=20,
        )
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            labels.get(branch["id"], _compact_label(branch["id"])),
            fontsize=14,
            fontweight="bold",
            color="#111111",
            ha="center", va="center",
            zorder=30,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=color, linewidth=1.6, alpha=0.95),
        )

    # Fruits — orange spheres scaled by radius/mass
    first_fruit = True
    for fruit in model_data.get("fruits", []):
        fruit_branch = branch_lookup.get(fruit.get("branch_id", ""))
        if fruit_branch is None:
            continue
        pos = resolve_branch_point(fruit_branch, float(fruit.get("location_s", 0.5)))
        radius_m = float(fruit.get("radius_m", fruit.get("equivalent_radius_m", 0.03)))
        marker_size = max(20.0, (radius_m * 1500.0) ** 1.6)
        ax.scatter(
            pos[0], pos[1], pos[2],
            s=marker_size,
            c="darkorange",
            marker="o",
            depthshade=True,
            label="Fruit" if first_fruit else None,
            zorder=5,
        )
        first_fruit = False

    # Excitation point — red star, offset radially so it isn't hidden inside the tube
    try:
        if not show_io_markers:
            raise RuntimeError("io markers suppressed")
        exc_pt, exc_label = resolve_excitation_point(model_data)
        exc_arr = np.asarray(exc_pt, dtype=float)
        radial = np.array([exc_arr[0] - mid_x, exc_arr[1] - mid_y, 0.0])
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm > 1.0e-6:
            radial /= radial_norm
        else:
            radial = np.array([-1.0, 0.0, 0.0])  # default outward direction for axial trunk
        offset_dist = max(max(x_range, z_range) * 0.06, 0.10)
        marker_pos = exc_arr + radial * offset_dist
        # leader line from branch surface to marker
        ax.plot(
            [exc_arr[0], marker_pos[0]],
            [exc_arr[1], marker_pos[1]],
            [exc_arr[2], marker_pos[2]],
            color="crimson", linewidth=1.3, alpha=0.85, zorder=15,
        )
        ax.scatter(
            marker_pos[0], marker_pos[1], marker_pos[2],
            s=420, c="crimson", marker="*",
            edgecolors="black", linewidths=0.9,
            zorder=50, label="Excitation",
        )
    except (KeyError, RuntimeError):
        pass

    # Observation points — square/triangle/circle for root/mid/tip
    node_styles = {
        "root": {"marker": "s", "color": "#1f3a93", "label": "Obs (root)"},
        "mid":  {"marker": "^", "color": "#1f77b4", "label": "Obs (mid)"},
        "tip":  {"marker": "o", "color": "#5eaeff", "label": "Obs (tip)"},
    }
    legend_seen = set()
    for obs in (model_data.get("observations", []) if show_io_markers else []):
        try:
            obs_pt, _obs_label = resolve_observation_point(model_data, obs)
            node = obs.get("target_node", "tip")
            style = node_styles.get(node, node_styles["tip"])
            label = style["label"] if node not in legend_seen else None
            legend_seen.add(node)
            ax.scatter(
                obs_pt[0], obs_pt[1], obs_pt[2],
                s=70,
                c=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.6,
                zorder=8,
                label=label,
            )
        except (KeyError, RuntimeError):
            pass

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    # Proportional 3D box — keeps the tree filling the view instead of a forced cube.
    # Y axis is widened to at least ~35% of the larger horizontal extent so it does not
    # collapse into a thin slab when the tree is nearly planar.
    box_x = x_range + 2 * x_pad
    box_y = max(y_range + 2 * y_pad, max(x_range, z_range) * 0.35)
    box_z = z_range + 2 * z_pad
    ax.set_box_aspect((box_x, box_y, box_z))

    # Loosen Y-axis tick density — tree depth is typically small, so default tick
    # spacing crowds 5+ labels into a thin slab. Cap at 3 evenly-spaced ticks.
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=7))

    ax.set_title("3D View", fontsize=20, pad=14)
    ax.set_xlabel("X (m)", fontsize=16, labelpad=14)
    ax.set_ylabel("Y (m)", fontsize=16, labelpad=18)
    ax.set_zlabel("Z (m)", fontsize=16, labelpad=14)
    ax.tick_params(axis="x", labelsize=13, pad=2)
    ax.tick_params(axis="y", labelsize=13, pad=6)
    ax.tick_params(axis="z", labelsize=13, pad=2)
    ax.view_init(elev=20, azim=-58)

    for level in range(max_level + 1):
        ax.plot([], [], [], color=level_colors[level], linewidth=5, label=f"Level {level}")
    ax.legend(loc="upper left", fontsize=14, framealpha=0.9, edgecolor="#aaaaaa")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches="tight" + pad_inches=0 removes outer whitespace around the
        # figure (the gap between content and image boundary).
        fig.savefig(str(out), dpi=200, bbox_inches="tight", pad_inches=0.0)

    if show:
        plt.show()

    return fig
