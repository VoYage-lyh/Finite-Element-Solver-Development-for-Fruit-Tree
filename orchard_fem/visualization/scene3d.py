from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from orchard_fem.visualization.dependencies import require_plotting_dependencies
from orchard_fem.visualization.model_scene import (
    branch_polyline_points,
    build_branch_lookup,
    resolve_branch_point,
    resolve_excitation_point,
    resolve_observation_point,
)


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
) -> Any:
    """Render the fruit-tree structure as an interactive 3D figure.

    One figure with tapered-tube branches, fruit spheres, and excitation marker.
    Returns the matplotlib Figure.
    """
    plt, np = require_plotting_dependencies(show)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Times New Roman throughout
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    branches = model_data.get("branches", [])
    if not branches:
        raise ValueError("Model JSON contains no branches")

    max_level = max(int(b.get("level", 0)) for b in branches) or 1

    # Clean scientific palette — blue / red / green (ColorBrewer Set1 spirit)
    _LEVEL_PALETTE = [
        "#2977B5",   # 0  azure blue
        "#D94F3D",   # 1  tomato red
        "#3DAE6B",   # 2  mint green
        "#9B59B6",   # 3  purple
        "#F39C12",   # 4+ amber
    ]
    level_colors = {
        level: _LEVEL_PALETTE[min(level, len(_LEVEL_PALETTE) - 1)]
        for level in range(max_level + 1)
    }

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")

    branch_lookup = build_branch_lookup(model_data)
    all_pts: list[list[float]] = []

    for branch in branches:
        level = int(branch.get("level", 0))
        color = level_colors[level]
        polyline = branch_polyline_points(branch)
        all_pts.extend(polyline)
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

        # Label at the midpoint of the branch axis
        mid = resolve_branch_point(branch, 0.5)
        ax.text(
            mid[0], mid[1], mid[2],
            branch["id"],
            fontsize=10, color="#111111",
            ha="center", va="center",
            zorder=30,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, linewidth=1.2, alpha=0.92),
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

    # Excitation point — red star
    try:
        exc_pt, exc_label = resolve_excitation_point(model_data)
        ax.scatter(
            exc_pt[0], exc_pt[1], exc_pt[2],
            s=250, c="crimson", marker="*", zorder=10,
            label=f"Excitation",
        )
    except (KeyError, RuntimeError):
        pass

    # Observation points — blue triangles
    first_obs = True
    for obs in model_data.get("observations", []):
        try:
            obs_pt, _obs_label = resolve_observation_point(model_data, obs)
            ax.scatter(
                obs_pt[0], obs_pt[1], obs_pt[2],
                s=60, c="royalblue", marker="^", zorder=8,
                label="Observation" if first_obs else None,
            )
            first_obs = False
        except (KeyError, RuntimeError):
            pass

    # Equal aspect ratio
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        zs = [p[2] for p in all_pts]
        span = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
            0.1,
        ) * 0.55
        mid_x = (max(xs) + min(xs)) / 2.0
        mid_y = (max(ys) + min(ys)) / 2.0
        mid_z = (max(zs) + min(zs)) / 2.0
        ax.set_xlim(mid_x - span, mid_x + span)
        ax.set_ylim(mid_y - span, mid_y + span)
        ax.set_zlim(mid_z - span, mid_z + span)

    model_name = model_data.get("metadata", {}).get("name", "")
    title = f"Tree 3D Structure — {model_name}" if model_name else "Tree 3D Structure"
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("X (m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Y (m)", fontsize=11, labelpad=8)
    ax.set_zlabel("Z (m)", fontsize=11, labelpad=8)
    ax.tick_params(labelsize=9)

    for level in range(max_level + 1):
        ax.plot([], [], [], color=level_colors[level], linewidth=4, label=f"Level {level}")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#aaaaaa")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
