"""End-to-end self-consistency check with **multi-clamp** Pareto search.

For each tree we:
    1. Run a wide FRF sweep to find the dominant in-band local-max resonance.
    2. Tighten the detachment displacement to 2 mm.
    3. Enumerate candidate clamps:
           * Trunk heights at s = 0.25, 0.40, 0.55, 0.70, 0.85
           * Root (s = 0) of every branch joined directly to the trunk
       For each clamp, run a Pareto sweep over (f, A) around the resonance
       and extract a knee.
    4. Select the **best clamp** as the one whose knee is closest to the
       ideal (coverage = 1, σ = 0) point in normalised objective space.

Per-tree outputs (in ``outputs/``):
    * ``verify_pareto_tree_<n>.{png,pdf}`` — multi-clamp Pareto overlay,
      best knee marked in red.
    * ``verify_frf_tree_<n>.{png,pdf}``    — FRF sweep with resonance marker.

Cross-tree outputs (only best-clamp knee per tree):
    * ``verify_pareto_all_trees.{png,pdf}`` — best-knee per tree on log axes.
    * ``verify_frf_all_trees.{png,pdf}``    — all 5 FRFs stacked.
    * ``verify_knees_summary.{png,pdf}``    — best (clamp, f, A, cov, σ) per tree.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ────────────────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────────────────
_FEASIBLE_BAND_HZ = (3.0, 20.0)
_TRUNK_CLAMP_S = (0.25, 0.40, 0.55, 0.70, 0.85)
_AMPLITUDE_GRID_N = (50.0, 100.0, 200.0, 400.0, 800.0)


# ────────────────────────────────────────────────────────────────────────────
#  Publication-grade matplotlib style
# ────────────────────────────────────────────────────────────────────────────
def _apply_pub_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fontsize": 10,
        "legend.edgecolor": "#cccccc",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_both(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))


# ────────────────────────────────────────────────────────────────────────────
#  Resonance detection
# ────────────────────────────────────────────────────────────────────────────
def _find_in_band_resonance(
    freqs: np.ndarray,
    mags: np.ndarray,
    band: tuple[float, float],
    *,
    prominence_ratio: float = 0.10,
) -> tuple[int, bool]:
    """Return ``(peak_index, has_genuine_local_max)`` inside *band*."""
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        raise RuntimeError(
            f"No FRF samples in {band[0]}–{band[1]} Hz; widen the sweep."
        )

    log_mags = np.log(np.maximum(mags, 1.0e-20))
    is_local_max = np.zeros(mags.size, dtype=bool)
    is_local_max[1:-1] = (
        (log_mags[1:-1] > log_mags[:-2]) & (log_mags[1:-1] > log_mags[2:])
    )
    candidates = is_local_max & in_band
    band_median = float(np.median(mags[in_band]))
    if candidates.any():
        cand_idx = np.flatnonzero(candidates)
        prominent = mags[cand_idx] >= band_median * (1.0 + prominence_ratio)
        if prominent.any():
            cand_idx = cand_idx[prominent]
        peak_idx = int(cand_idx[int(np.argmax(mags[cand_idx]))])
        return peak_idx, True

    # Fallback: curvature inflection.
    curvature = np.zeros_like(mags)
    curvature[1:-1] = log_mags[2:] - 2.0 * log_mags[1:-1] + log_mags[:-2]
    band_idx = np.flatnonzero(in_band)
    interior = band_idx[(band_idx > 0) & (band_idx < mags.size - 1)]
    if interior.size == 0:
        peak_idx = int(band_idx[int(np.argmax(mags[in_band]))])
        return peak_idx, False
    peak_idx = int(interior[int(np.argmin(curvature[interior]))])
    return peak_idx, False


# ────────────────────────────────────────────────────────────────────────────
#  Workflow data containers
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class ClampResult:
    clamp_label: str        # internal label, e.g. "left_scaffold@0.00"
    display_label: str      # reader-friendly, e.g. "B1 root"
    front: object           # ParetoFront
    knee: object            # ParetoKnee


@dataclass
class TreeResult:
    label: str
    n_fruits: int
    freqs: np.ndarray
    mags: np.ndarray
    f_resonance: float
    clamps: list[ClampResult] = field(default_factory=list)
    best_idx: int = 0

    @property
    def best(self) -> ClampResult:
        return self.clamps[self.best_idx]


# ────────────────────────────────────────────────────────────────────────────
#  Per-tree workflow
# ────────────────────────────────────────────────────────────────────────────
def _coarse_frf_sweep(model, f_min: float, f_max: float, steps: int):
    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    swept = replace(
        model,
        analysis=replace(
            model.analysis,
            frequency_start_hz=float(f_min),
            frequency_end_hz=float(f_max),
            frequency_steps=int(steps),
        ),
    )
    exp = solve_embedded_beam_frequency_response_experiment(swept, polynomial_degree=1)
    res = exp.result
    freqs = np.array([p.frequency_hz for p in res.points])
    name_to_idx = {n: i for i, n in enumerate(res.observation_names)}
    tip_obs = [n for n in res.observation_names if n.endswith("_tip_ux")]
    mags = np.zeros_like(freqs)
    for j, p in enumerate(res.points):
        mags[j] = float(np.mean([p.observation_magnitudes[name_to_idx[n]]
                                  for n in tip_obs]))
    return freqs, mags


def _candidate_clamps(model) -> list[str]:
    """Return clamp labels for trunk heights + each trunk-child branch root."""
    labels = [f"trunk@{s:.2f}" for s in _TRUNK_CLAMP_S]
    trunk_node = model.topology.require_node("trunk")
    for bid in trunk_node.child_branch_ids:
        labels.append(f"{bid}@0.00")
    return labels


def _build_hierarchical_label_map(model) -> dict[str, str]:
    """Return ``{branch_id: hierarchical_label}`` matching ``view-tree``.

    Trunk → ``T``; primary branches → ``1, 2, …`` ordered left→right by the
    branch endpoint's x-coordinate; secondaries → ``1.1, 1.2, …``; and so on.
    Mirrors :func:`orchard_fem.visualization.model_scene.hierarchical_labels`
    so figure callouts stay consistent with the 3D scene plot.
    """
    children_by_parent: dict[str | None, list[str]] = {}
    for n_id in model.topology.nodes:
        node = model.topology.require_node(n_id)
        parent = node.parent_branch_id
        children_by_parent.setdefault(parent, []).append(n_id)

    def _end_x(branch_id: str) -> float:
        branch = next(b for b in model.branches if b.branch_id == branch_id)
        return float(branch.path.end.x)

    out: dict[str, str] = {}

    def _assign(parent_id: str | None, parent_label: str) -> None:
        kids = sorted(children_by_parent.get(parent_id, []), key=_end_x)
        for i, kid in enumerate(kids, start=1):
            kid_label = f"{parent_label}.{i}" if parent_label else str(i)
            out[kid] = kid_label
            _assign(kid, kid_label)

    for root_id in children_by_parent.get(None, []):
        out[root_id] = "T"
        _assign(root_id, "")

    return out


def _pretty_clamp_label(raw: str, label_map: dict[str, str]) -> str:
    """Translate ``branch_id@s`` to ``HierarchicalLabel@position`` notation.

    ``s = 0`` becomes ``root``, ``s = 1`` becomes ``tip``, ``s = 0.5`` becomes
    ``mid``; intermediate fractions are rendered as ``XX%``. For example,
    ``left_scaffold@0.00`` (with hierarchical label ``1``) → ``B1 root``;
    ``trunk@0.25`` → ``T @ 25%``.
    """
    if "@" in raw:
        bid, s_str = raw.split("@", 1)
        s = float(s_str)
    else:
        bid, s = raw, 0.0

    hier = label_map.get(bid, bid)
    if hier == "T":
        prefix = "T"
    else:
        prefix = f"B{hier}"

    if abs(s) < 1.0e-6:
        suffix = "root"
    elif abs(s - 1.0) < 1.0e-6:
        suffix = "tip"
    elif abs(s - 0.5) < 1.0e-6:
        suffix = "mid"
    else:
        suffix = f"{s * 100:.0f}%"

    return f"{prefix} @ {suffix}"


def _knee_distance_to_ideal(
    knees: list, *, sigma_norm: float
) -> np.ndarray:
    """Normalised distance from each knee to the ideal (cov=1, σ=0) point.

    The coverage axis is already in [0, 1]. The stress axis is normalised by
    ``sigma_norm`` — typically the max stress observed across all knees of
    this tree — so the two axes carry comparable weight.
    """
    distances = np.zeros(len(knees))
    for i, k in enumerate(knees):
        cov = float(k.detachment_coverage)
        sigma_n = float(k.trunk_max_stress) / max(sigma_norm, 1.0e-12)
        distances[i] = float(np.sqrt((1.0 - cov) ** 2 + sigma_n ** 2))
    return distances


def _evaluate_tree(model_path: Path, label: str) -> TreeResult:
    from orchard_fem.calibration.fenicsx_bridge import (
        build_fenicsx_pareto_evaluator,
    )
    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.recommendation.pareto import pareto_front_from_grid

    print(f"\n[{label}] loading {model_path.name} …")
    model = load_orchard_model(str(model_path))
    print(f"[{label}]   fruits={len(model.fruits)}, branches={len(model.branches)}")

    model = replace(
        model,
        fruit_policy=replace(model.fruit_policy, detachment_displacement_m=0.002),
    )

    print(f"[{label}] FRF sweep 0.5–30 Hz …")
    t0 = time.time()
    freqs, mags = _coarse_frf_sweep(model, 0.5, 30.0, 60)
    peak_idx, has_local_max = _find_in_band_resonance(
        freqs, mags, _FEASIBLE_BAND_HZ,
    )
    f_resonance = float(freqs[peak_idx])
    detect_type = "local max" if has_local_max else "curvature inflection"
    print(f"[{label}]   sweep {time.time() - t0:.1f} s   "
          f"resonance ({detect_type}) @ {f_resonance:.2f} Hz "
          f"({mags[peak_idx] / model.excitation.amplitude * 1e3:.3f} mm/N)")

    theta = {
        "E": float(model.materials[0].youngs_modulus),
        "rho": float(model.materials[0].density),
    }
    evaluator = build_fenicsx_pareto_evaluator(model, amplitude_unit="m")

    f_grid = [
        max(0.5, f_resonance - 2.0),
        max(0.5, f_resonance - 1.0),
        f_resonance,
        f_resonance + 1.0,
        f_resonance + 2.0,
    ]
    A_grid = list(_AMPLITUDE_GRID_N)

    clamp_labels = _candidate_clamps(model)
    label_map = _build_hierarchical_label_map(model)
    print(f"[{label}] {len(clamp_labels)} candidate clamps × |f|={len(f_grid)} × "
          f"|A|={len(A_grid)} = {len(clamp_labels) * len(f_grid) * len(A_grid)} "
          f"FE solves")

    clamps: list[ClampResult] = []
    t0 = time.time()
    for clamp_label in clamp_labels:
        try:
            front = pareto_front_from_grid(
                clamp_label, f_grid, A_grid, evaluator, theta,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}]   skip {clamp_label}: {exc}")
            continue
        if front.non_dominated_index.size == 0:
            continue
        clamps.append(ClampResult(
            clamp_label=clamp_label,
            display_label=_pretty_clamp_label(clamp_label, label_map),
            front=front,
            knee=front.knee,
        ))
    print(f"[{label}]   {len(clamps)} clamps produced a knee "
          f"({time.time() - t0:.1f} s)")

    if not clamps:
        raise RuntimeError(f"No feasible clamp for {label}.")

    # Best-clamp selection: knee closest to the ideal (cov = 1, σ = 0) point
    # after normalising σ by the max σ over knees of this tree.
    knees = [c.knee for c in clamps]
    sigma_norm = max(float(k.trunk_max_stress) for k in knees)
    distances = _knee_distance_to_ideal(knees, sigma_norm=sigma_norm)
    best_idx = int(np.argmin(distances))
    best = clamps[best_idx]
    print(f"[{label}] best clamp: {best.display_label} ({best.clamp_label})  "
          f"f*={best.knee.frequency_hz:.2f} Hz, "
          f"A*={best.knee.amplitude:.0f} N, "
          f"coverage={best.knee.detachment_coverage:.2f}, "
          f"σ={best.knee.trunk_max_stress / 1e6:.3f} MPa")

    return TreeResult(
        label=label,
        n_fruits=len(model.fruits),
        freqs=freqs,
        mags=mags,
        f_resonance=f_resonance,
        clamps=clamps,
        best_idx=best_idx,
    )


def main() -> int:
    _apply_pub_style()
    out_dir = REPO / "outputs"
    out_dir.mkdir(exist_ok=True)

    results: list[TreeResult] = []
    for n in (1, 2, 3, 4, 5):
        model_path = REPO / "trees" / f"tree_{n}.json"
        if not model_path.exists():
            print(f"[skip] {model_path} not found")
            continue
        result = _evaluate_tree(model_path, label=f"tree_{n}")
        results.append(result)

        _save_pareto_multi_clamp(result, out_dir / f"verify_pareto_tree_{n}")
        _save_frf(result, out_dir / f"verify_frf_tree_{n}")
        print(f"[tree_{n}] figures → outputs/verify_pareto_tree_{n}.{{png,pdf}} "
              f"+ verify_frf_tree_{n}.{{png,pdf}}")

    if len(results) >= 2:
        _save_all_pareto_overlay(results, out_dir / "verify_pareto_all_trees")
        _save_all_frf_overlay(results, out_dir / "verify_frf_all_trees")
        _save_knees_summary(results, out_dir / "verify_knees_summary")
        print(f"\n[summary] cross-tree figures → outputs/verify_*_all_trees.{{png,pdf}} "
              f"+ verify_knees_summary.{{png,pdf}}")

    _print_recommendation_table(results)
    print(f"\n[done] processed {len(results)} tree(s).")
    return 0


# ────────────────────────────────────────────────────────────────────────────
#  Figures (per-tree, multi-clamp)
# ────────────────────────────────────────────────────────────────────────────
_CLAMP_PALETTE = [
    "#2166AC", "#1B7837", "#762A83", "#E08214",
    "#5AAE61", "#9970AB", "#FDB863", "#80CDC1",
    "#D6604D", "#4393C3",
]


def _save_pareto_multi_clamp(result: TreeResult, stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    best = result.best
    best_sigma_max = max(c.knee.trunk_max_stress / 1e6 for c in result.clamps)
    best_cov_max = max(c.knee.detachment_coverage for c in result.clamps)

    sigma_lo = float("inf")
    sigma_hi = 0.0
    for i, c in enumerate(result.clamps):
        front = c.front
        cov = -front.objectives[:, 0]
        sigma = front.objectives[:, 1] / 1.0e6
        nd = front.non_dominated_index
        if nd.size == 0:
            continue
        order = np.argsort(cov[nd])
        color = _CLAMP_PALETTE[i % len(_CLAMP_PALETTE)]
        is_best = c is best

        ax.plot(cov[nd][order], sigma[nd][order],
                color=color, linewidth=1.6 if is_best else 1.1,
                alpha=0.85 if is_best else 0.55, zorder=3 if is_best else 2)
        ax.scatter(cov[nd], sigma[nd],
                   color=color,
                   s=70 if is_best else 42,
                   edgecolors="white", linewidths=0.9,
                   label=f"{c.display_label} (best)" if is_best
                         else c.display_label,
                   zorder=4 if is_best else 3)

        # Use floor at the smallest positive sigma on each front
        pos = sigma[nd][sigma[nd] > 0]
        if pos.size:
            sigma_lo = min(sigma_lo, float(pos.min()))
        sigma_hi = max(sigma_hi, float(sigma[nd].max()))

    # Mark the best knee
    bk = best.knee
    bx = bk.detachment_coverage
    by = bk.trunk_max_stress / 1.0e6
    ax.scatter([bx], [by], s=380, marker="o",
               facecolor="none", edgecolor="#B2182B", linewidth=2.6,
               zorder=8)
    ax.annotate(
        f"$f^* = {bk.frequency_hz:.1f}$ Hz\n"
        f"$A^* = {bk.amplitude:.0f}$ N\n"
        f"coverage $= {bk.detachment_coverage:.2f}$\n"
        f"$\\sigma_{{\\max}} = {by:.2f}$ MPa\n"
        f"clamp: {best.display_label}",
        xy=(bx, by),
        xytext=(-150, -55), textcoords="offset points",
        fontsize=10, color="#B2182B",
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec="#B2182B", lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.0),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Detachment coverage")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa, log]")
    ax.set_title(
        f"Multi-clamp Pareto — {result.label} "
        f"({result.n_fruits} fruits, resonance {result.f_resonance:.1f} Hz)"
    )
    ax.set_xlim(-0.03, max(0.75, best_cov_max * 1.10))
    if np.isfinite(sigma_lo) and sigma_lo > 0:
        ax.set_ylim(sigma_lo * 0.5, sigma_hi * 1.5)
    ax.legend(loc="lower right", fontsize=9,
              ncol=2 if len(result.clamps) > 5 else 1)

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


def _save_frf(result: TreeResult, stem: Path) -> None:
    import matplotlib.pyplot as plt

    freqs, mags = result.freqs, result.mags
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    ax.axvspan(
        _FEASIBLE_BAND_HZ[0], _FEASIBLE_BAND_HZ[1],
        facecolor="#cfe0ee", alpha=0.35, zorder=0,
        label=f"feasible band ({_FEASIBLE_BAND_HZ[0]:.0f}–"
              f"{_FEASIBLE_BAND_HZ[1]:.0f} Hz)",
    )

    ax.semilogy(
        freqs, mags * 1.0e3,
        color="#2166AC", linewidth=1.4,
        marker="o", markersize=3.2,
        markerfacecolor="#2166AC", markeredgecolor="white", markeredgewidth=0.5,
        zorder=2,
    )

    ax.axvline(result.f_resonance, color="#B2182B",
               linestyle="--", linewidth=1.0, alpha=0.85, zorder=3)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title(f"FRF sweep — {result.label} (F = 10 N at trunk mid)")
    ax.set_xlim(0.0, freqs.max())
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")
    ax.legend(loc="upper right", fontsize=10)

    y_min, y_max = ax.get_ylim()
    label_y = y_min * (y_max / y_min) ** 0.06
    ax.text(
        result.f_resonance + 0.4, label_y,
        f"resonance ≈ {result.f_resonance:.1f} Hz",
        color="#B2182B", fontsize=11,
        ha="left", va="bottom", zorder=5,
    )

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
#  Figures (cross-tree comparison)
# ────────────────────────────────────────────────────────────────────────────
_TREE_PALETTE = [
    "#2166AC",  # tree_1 blue
    "#1B7837",  # tree_2 green
    "#B2182B",  # tree_3 red
    "#762A83",  # tree_4 purple
    "#E08214",  # tree_5 orange
]


def _save_all_pareto_overlay(results: list[TreeResult], stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    cov_max_all = 0.0
    for i, r in enumerate(results):
        front = r.best.front
        cov = -front.objectives[:, 0]
        sigma = front.objectives[:, 1] / 1.0e6
        nd = front.non_dominated_index
        if nd.size == 0:
            continue
        order = np.argsort(cov[nd])
        color = _TREE_PALETTE[i % len(_TREE_PALETTE)]

        ax.plot(cov[nd][order], sigma[nd][order],
                color=color, linewidth=1.4, alpha=0.65, zorder=2)
        ax.scatter(cov[nd], sigma[nd],
                   color=color, s=68, edgecolors="white", linewidths=0.9,
                   label=f"{r.label} ({r.best.display_label}, "
                         f"resonance {r.f_resonance:.1f} Hz)", zorder=3)
        k = nd[front.knee_index]
        ax.scatter([cov[k]], [sigma[k]],
                   s=230, marker="o",
                   facecolor="none", edgecolor=color, linewidth=2.0,
                   zorder=5)
        cov_max_all = max(cov_max_all, float(cov.max()))

    ax.set_yscale("log")
    ax.set_xlabel("Detachment coverage")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa, log]")
    ax.set_title("Best-clamp Pareto fronts — 5 trees")
    ax.set_xlim(-0.03, max(0.75, cov_max_all * 1.08))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


def _save_all_frf_overlay(results: list[TreeResult], stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axvspan(
        _FEASIBLE_BAND_HZ[0], _FEASIBLE_BAND_HZ[1],
        facecolor="#cfe0ee", alpha=0.30, zorder=0,
        label=f"feasible band ({_FEASIBLE_BAND_HZ[0]:.0f}–"
              f"{_FEASIBLE_BAND_HZ[1]:.0f} Hz)",
    )
    for i, r in enumerate(results):
        color = _TREE_PALETTE[i % len(_TREE_PALETTE)]
        ax.semilogy(r.freqs, r.mags * 1.0e3,
                    color=color, linewidth=1.3, alpha=0.85,
                    label=f"{r.label} (resonance {r.f_resonance:.1f} Hz)",
                    zorder=2)
        in_band = (r.freqs >= _FEASIBLE_BAND_HZ[0]) & \
                  (r.freqs <= _FEASIBLE_BAND_HZ[1])
        ax.scatter([r.f_resonance], [r.mags[r.freqs.tolist().index(r.f_resonance)] * 1e3],
                   color=color, s=70, edgecolors="white", linewidths=0.8,
                   zorder=4)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title("FRF sweeps across 5 trees — markers at in-band resonance")
    ax.set_xlim(0.0, max(r.freqs.max() for r in results))
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


def _save_knees_summary(results: list[TreeResult], stem: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [r.label for r in results]
    f_stars = [r.best.knee.frequency_hz for r in results]
    a_stars = [r.best.knee.amplitude for r in results]
    covs = [r.best.knee.detachment_coverage for r in results]
    sigmas = [r.best.knee.trunk_max_stress / 1.0e6 for r in results]
    clamps = [r.best.display_label for r in results]
    colors = [_TREE_PALETTE[i % len(_TREE_PALETTE)] for i in range(len(results))]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))

    for ax, values, ylabel, title in (
        (axes[0, 0], f_stars, "f* [Hz]", "Knee drive frequency"),
        (axes[0, 1], a_stars, "A* [N]",  "Knee force amplitude"),
        (axes[1, 0], covs,    "Detachment coverage", "Knee coverage"),
        (axes[1, 1], sigmas,  r"$\sigma_{\mathrm{max}}$ [MPa]", "Knee trunk stress"),
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.62)
        for bar, val in zip(bars, values):
            txt = f"{val:.2f}" if val < 1.0 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    txt, ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add a footer legend listing the best clamp for each tree.
    legend_lines = "    ".join(
        f"{lbl}: {c}" for lbl, c in zip(labels, clamps)
    )
    fig.text(
        0.5, 0.01,
        f"Best clamps  —  {legend_lines}",
        ha="center", va="bottom",
        fontsize=9.5, color="#444444",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f6f6f6",
                  ec="#cccccc", lw=0.6),
    )

    fig.suptitle("Best-clamp knee recommendations across 5 trees",
                 fontsize=14, y=1.00)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    _save_both(fig, stem)
    plt.close(fig)


def _print_recommendation_table(results: list[TreeResult]) -> None:
    if not results:
        return
    print("\n┌────────┬─────────┬─────────────┬─────────┬────────┬──────────┬─────────────┐")
    print("│  Tree  │ Fruits  │ Best clamp  │  f* Hz  │  A* N  │ Coverage │ σ_max [MPa] │")
    print("├────────┼─────────┼─────────────┼─────────┼────────┼──────────┼─────────────┤")
    for r in results:
        b = r.best
        k = b.knee
        print(
            f"│ {r.label:>6} │ {r.n_fruits:>7} │ {b.display_label:<11} │ "
            f"{k.frequency_hz:>7.2f} │ {k.amplitude:>6.0f} │ "
            f"{k.detachment_coverage:>8.2f} │ {k.trunk_max_stress/1e6:>11.3f} │"
        )
    print("└────────┴─────────┴─────────────┴─────────┴────────┴──────────┴─────────────┘")


if __name__ == "__main__":
    raise SystemExit(main())
