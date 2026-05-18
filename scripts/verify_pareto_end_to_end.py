"""End-to-end self-consistency check across the full tree set (tree_1–tree_5).

For each tree we:
    1. Run a wide FRF sweep to find the dominant resonance.
    2. Tighten the detachment displacement to 2 mm (default 10 mm makes the
       detachment force unrealistically high for these slender shoots).
    3. Build a Pareto grid spanning force amplitudes (N) and a band around the
       resonance, evaluate (coverage, σ_max), find the Pareto knee.

Per-tree outputs (in ``outputs/``):
    * ``verify_pareto_tree_<n>.{png,pdf}`` — Pareto scatter with knee.
    * ``verify_frf_tree_<n>.{png,pdf}``    — FRF sweep with resonance marker.

Cross-tree comparison outputs:
    * ``verify_pareto_all_trees.{png,pdf}`` — all 5 Pareto fronts in one axes.
    * ``verify_frf_all_trees.{png,pdf}``    — all 5 FRF curves stacked.
    * ``verify_knees_summary.{png,pdf}``    — knee (f*, A*) per tree with bars.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


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
        "legend.fontsize": 11,
        "legend.edgecolor": "#cccccc",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_both(fig, stem: Path) -> None:
    """Save *fig* as both ``stem.png`` and ``stem.pdf``."""
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))


# ────────────────────────────────────────────────────────────────────────────
#  Workflow data containers
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class TreeResult:
    label: str
    n_fruits: int
    freqs: np.ndarray
    mags: np.ndarray
    f_resonance: float
    front: object  # ParetoFront
    knee: object   # ParetoKnee


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

    print(f"[{label}] FRF sweep 0.5–30 Hz to locate resonance …")
    t0 = time.time()
    freqs, mags = _coarse_frf_sweep(model, 0.5, 30.0, 60)
    # Restrict peak search to the engineering-feasible band: below 3 Hz the
    # response is dominated by the quasi-static K^{-1} regime (not a real
    # mode), and above 20 Hz typical eccentric-cam shakers can't reach.
    PEAK_BAND_HZ = (3.0, 20.0)
    in_band = (freqs >= PEAK_BAND_HZ[0]) & (freqs <= PEAK_BAND_HZ[1])
    if not in_band.any():
        raise RuntimeError(
            f"No FRF samples in the {PEAK_BAND_HZ[0]}–{PEAK_BAND_HZ[1]} Hz "
            f"band; widen the sweep or relax the band."
        )
    band_idx = np.flatnonzero(in_band)
    peak_idx = int(band_idx[int(np.argmax(mags[in_band]))])
    f_resonance = float(freqs[peak_idx])
    print(f"[{label}]   sweep {time.time() - t0:.1f} s   peak (in "
          f"{PEAK_BAND_HZ[0]:.0f}–{PEAK_BAND_HZ[1]:.0f} Hz band) @ "
          f"{f_resonance:.2f} Hz "
          f"({mags[peak_idx] / model.excitation.amplitude * 1e3:.3f} mm/N)")

    theta = {
        "E": float(model.materials[0].youngs_modulus),
        "rho": float(model.materials[0].density),
    }
    evaluator = build_fenicsx_pareto_evaluator(model, amplitude_unit="m")

    clamp_label = "trunk@0.50"
    f_grid = [
        max(0.5, f_resonance - 2.0),
        max(0.5, f_resonance - 1.0),
        f_resonance,
        f_resonance + 1.0,
        f_resonance + 2.0,
    ]
    A_grid = [50.0, 100.0, 200.0, 400.0, 800.0]
    print(f"[{label}] Pareto grid: |f|={len(f_grid)} × |A|={len(A_grid)} = "
          f"{len(f_grid) * len(A_grid)} FE solves")

    t0 = time.time()
    front = pareto_front_from_grid(
        clamp_label, f_grid, A_grid, evaluator, theta,
    )
    print(f"[{label}]   Pareto sweep {time.time() - t0:.1f} s   "
          f"non-dominated {front.non_dominated_index.size} / "
          f"{front.frequencies_hz.size}")

    knee = front.knee
    print(f"[{label}] knee: f*={knee.frequency_hz:.2f} Hz, "
          f"A*={knee.amplitude:.0f} N, "
          f"coverage={knee.detachment_coverage:.2f}, "
          f"σ={knee.trunk_max_stress / 1.0e6:.3f} MPa")

    return TreeResult(
        label=label,
        n_fruits=len(model.fruits),
        freqs=freqs,
        mags=mags,
        f_resonance=f_resonance,
        front=front,
        knee=knee,
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

        _save_pareto_scatter(result, out_dir / f"verify_pareto_tree_{n}")
        _save_frf(result, out_dir / f"verify_frf_tree_{n}")
        print(f"[tree_{n}] figures → outputs/verify_pareto_tree_{n}.{{png,pdf}} "
              f"+ verify_frf_tree_{n}.{{png,pdf}}")

    if len(results) >= 2:
        _save_all_pareto_overlay(results, out_dir / "verify_pareto_all_trees")
        _save_all_frf_overlay(results, out_dir / "verify_frf_all_trees")
        _save_knees_summary(results, out_dir / "verify_knees_summary")
        print(f"\n[summary] cross-tree figures → outputs/verify_*_all_trees.{{png,pdf}} "
              f"+ verify_knees_summary.{{png,pdf}}")

    print(f"\n[done] processed {len(results)} tree(s).")
    return 0


# ────────────────────────────────────────────────────────────────────────────
#  Figures (per-tree)
# ────────────────────────────────────────────────────────────────────────────
def _save_pareto_scatter(result: TreeResult, stem: Path) -> None:
    import matplotlib.pyplot as plt

    front = result.front
    cov = -front.objectives[:, 0]
    sigma = front.objectives[:, 1] / 1.0e6
    nd = front.non_dominated_index
    is_nd = np.zeros(cov.size, dtype=bool)
    is_nd[nd] = True

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    ax.scatter(
        cov[~is_nd], sigma[~is_nd],
        c="#b8b8b8", s=46, alpha=0.7,
        edgecolors="white", linewidths=0.8,
        label="Dominated",
    )

    if nd.size > 0:
        order = np.argsort(cov[nd])
        ax.plot(cov[nd][order], sigma[nd][order],
                color="#2166AC", linewidth=1.2, alpha=0.55, zorder=3)
        ax.scatter(
            cov[nd], sigma[nd],
            c="#2166AC", s=92,
            edgecolors="white", linewidths=1.1,
            label="Pareto front", zorder=4,
        )

        for i in nd:
            ax.annotate(
                f"{front.frequencies_hz[i]:.0f} Hz / {front.amplitudes[i]:.0f} N",
                xy=(cov[i], sigma[i]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="#2166AC",
            )

        k = nd[front.knee_index]
        ax.scatter(
            [cov[k]], [sigma[k]],
            s=340, marker="o",
            facecolor="none", edgecolor="#B2182B", linewidth=2.4,
            label="Knee", zorder=6,
        )
        ax.annotate(
            f"$f^* = {front.frequencies_hz[k]:.1f}$ Hz\n"
            f"$A^* = {front.amplitudes[k]:.0f}$ N\n"
            f"coverage $= {-front.objectives[k, 0]:.2f}$",
            xy=(cov[k], sigma[k]),
            xytext=(28, -34), textcoords="offset points",
            fontsize=11, color="#B2182B",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#B2182B", lw=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.0),
        )

    ax.set_xlabel("Detachment coverage")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa]")
    ax.set_title(f"Pareto trade-off — {result.label} ({result.n_fruits} fruits)")
    ax.set_xlim(-0.03, max(0.7, cov.max() * 1.08))
    ax.set_ylim(-0.05, sigma.max() * 1.10)
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


_FEASIBLE_BAND_HZ = (3.0, 20.0)


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

    # Place the resonance label at the bottom-right of the red line, just
    # above the x-axis, so it never collides with the legend (top-right).
    y_min, y_max = ax.get_ylim()
    label_y = y_min * (y_max / y_min) ** 0.06  # 6 % up the log axis
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
    "#2166AC",  # blue   — tree_1
    "#1B7837",  # green  — tree_2
    "#B2182B",  # red    — tree_3
    "#762A83",  # purple — tree_4
    "#E08214",  # orange — tree_5
]


def _save_all_pareto_overlay(results: list[TreeResult], stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    cov_max_all = 0.0
    for i, r in enumerate(results):
        front = r.front
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
                   label=f"{r.label} (peak {r.f_resonance:.1f} Hz)", zorder=3)

        k = nd[front.knee_index]
        ax.scatter([cov[k]], [sigma[k]],
                   s=230, marker="o",
                   facecolor="none", edgecolor=color, linewidth=2.0,
                   zorder=5)
        cov_max_all = max(cov_max_all, float(cov.max()))

    ax.set_yscale("log")
    ax.set_xlabel("Detachment coverage")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa, log]")
    ax.set_title("Pareto fronts across 5 trees — open circles mark knee points")
    ax.set_xlim(-0.03, max(0.75, cov_max_all * 1.08))
    ax.legend(loc="upper left", fontsize=10)
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
        # Plot the in-band peak marker at the actual selected frequency.
        in_band = (r.freqs >= _FEASIBLE_BAND_HZ[0]) & \
                  (r.freqs <= _FEASIBLE_BAND_HZ[1])
        peak_mag = float(r.mags[in_band].max())
        ax.scatter([r.f_resonance], [peak_mag * 1e3],
                   color=color, s=70, edgecolors="white", linewidths=0.8,
                   zorder=4)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title("FRF sweeps across 5 trees — markers indicate in-band resonance")
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
    f_stars = [r.knee.frequency_hz for r in results]
    a_stars = [r.knee.amplitude for r in results]
    covs = [r.knee.detachment_coverage for r in results]
    sigmas = [r.knee.trunk_max_stress / 1.0e6 for r in results]
    colors = [_TREE_PALETTE[i % len(_TREE_PALETTE)] for i in range(len(results))]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))

    for ax, values, ylabel, title in (
        (axes[0, 0], f_stars, "f* [Hz]", "Knee drive frequency"),
        (axes[0, 1], a_stars, "A* [N]",  "Knee force amplitude"),
        (axes[1, 0], covs,    "Detachment coverage", "Knee coverage"),
        (axes[1, 1], sigmas,  r"$\sigma_{\mathrm{max}}$ [MPa]", "Knee trunk stress"),
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.62)
        for bar, val in zip(bars, values):
            if isinstance(val, float) and val < 1.0:
                txt = f"{val:.2f}"
            else:
                txt = f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    txt, ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Knee recommendations across 5 trees", fontsize=14, y=1.00)
    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
