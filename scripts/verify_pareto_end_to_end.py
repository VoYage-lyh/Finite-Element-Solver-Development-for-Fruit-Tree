"""End-to-end self-consistency check for the redesigned Pareto evaluator.

Tree_3 uses a harmonic-force excitation. We:
    1. Run a wide FRF sweep to find the dominant resonance.
    2. Tighten the detachment displacement to 2 mm (default 10 mm makes the
       detachment force unrealistically high for tree_3's fruit stiffness).
    3. Build a Pareto grid spanning force amplitudes (N) and a band around the
       resonance, evaluate (coverage, σ_max), find the Pareto knee, save the
       scatter plot.
    4. Sanity check: σ is linear in A at fixed f (linear FRF property), and
       coverage is non-decreasing in A.

Figures are exported to ``outputs/`` in both PNG and PDF formats with a
publication-style Times New Roman appearance.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
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
#  Workflow helpers
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


def main() -> int:
    from orchard_fem.calibration.fenicsx_bridge import (
        build_fenicsx_pareto_evaluator,
    )
    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.recommendation.pareto import pareto_front_from_grid

    _apply_pub_style()

    model_path = REPO / "trees" / "tree_3.json"
    out_dir = REPO / "outputs"
    out_dir.mkdir(exist_ok=True)

    print(f"[verify] loading {model_path.name} …")
    model = load_orchard_model(str(model_path))
    print(f"[verify]   fruits={len(model.fruits)}, branches={len(model.branches)}")

    model = replace(
        model,
        fruit_policy=replace(model.fruit_policy, detachment_displacement_m=0.002),
    )
    print(f"[verify] detachment displacement set to "
          f"{model.fruit_policy.detachment_displacement_m * 1000:.1f} mm")

    print("[verify] FRF sweep 0.5–30 Hz (60 points) to find resonance …")
    t0 = time.time()
    freqs, mags = _coarse_frf_sweep(model, 0.5, 30.0, 60)
    print(f"[verify]   sweep took {time.time() - t0:.1f} s")
    peak_idx = int(np.argmax(mags))
    f_resonance = float(freqs[peak_idx])
    print(f"[verify] dominant peak: {f_resonance:.2f} Hz, "
          f"|u_tip|/F = {mags[peak_idx] / model.excitation.amplitude * 1e3:.3f} mm/N")

    theta = {
        "E": float(model.materials[0].youngs_modulus),
        "rho": float(model.materials[0].density),
    }
    print(f"[verify] θ = {theta}")

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
    print(f"\n[verify] Pareto grid: f={[round(f, 1) for f in f_grid]} Hz "
          f"× A={A_grid} N  = {len(f_grid) * len(A_grid)} FE solves\n")

    t0 = time.time()
    table: list[tuple[float, float, float, float]] = []
    print(f"{'f [Hz]':>8}  {'A [N]':>8}  {'coverage':>10}  {'σ_max [MPa]':>14}")
    print("-" * 48)
    for f in f_grid:
        for A in A_grid:
            obj = evaluator(theta, float(f), float(A), clamp_label)
            cov = float(obj.detachment_coverage)
            stress = float(obj.trunk_max_stress)
            table.append((f, A, cov, stress))
            print(f"{f:8.2f}  {A:8.1f}  {cov:10.4f}  {stress / 1.0e6:14.4f}")
    elapsed = time.time() - t0
    print(f"\n[verify] {len(table)} FE solves in {elapsed:.1f} s "
          f"({elapsed / len(table):.2f} s/point)")

    front = pareto_front_from_grid(
        clamp_label, f_grid, A_grid, evaluator, theta,
    )
    nd = front.non_dominated_index
    knee = front.knee
    print(f"\n[verify] non-dominated points: {nd.size} / {len(table)}")
    print(f"[verify] knee: f*={knee.frequency_hz:.2f} Hz, "
          f"A*={knee.amplitude:.1f} N, "
          f"coverage={knee.detachment_coverage:.3f}, "
          f"σ={knee.trunk_max_stress / 1.0e6:.4f} MPa")

    _save_pareto_scatter(front, out_dir / "verify_pareto")
    print(f"[verify] Pareto figure saved → outputs/verify_pareto.{{png,pdf}}")

    _save_frf(freqs, mags, f_resonance, out_dir / "verify_frf")
    print(f"[verify] FRF figure saved → outputs/verify_frf.{{png,pdf}}")

    _physics_checks(table)
    print("\n[verify] all physics sanity checks passed.")
    return 0


# ────────────────────────────────────────────────────────────────────────────
#  Figures
# ────────────────────────────────────────────────────────────────────────────
def _save_pareto_scatter(front, stem: Path) -> None:
    import matplotlib.pyplot as plt

    cov = -front.objectives[:, 0]
    sigma = front.objectives[:, 1] / 1.0e6
    nd = front.non_dominated_index
    is_nd = np.zeros(cov.size, dtype=bool)
    is_nd[nd] = True

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    # Dominated cloud
    ax.scatter(
        cov[~is_nd], sigma[~is_nd],
        c="#b8b8b8", s=46, alpha=0.7,
        edgecolors="white", linewidths=0.8,
        label="Dominated",
    )

    # Pareto front
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

        # Label each Pareto-front candidate with its (f, A)
        for i in nd:
            ax.annotate(
                f"{front.frequencies_hz[i]:.0f} Hz / {front.amplitudes[i]:.0f} N",
                xy=(cov[i], sigma[i]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color="#2166AC",
            )

        # Knee marker
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
    ax.set_title("Pareto trade-off — coverage vs. trunk damage")
    ax.set_xlim(-0.03, max(0.7, cov.max() * 1.08))
    ax.set_ylim(-0.05, sigma.max() * 1.10)
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


def _save_frf(freqs, mags, f_peak, stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogy(
        freqs, mags * 1.0e3,
        color="#2166AC", linewidth=1.4,
        marker="o", markersize=3.2,
        markerfacecolor="#2166AC", markeredgecolor="white", markeredgewidth=0.5,
    )

    # Resonance marker
    ax.axvline(f_peak, color="#B2182B", linestyle="--", linewidth=1.0, alpha=0.85)
    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"resonance ≈ {f_peak:.1f} Hz",
        xy=(f_peak, mags[int(np.argmax(mags))] * 1e3),
        xytext=(12, -4), textcoords="offset points",
        color="#B2182B", fontsize=11,
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title("FRF sweep — force excitation (F = 10 N at trunk mid)")
    ax.set_xlim(0.0, freqs.max())
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


def _physics_checks(table: list[tuple[float, float, float, float]]) -> None:
    by_f: dict[float, list[tuple[float, float, float]]] = {}
    for f, A, cov, stress in table:
        by_f.setdefault(f, []).append((A, cov, stress))

    for f, rows in by_f.items():
        rows.sort()
        amps = np.array([r[0] for r in rows])
        sigmas = np.array([r[2] for r in rows])
        if sigmas[0] > 1.0:
            ratios = sigmas / sigmas[0]
            expected = amps / amps[0]
            err = np.abs(ratios - expected) / np.maximum(expected, 1.0e-9)
            assert err.max() < 0.05, (
                f"Linearity violation at f={f}: σ/σ0 = {ratios} "
                f"vs A/A0 = {expected}, rel err {err.max():.2%}"
            )

        coverages = np.array([r[1] for r in rows])
        assert np.all(np.diff(coverages) >= -1.0e-9), (
            f"Coverage decreased with A at f={f}: {coverages}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
