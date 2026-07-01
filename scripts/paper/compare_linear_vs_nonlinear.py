"""Compare linear-baseline vs nonlinear-corrected FRFs on the five sample trees.

For each tree we run two FRF sweeps over the same band:

1. **Linear baseline** — auto-injected cubic + quadratic links and the
   clamp cubic stiffness are zeroed, so the embedded-beam system is purely
   linear.
2. **Nonlinear** — the tree's JSON settings are used as-is, with
   ``auto_nonlinear_randomize: true`` driving the manuscript-calibrated
   Duffing-type link sampler.

The resonance peaks in the harvesting band (3-20 Hz) are extracted via the
same prominence-based detector that ``verify_pareto_end_to_end.py`` uses for
its main pipeline, so the shift number is directly comparable.

Outputs (under ``results/verification/``):

* ``frequency_shift_tree_<n>.{png,pdf}`` — overlay of linear vs nonlinear FRF
  with resonance markers and the shift annotation.
* ``summary_frequency_shift.{png,pdf}`` — bar chart of Δf/f_lin across the
  five trees, with the manuscript reference (-11.0 ± 4.1 %) overlaid.
* ``summary_frequency_shift.csv`` — table per tree (f_lin, f_nl, shift %, peak
  magnitudes).

Run::

    conda run -n orchard-fenicsx python scripts/compare_linear_vs_nonlinear.py
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard_fem.io.loaders.orchard import load_orchard_model
from orchard_fem.fenicsx.frequency_response import (
    solve_embedded_beam_frequency_response_experiment,
)


_FEASIBLE_BAND_HZ = (3.0, 20.0)
_SWEEP_BAND_HZ = (0.5, 30.0)
_SWEEP_STEPS = 90


def _apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fontsize": 11,
        "legend.edgecolor": "#cccccc",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _strip_nonlinear(model):
    """Return a copy of *model* with all nonlinear contributions disabled.

    Three sources of nonlinearity are removed:

    * ``clamps[*].cubic_stiffness`` set to 0.
    * ``analysis.auto_nonlinear_levels`` cleared (no auto-injected cubic /
      quadratic-damping joint links).
    * ``analysis.auto_nonlinear_randomize`` set to ``False`` for safety.
    """
    new_clamps = [replace(c, cubic_stiffness=0.0) for c in model.clamps]
    new_analysis = replace(
        model.analysis,
        auto_nonlinear_levels=[],
        auto_nonlinear_cubic_scale=0.0,
        auto_nonlinear_randomize=False,
    )
    return replace(model, clamps=new_clamps, analysis=new_analysis)


def _frf_tip_mean(model, f_min: float, f_max: float, steps: int):
    """Sweep the FRF and return (freqs, tip-ux-mean magnitudes)."""
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
    if not tip_obs:
        raise RuntimeError("No '_tip_ux' observation channels found in the FRF result.")
    mags = np.array([
        float(np.mean([p.observation_magnitudes[name_to_idx[n]] for n in tip_obs]))
        for p in res.points
    ])
    return freqs, mags


def _find_in_band_resonance(
    freqs: np.ndarray,
    mags: np.ndarray,
    band: tuple[float, float],
    *,
    prominence_ratio: float = 0.10,
) -> tuple[int, bool]:
    """Same algorithm as verify_pareto_end_to_end._find_in_band_resonance."""
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        raise RuntimeError(f"No FRF samples in {band[0]}-{band[1]} Hz.")

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

    curvature = np.zeros_like(mags)
    curvature[1:-1] = log_mags[2:] - 2.0 * log_mags[1:-1] + log_mags[:-2]
    band_idx = np.flatnonzero(in_band)
    interior = band_idx[(band_idx > 0) & (band_idx < mags.size - 1)]
    if interior.size == 0:
        peak_idx = int(band_idx[int(np.argmax(mags[in_band]))])
        return peak_idx, False
    peak_idx = int(interior[int(np.argmin(curvature[interior]))])
    return peak_idx, False


def _plot_overlay(
    out_stem: Path,
    *,
    tree_label: str,
    f_lin: np.ndarray,
    m_lin: np.ndarray,
    f_nl: np.ndarray,
    m_nl: np.ndarray,
    fp_lin: float,
    fp_nl: float,
    shift_pct: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogy(
        f_lin, m_lin,
        color="#2166AC", lw=1.8, label=r"Linear ($k_3 = 0,\ c_2 = 0$)",
    )
    ax.semilogy(
        f_nl, m_nl,
        color="#B2182B", lw=1.8, label=r"Nonlinear ($k_3 \neq 0,\ c_2 \neq 0$)",
    )
    ax.axvline(fp_lin, color="#2166AC", ls=":", alpha=0.7, lw=1.2)
    ax.axvline(fp_nl, color="#B2182B", ls=":", alpha=0.7, lw=1.2)
    ax.axvspan(
        _FEASIBLE_BAND_HZ[0], _FEASIBLE_BAND_HZ[1],
        color="#888888", alpha=0.06,
        label=rf"feasible band ({_FEASIBLE_BAND_HZ[0]:.0f}--"
              rf"{_FEASIBLE_BAND_HZ[1]:.0f} Hz)",
    )
    ax.set_xlabel(r"Frequency $f$ [Hz]")
    ax.set_ylabel(r"$|U_{\rm tip}|$ at $F=10$ N excitation [m]")
    ax.set_title(
        f"{tree_label}: "
        rf"$f_{{\rm lin}}={fp_lin:.2f}$ Hz, "
        rf"$f_{{\rm nl}}={fp_nl:.2f}$ Hz, "
        rf"$\Delta f/f_{{\rm lin}}={shift_pct:+.2f}\%$"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def _plot_summary(out_stem: Path, rows: list[tuple]) -> None:
    labels = [f"tree_{r[0]}" for r in rows]
    shifts = np.array([r[3] for r in rows])
    mean_shift = float(np.mean(shifts))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#2ca02c" if s < 0 else "#d62728" for s in shifts]
    bars = ax.bar(labels, shifts, color=colors, edgecolor="#333333", linewidth=0.7)
    for bar, val in zip(bars, shifts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + (0.4 if val >= 0 else -0.9),
            f"{val:+.1f}%",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=11,
        )
    ax.axhline(mean_shift, color="#333333", ls="--", lw=1.0,
               label=f"mean = {mean_shift:+.2f} %")
    ax.axhspan(-11.0 - 4.1, -11.0 + 4.1, color="#ff9896", alpha=0.18,
               label="manuscript band (-11.0 ± 4.1 %)")
    ax.axhline(-11.0, color="#d62728", ls=":", lw=1.0, alpha=0.8)
    ax.set_ylabel(r"Resonance shift $\Delta f / f_{\rm lin}$ [\%]")
    ax.set_title(r"Linear $\rightarrow$ nonlinear resonance shift across five trees")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    _apply_pub_style()
    out_dir = REPO / "results" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple] = []
    for n in (1, 2, 3, 4, 5):
        model_path = REPO / "trees" / f"tree_{n}.json"
        if not model_path.exists():
            print(f"[skip] {model_path} not found")
            continue
        print(f"\n[tree_{n}] loading {model_path.name}")
        model = load_orchard_model(str(model_path))

        t0 = time.time()
        print(f"[tree_{n}] linear sweep ({_SWEEP_BAND_HZ[0]}-{_SWEEP_BAND_HZ[1]} Hz, "
              f"{_SWEEP_STEPS} pts) ...")
        f_lin, m_lin = _frf_tip_mean(_strip_nonlinear(model), *_SWEEP_BAND_HZ, _SWEEP_STEPS)
        print(f"[tree_{n}]   linear sweep {time.time() - t0:.1f} s")

        t1 = time.time()
        print(f"[tree_{n}] nonlinear sweep ...")
        f_nl, m_nl = _frf_tip_mean(model, *_SWEEP_BAND_HZ, _SWEEP_STEPS)
        print(f"[tree_{n}]   nonlinear sweep {time.time() - t1:.1f} s")

        idx_lin, _ = _find_in_band_resonance(f_lin, m_lin, _FEASIBLE_BAND_HZ)
        idx_nl, _ = _find_in_band_resonance(f_nl, m_nl, _FEASIBLE_BAND_HZ)
        fp_lin = float(f_lin[idx_lin])
        fp_nl = float(f_nl[idx_nl])
        mp_lin = float(m_lin[idx_lin])
        mp_nl = float(m_nl[idx_nl])
        shift_pct = (fp_nl - fp_lin) / fp_lin * 100.0
        peak_ratio = mp_nl / mp_lin if mp_lin > 0 else float("nan")
        rows.append((n, fp_lin, fp_nl, shift_pct, mp_lin, mp_nl, peak_ratio))

        print(f"[tree_{n}]   f_lin={fp_lin:.3f} Hz, f_nl={fp_nl:.3f} Hz, "
              f"shift={shift_pct:+.2f} %, peak ratio nl/lin = {peak_ratio:.3f}")

        _plot_overlay(
            out_dir / f"frequency_shift_tree_{n}",
            tree_label=f"tree_{n}",
            f_lin=f_lin, m_lin=m_lin,
            f_nl=f_nl, m_nl=m_nl,
            fp_lin=fp_lin, fp_nl=fp_nl,
            shift_pct=shift_pct,
        )

    if not rows:
        print("[error] no trees processed")
        return 1

    _plot_summary(out_dir / "summary_frequency_shift", rows)

    csv_path = out_dir / "summary_frequency_shift.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tree", "f_lin_Hz", "f_nl_Hz", "shift_percent",
                    "peak_mag_lin", "peak_mag_nl", "peak_ratio_nl_over_lin"])
        for row in rows:
            w.writerow(row)

    shifts = np.array([r[3] for r in rows])
    print()
    print("=" * 64)
    print(f"Mean resonance shift : {float(np.mean(shifts)):+.2f} %")
    print(f"Std  resonance shift : {float(np.std(shifts)):.2f} %")
    print("Manuscript reference : -11.0 ± 4.1 % (Liu et al., Table 4)")
    print(f"Summary CSV          : {csv_path.relative_to(REPO)}")
    print(f"Per-tree overlays    : {out_dir.relative_to(REPO)}/frequency_shift_tree_<n>.{{png,pdf}}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
