"""Isolate the contribution of the quadratic damping term c2|v|v.

In the harmonic-balance / FRF context, the cubic stiffness k3 u^3 shows up
even at modest amplitude (its first-harmonic equivalent stiffness is
proportional to A^2), whereas the c2|v|v term enters as an equivalent linear
damping c_eq = (8/3 pi) c2 omega A that only matters once the response
velocity is large.  To isolate it we run three FRF sweeps per tree:

* linear     — no cubic, no quadratic damping
* cubic only — random cubic stiffness on, quadratic damping zeroed
* full       — random cubic + quadratic damping (current default)

The cubic_only -> full delta is the quadratic damping contribution.  We
expect the response peak to drop and the peak to widen (added dissipation).

Outputs (under ``results/verification/``):

* ``c2_effect_tree_<n>.{png,pdf}`` — three-curve overlay per tree.
* ``summary_c2_suppression.{png,pdf}`` — peak amplitude reduction across trees.
* ``summary_c2_suppression.csv``      — table per tree.
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
import orchard_fem.fenicsx.joints as joints_module
import orchard_fem.fenicsx.assembly as assembly_module


_FEASIBLE_BAND_HZ = (3.0, 20.0)
_NARROW_BAND_HALF_WIDTH = 0.40
_SWEEP_STEPS = 120


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


# ---------------------------------------------------------------------------
#  Model variants
# ---------------------------------------------------------------------------
def _strip_all_nonlinear(model):
    new_clamps = [replace(c, cubic_stiffness=0.0) for c in model.clamps]
    new_analysis = replace(
        model.analysis,
        auto_nonlinear_levels=[],
        auto_nonlinear_cubic_scale=0.0,
        auto_nonlinear_randomize=False,
    )
    return replace(model, clamps=new_clamps, analysis=new_analysis)


_ORIGINAL_AUTO_INJECT = joints_module.augment_operators_with_auto_nonlinear_links
_ORIGINAL_CLAMP_INJECT = joints_module.augment_operators_with_nonlinear_clamps


def _install_c2_zeroing_patch() -> None:
    """Force every nonlinear link (auto-joint AND clamp) to have c2 = 0.

    Cubic stiffness k3 is preserved: same RNG seed → same hardening/softening
    selection and same k3 values. The patch isolates the quadratic-damping
    contribution by removing it from both the auto-joint links and the
    grounded clamp links.
    """
    from dataclasses import replace as dc_replace

    def _zero_c2(bundle):
        zero_c2 = [
            dc_replace(link, quadratic_damping=0.0)
            for link in bundle.nonlinear_links
        ]
        return dc_replace(bundle, nonlinear_links=zero_c2)

    def auto_patched(model, space_bundle, mesh_spec, operator_bundle):
        bundle = _ORIGINAL_AUTO_INJECT(model, space_bundle, mesh_spec, operator_bundle)
        return _zero_c2(bundle)

    def clamp_patched(model, space_bundle, mesh_spec, operator_bundle):
        bundle = _ORIGINAL_CLAMP_INJECT(model, space_bundle, mesh_spec, operator_bundle)
        return _zero_c2(bundle)

    joints_module.augment_operators_with_auto_nonlinear_links = auto_patched
    assembly_module.augment_operators_with_auto_nonlinear_links = auto_patched
    joints_module.augment_operators_with_nonlinear_clamps = clamp_patched
    assembly_module.augment_operators_with_nonlinear_clamps = clamp_patched


def _uninstall_c2_zeroing_patch() -> None:
    joints_module.augment_operators_with_auto_nonlinear_links = _ORIGINAL_AUTO_INJECT
    assembly_module.augment_operators_with_auto_nonlinear_links = _ORIGINAL_AUTO_INJECT
    joints_module.augment_operators_with_nonlinear_clamps = _ORIGINAL_CLAMP_INJECT
    assembly_module.augment_operators_with_nonlinear_clamps = _ORIGINAL_CLAMP_INJECT


# ---------------------------------------------------------------------------
#  FRF helpers
# ---------------------------------------------------------------------------
def _frf_tip_mean(model, f_min: float, f_max: float, steps: int):
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
        raise RuntimeError("No '_tip_ux' observation channels found.")
    mags = np.array([
        float(np.mean([p.observation_magnitudes[name_to_idx[n]] for n in tip_obs]))
        for p in res.points
    ])
    return freqs, mags


def _peak_in_band(freqs: np.ndarray, mags: np.ndarray, band: tuple[float, float]):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return float("nan"), float("nan")
    idx_local = int(np.argmax(mags[mask]))
    idx = int(np.flatnonzero(mask)[idx_local])
    return float(freqs[idx]), float(mags[idx])


def _coarse_resonance(model):
    """Sweep 0.5-30 Hz to pick the dominant peak in the feasible band."""
    f, m = _frf_tip_mean(model, 0.5, 30.0, 90)
    fp, _ = _peak_in_band(f, m, _FEASIBLE_BAND_HZ)
    return fp


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def _plot_overlay(out_stem: Path, *, tree_label: str, curves: dict, peaks: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"linear": "#2166AC", "cubic_only": "#762A83", "full": "#B2182B"}
    style = {
        "linear":     (r"Linear ($k_3 = 0,\ c_2 = 0$)", "-", 1.6),
        "cubic_only": (r"$k_3 \neq 0,\ c_2 = 0$",       "--", 1.7),
        "full":       (r"$k_3 \neq 0,\ c_2 \neq 0$",     "-", 2.0),
    }
    for tag in ("linear", "cubic_only", "full"):
        f, m = curves[tag]
        label, ls, lw = style[tag]
        ax.semilogy(f, m, color=colors[tag], ls=ls, lw=lw, label=label)
        fp, mp = peaks[tag]
        if not (np.isnan(fp) or np.isnan(mp)):
            ax.plot([fp], [mp], marker="o", color=colors[tag], ms=5)

    drop_pct = float("nan")
    if not (np.isnan(peaks["cubic_only"][1]) or np.isnan(peaks["full"][1])):
        m_co = peaks["cubic_only"][1]
        m_full = peaks["full"][1]
        if m_co > 0:
            drop_pct = (m_co - m_full) / m_co * 100.0
    ax.set_xlabel(r"Frequency $f$ [Hz]")
    ax.set_ylabel(r"$|U_{\rm tip}|$ at $F=10$ N excitation [m]")
    ax.set_title(
        f"{tree_label}: $c_2$ peak suppression "
        rf"$={drop_pct:+.1f}\%$"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def _plot_summary(out_stem: Path, rows: list[tuple]) -> None:
    labels = [f"tree_{r[0]}" for r in rows]
    drops = np.array([r[5] for r in rows])
    mean_drop = float(np.nanmean(drops))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(labels, drops, color="#d62728", edgecolor="#333333", linewidth=0.7)
    for bar, val in zip(bars, drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + (0.6 if val >= 0 else -1.2),
            f"{val:+.1f}%",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=11,
        )
    ax.axhline(mean_drop, color="#333333", ls="--", lw=1.0,
               label=f"mean = {mean_drop:+.2f}\\%")
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_ylabel(r"Peak suppression by $c_2$ [\%]")
    ax.set_title(r"$c_2$ peak reduction ($k_3 \neq 0, c_2 = 0 \rightarrow k_3 \neq 0, c_2 \neq 0$)")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> int:
    _apply_pub_style()
    out_dir = REPO / "results" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple] = []
    for n in (1, 2, 3, 4, 5):
        path = REPO / "trees" / f"tree_{n}.json"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        print(f"\n[tree_{n}] loading {path.name}")
        model = load_orchard_model(str(path))

        # 1. Coarse sweep on the full nonlinear model to find the resonance.
        t0 = time.time()
        f_nl = _coarse_resonance(model)
        print(f"[tree_{n}] coarse resonance @ {f_nl:.2f} Hz ({time.time()-t0:.1f}s)")
        band = (
            max(0.5, f_nl * (1.0 - _NARROW_BAND_HALF_WIDTH)),
            f_nl * (1.0 + _NARROW_BAND_HALF_WIDTH),
        )

        curves: dict = {}
        peaks: dict = {}

        # 2. linear baseline.
        t0 = time.time()
        f, m = _frf_tip_mean(_strip_all_nonlinear(model), *band, _SWEEP_STEPS)
        curves["linear"] = (f, m)
        peaks["linear"] = _peak_in_band(f, m, _FEASIBLE_BAND_HZ)
        print(f"[tree_{n}]   linear      peak @ {peaks['linear'][0]:.2f} Hz, "
              f"|U|={peaks['linear'][1]:.3e}  ({time.time()-t0:.1f}s)")

        # 3. cubic_only (full nonlinear with quadratic_damping forced to 0).
        _install_c2_zeroing_patch()
        try:
            t0 = time.time()
            f, m = _frf_tip_mean(model, *band, _SWEEP_STEPS)
            curves["cubic_only"] = (f, m)
            peaks["cubic_only"] = _peak_in_band(f, m, _FEASIBLE_BAND_HZ)
            print(f"[tree_{n}]   cubic-only  peak @ {peaks['cubic_only'][0]:.2f} Hz, "
                  f"|U|={peaks['cubic_only'][1]:.3e}  ({time.time()-t0:.1f}s)")
        finally:
            _uninstall_c2_zeroing_patch()

        # 4. full (cubic + c2).
        t0 = time.time()
        f, m = _frf_tip_mean(model, *band, _SWEEP_STEPS)
        curves["full"] = (f, m)
        peaks["full"] = _peak_in_band(f, m, _FEASIBLE_BAND_HZ)
        print(f"[tree_{n}]   full        peak @ {peaks['full'][0]:.2f} Hz, "
              f"|U|={peaks['full'][1]:.3e}  ({time.time()-t0:.1f}s)")

        # 5. summary stats.
        fp_co, mp_co = peaks["cubic_only"]
        fp_full, mp_full = peaks["full"]
        suppression_pct = float("nan")
        freq_shift_pct = float("nan")
        if mp_co > 0:
            suppression_pct = (mp_co - mp_full) / mp_co * 100.0
        if fp_co > 0:
            freq_shift_pct = (fp_full - fp_co) / fp_co * 100.0
        rows.append((n, fp_co, fp_full, mp_co, mp_full, suppression_pct,
                     freq_shift_pct))

        _plot_overlay(out_dir / f"c2_effect_tree_{n}",
                      tree_label=f"tree_{n}", curves=curves, peaks=peaks)
        print(f"[tree_{n}]   suppression = {suppression_pct:+.2f}%  "
              f"frequency shift = {freq_shift_pct:+.2f}%")

    if not rows:
        print("[error] no trees processed")
        return 1

    _plot_summary(out_dir / "summary_c2_suppression", rows)

    csv_path = out_dir / "summary_c2_suppression.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tree", "f_cubic_only_Hz", "f_full_Hz",
                    "peak_mag_cubic_only", "peak_mag_full",
                    "suppression_percent", "freq_shift_percent"])
        for row in rows:
            w.writerow(row)

    drops = np.array([r[5] for r in rows])
    shifts = np.array([r[6] for r in rows])
    print()
    print("=" * 72)
    print(f"Mean peak suppression by c2 : {float(np.nanmean(drops)):+.2f} % "
          f"(positive = damping clipped the peak)")
    print(f"Mean further freq shift     : {float(np.nanmean(shifts)):+.2f} %  "
          "(cubic-only -> full)")
    print(f"Summary CSV  : {csv_path.relative_to(REPO)}")
    print(f"Per-tree fig : {out_dir.relative_to(REPO)}/c2_effect_tree_<n>.{{png,pdf}}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
