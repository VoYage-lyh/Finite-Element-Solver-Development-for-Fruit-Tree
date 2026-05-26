"""Overlay a measured FRF curve against the linear / nonlinear simulation FRFs.

Three curves on a common axis:

  1. **Measured**  — hammer-impact test, tip–z accelerance |H_a| read from
     ``results/hammer_test/<test>/frf_tip.csv``; converted to displacement
     compliance |H_x| = |H_a| / (2π f)² so the unit lines up with the sim.
  2. **Linear sim (k_3 = 0, c_2 = 0)** — pareto-pipeline FRF curve cached at
     ``cache/verify_pareto_results/tree_<n>.pkl``.
     Magnitudes are displacement amplitudes at the model's excitation amplitude
     F (10 N for the supplied tree JSONs), so |H_x|_sim = |U_tip| / F.
  3. **Nonlinear sim (k_3 ≠ 0, c_2 ≠ 0)** — same conversion, cached at
     ``cache/verify_pareto_results_nonlinear/tree_<n>.pkl``.

The script ALSO back-estimates physical (k_3, c_2) ranges from the
simulation's observed cubic-only frequency shift and c_2 peak suppression
using the standard Duffing describing function — see ``estimate_k3_c2``.

Output (default): a two-panel figure (|H_x| overlay above, measured
coherence γ² below) written to BOTH
  - ``<measured-dir>/compare_vs_sim_tree_<n>.{png,pdf}`` (per-test record),
  - ``results_nonlinear/verification/measured_vs_simulation_tree_<n>.{png,pdf}``
    (paper validation figure).
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Cache locations populated by verify_pareto_end_to_end.py.
PARETO_CACHE_LIN = REPO / "cache" / "verify_pareto_results"
PARETO_CACHE_NL = REPO / "cache" / "verify_pareto_results_nonlinear"

# Excitation amplitude used by the pareto pipeline (= tree_*.json "amplitude").
F_EXCITATION_N = 10.0


def _read_csv_dict(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    cols = {
        name: np.array(
            [float(row[i]) if row[i] else np.nan for row in rows]
        )
        for i, name in enumerate(header)
    }
    return header, cols


def load_measured_frf(test_dir: Path, station: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = test_dir / f"frf_{station}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Measured FRF CSV not found: {path}")
    _, cols = _read_csv_dict(path)
    return cols["frequency_hz"], cols


def load_summary_rows(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    _, cols = _read_csv_dict(path)
    out: dict[int, dict[str, float]] = {}
    for i, t in enumerate(cols["tree"]):
        out[int(t)] = {name: float(cols[name][i]) for name in cols if name != "tree"}
    return out


def _register_pareto_pickle_classes() -> None:
    """Make ``pickle.load`` find verify_pareto_end_to_end's classes.

    The pareto cache pickles were written from inside that script, so unpickle
    needs the same class objects available in ``__main__``.
    """
    scripts_dir = REPO / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import verify_pareto_end_to_end as vp
    main = sys.modules["__main__"]
    for name in dir(vp):
        obj = getattr(vp, name)
        if isinstance(obj, type):
            main.__dict__[name] = obj


def load_sim_displacement_frf(cache_dir: Path, tree_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs_hz, H_x)`` from a pareto cache pickle.

    ``H_x`` is the displacement compliance |U_tip|/F in m·N⁻¹, obtained by
    dividing the stored displacement amplitudes by ``F_EXCITATION_N``.
    """
    path = cache_dir / f"tree_{tree_n}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Pareto cache missing: {path}. "
            "Run scripts/verify_pareto_end_to_end.py first."
        )
    with path.open("rb") as fh:
        record = pickle.load(fh)
    freqs = np.asarray(record.freqs, dtype=float)
    mags = np.asarray(record.mags, dtype=float) / F_EXCITATION_N
    return freqs, mags


def measured_displacement_frf(
    freqs_hz: np.ndarray, H_accel_mag: np.ndarray
) -> np.ndarray:
    """Convert accelerance |H_a| (m·s⁻²·N⁻¹) to compliance |H_x| (m·N⁻¹)."""
    omega = 2.0 * math.pi * freqs_hz
    out = np.full_like(H_accel_mag, np.nan, dtype=float)
    mask = omega > 0
    out[mask] = H_accel_mag[mask] / (omega[mask] ** 2)
    return out


def estimate_k3_c2(
    *,
    f_lin_hz: float,
    f_cubic_only_hz: float,
    f_full_hz: float,
    peak_cubic_only: float,
    peak_full: float,
    k1: float,
    m: float,
    A_resp_m: float,
    zeta_lin: float,
) -> dict[str, float]:
    """Back-estimate k_3, c_2 from the simulation's observed frequency shift
    and quadratic-damping peak suppression."""
    omega0 = 2.0 * math.pi * f_lin_hz
    omega_nl_k3 = 2.0 * math.pi * f_cubic_only_hz
    ratio_sq = (omega_nl_k3 / omega0) ** 2
    k3 = (4.0 / 3.0) * k1 * (ratio_sq - 1.0) / (A_resp_m ** 2)
    suppression_ratio = peak_cubic_only / peak_full if peak_full > 0 else float("inf")
    d_zeta = zeta_lin * (suppression_ratio - 1.0)
    omega_full = 2.0 * math.pi * f_full_hz
    c_eq = 2.0 * m * omega0 * d_zeta
    c2 = c_eq * (3.0 * math.pi / 8.0) / (omega_full * A_resp_m)
    return {
        "k3_N_per_m3": k3,
        "c2_N_s2_per_m2": c2,
        "delta_zeta": d_zeta,
        "suppression_ratio": suppression_ratio,
    }


def _configure_paper_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--measured-dir", default="results/hammer_test/tree1_p1",
        help="Directory containing frf_<station>.csv (default tree1_p1).",
    )
    parser.add_argument("--station", choices=("root", "mid", "tip"), default="tip")
    parser.add_argument("--component", choices=("X", "Y", "Z"), default="Z")
    parser.add_argument("--tree", type=int, default=1)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument(
        "--no-verification-copy", action="store_true",
        help="Skip writing the same figure to results_nonlinear/verification/.",
    )
    parser.add_argument("--k1", type=float, default=40000.0,
                        help="Linear support stiffness k_1 [N·m⁻¹] for the back-estimate.")
    parser.add_argument("--m", type=float, default=2.0,
                        help="Effective modal mass [kg].")
    parser.add_argument("--zeta-lin", type=float, default=0.05,
                        help="Linear damping ratio ζ_lin.")
    parser.add_argument("--A-resp", type=float, default=5.0e-3,
                        help="Modal response amplitude at resonance [m].")
    args = parser.parse_args()

    measured_dir = REPO / args.measured_dir
    freqs_meas, cols = load_measured_frf(measured_dir, args.station)
    H_accel = cols[f"H_{args.component}_mag_ms2_per_N"]
    coherence = cols[f"coherence_{args.component}"]
    H_x_meas = measured_displacement_frf(freqs_meas, H_accel)

    # Simulation FRF curves from pareto cache (instant, no FE re-run).
    _register_pareto_pickle_classes()
    f_sim_lin, Hx_sim_lin = load_sim_displacement_frf(PARETO_CACHE_LIN, args.tree)
    f_sim_nl, Hx_sim_nl = load_sim_displacement_frf(PARETO_CACHE_NL, args.tree)

    # Summary tables for the k3/c2 back-estimate (still useful for the printout).
    shift_rows = load_summary_rows(REPO / "results_nonlinear/verification/summary_frequency_shift.csv")
    c2_rows = load_summary_rows(REPO / "results_nonlinear/verification/summary_c2_suppression.csv")
    if args.tree not in shift_rows or args.tree not in c2_rows:
        print(f"[warn] tree {args.tree} not in summary CSVs — back-estimate skipped.")
        est = None
        f_lin = float(f_sim_lin[np.argmax(Hx_sim_lin)])
        f_full = float(f_sim_nl[np.argmax(Hx_sim_nl)])
        f_cubic = f_lin
        peak_lin_sim = float(np.max(Hx_sim_lin))
        peak_nl_sim = float(np.max(Hx_sim_nl))
    else:
        s = shift_rows[args.tree]
        c = c2_rows[args.tree]
        f_lin = s["f_lin_Hz"]
        f_full = c["f_full_Hz"]
        f_cubic = c["f_cubic_only_Hz"]
        peak_lin_sim = float(np.max(Hx_sim_lin))
        peak_nl_sim = float(np.max(Hx_sim_nl))
        est = estimate_k3_c2(
            f_lin_hz=f_lin, f_cubic_only_hz=f_cubic, f_full_hz=f_full,
            peak_cubic_only=c["peak_mag_cubic_only"], peak_full=c["peak_mag_full"],
            k1=args.k1, m=args.m, A_resp_m=args.A_resp, zeta_lin=args.zeta_lin,
        )

    print(f"\n=== Simulation summary, tree {args.tree} ===")
    print(f"  f_lin       = {f_lin:.2f} Hz   |H_x|_peak = {peak_lin_sim:.3e} m·N⁻¹")
    print(f"  f_full(k3,c2) = {f_full:.2f} Hz   |H_x|_peak = {peak_nl_sim:.3e} m·N⁻¹")
    print(f"  shift (lin → full)      : {(f_full-f_lin)/f_lin*100:+.2f} %")
    print(f"  peak ratio nl/lin       : {peak_nl_sim/peak_lin_sim:.3f}")

    if est is not None:
        print("\n=== k_3, c_2 back-estimate (assumes k_1, m, ζ_lin, A_resp) ===")
        print(f"  k_1 = {args.k1:.0f} N·m⁻¹, m = {args.m:.2f} kg, "
              f"ζ_lin = {args.zeta_lin:.2%}")
        print("  A_resp [mm] |   k_3 [N·m⁻³]   |  c_2 [N·s²·m⁻²]")
        print("  ------------|-----------------|----------------")
        for A in (0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3, 10.0e-3, 20.0e-3):
            e = estimate_k3_c2(
                f_lin_hz=f_lin, f_cubic_only_hz=f_cubic, f_full_hz=f_full,
                peak_cubic_only=c["peak_mag_cubic_only"], peak_full=c["peak_mag_full"],
                k1=args.k1, m=args.m, A_resp_m=A, zeta_lin=args.zeta_lin,
            )
            print(f"  {A*1000:>6.1f}      | {e['k3_N_per_m3']:+.2e}      | "
                  f"{e['c2_N_s2_per_m2']:+.2e}")
        print("\n  Reference (Liu et al., Table 3, unscaled):")
        print("    k_3 ∈ [-2.1e+09, -0.9e+09] N·m⁻³ (softening) or "
              "[+1.8e+09, +3.3e+09] (hardening)")
        print("    c_2 ∈ [+2.6e+04, +4.3e+04] N·s²·m⁻² (softening) or "
              "[+1.2e+04, +2.5e+04] (hardening)")
        print("\n  Implementation range (joints.py, internally rescaled):")
        print("    k_3 ∈ ±[0.9, 2.1]×10⁷ N·m⁻³,   c_2 ∈ [1.2, 4.3]×10³ N·s²·m⁻²")

    # ---- 3-curve overlay plot
    _configure_paper_style()
    import matplotlib.pyplot as plt

    PRIMARY = "#2166AC"   # linear sim
    ACCENT = "#B2182B"    # nonlinear sim
    GRID_MAJOR = "#d0d0d0"
    GRID_MINOR = "#ececec"
    UNIT_FRF_X = r"m$\cdot$N$^{-1}$"

    fig, (ax_h, ax_c) = plt.subplots(
        2, 1, figsize=(6.6, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    mask_meas = (freqs_meas > 0.5) & (freqs_meas <= args.fmax) & np.isfinite(H_x_meas)
    mask_sim_lin = (f_sim_lin > 0.0) & (f_sim_lin <= args.fmax)
    mask_sim_nl = (f_sim_nl > 0.0) & (f_sim_nl <= args.fmax)

    ax_h.semilogy(
        f_sim_lin[mask_sim_lin], Hx_sim_lin[mask_sim_lin],
        color=PRIMARY, linewidth=1.6, linestyle="-",
        label=r"Linear sim  ($k_3 = 0,\ c_2 = 0$)",
    )
    ax_h.semilogy(
        f_sim_nl[mask_sim_nl], Hx_sim_nl[mask_sim_nl],
        color=ACCENT, linewidth=1.6, linestyle="-",
        label=r"Nonlinear sim  ($k_3 \neq 0,\ c_2 \neq 0$)",
    )
    ax_h.semilogy(
        freqs_meas[mask_meas], H_x_meas[mask_meas],
        color="black", linewidth=1.3,
        marker="o", markersize=3.4,
        markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.5,
        linestyle="--",
        label=fr"Measured (tip–${args.component.lower()}$, $H_1$)",
    )
    ax_h.axvline(f_lin, color=PRIMARY, linewidth=0.9, linestyle=":", alpha=0.7)
    ax_h.axvline(f_full, color=ACCENT, linewidth=0.9, linestyle=":", alpha=0.7)
    ax_h.set_ylabel(rf"$|H_x| = |U_{{\rm tip}}/F|$ [{UNIT_FRF_X}]")
    ax_h.set_title(
        rf"tree {args.tree}: $f_{{\rm lin}}={f_lin:.2f}$ Hz $\rightarrow$ "
        rf"$f_{{\rm full}}={f_full:.2f}$ Hz "
        rf"($\Delta f/f_{{\rm lin}}={(f_full-f_lin)/f_lin*100:+.1f}\%$)"
    )
    ax_h.set_xlim(0.0, args.fmax)
    ax_h.grid(True, which="major", linewidth=0.6, color=GRID_MAJOR)
    ax_h.grid(True, which="minor", linewidth=0.4, color=GRID_MINOR)
    ax_h.legend(loc="upper right", fontsize=9)

    ax_c.plot(freqs_meas[mask_meas], coherence[mask_meas],
              color=ACCENT, linewidth=1.0)
    ax_c.set_ylim(0.0, 1.05)
    ax_c.set_ylabel(r"$\gamma^2$")
    ax_c.set_xlabel(r"Frequency $f$ [Hz]")
    ax_c.grid(True, which="major", linewidth=0.6, color=GRID_MAJOR)
    ax_c.grid(True, which="minor", linewidth=0.4, color=GRID_MINOR)

    fig.tight_layout(pad=0.4)

    stems = [measured_dir / f"compare_vs_sim_tree_{args.tree}"]
    if not args.no_verification_copy:
        verif_dir = REPO / "results_nonlinear" / "verification"
        verif_dir.mkdir(parents=True, exist_ok=True)
        stems.append(verif_dir / f"measured_vs_simulation_{measured_dir.name}")

    for stem in stems:
        fig.savefig(stem.with_suffix(".png"), dpi=150)
        fig.savefig(stem.with_suffix(".pdf"))
    import matplotlib.pyplot as _plt  # local re-import for the close call
    _plt.close(fig)
    for stem in stems:
        print(f"Wrote overlay: {stem}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
