"""Overlay a measured FRF curve against the linear / nonlinear simulation FRFs.

Three curves on a common axis:

  1. **Measured**  — hammer-impact test, tip–z accelerance |H_a| read from
     ``workspace/outputs/hammer_test/<test>/frf_tip.csv``; converted to displacement
     compliance |H_x| = |H_a| / (2π f)² so the unit lines up with the sim.
  2. **Linear sim (k_3 = 0, c_2 = 0)** — pareto-pipeline FRF curve cached at
     ``workspace/cache/figures_outputs/tree_<n>.pkl``.
     Magnitudes are displacement amplitudes at the model's excitation amplitude
     F (10 N for the supplied tree JSONs), so |H_x|_sim = |U_tip| / F.
  3. **Nonlinear sim (k_3 ≠ 0, c_2 ≠ 0)** — same conversion, cached at
     ``workspace/cache/figures_outputs/tree_<n>.pkl``.

The script ALSO back-estimates physical (k_3, c_2) ranges from the
simulation's observed cubic-only frequency shift and c_2 peak suppression
using the standard Duffing describing function — see ``estimate_k3_c2``.

Output (default): a two-panel figure (|H_x| overlay above, measured
coherence γ² below) written to BOTH
  - ``<measured-dir>/compare_vs_sim_tree_<n>.{png,pdf}`` (per-test record),
  - ``workspace/outputs/verification/measured_vs_simulation_tree_<n>.{png,pdf}``
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard_fem.workspace import workspace_paths

WORKSPACE = workspace_paths()

# Cache locations populated by generate_all_figures.py (whole-tree-mean FRF).
PARETO_CACHE_LIN = WORKSPACE.cache / "figures_outputs"
PARETO_CACHE_NL = WORKSPACE.cache / "figures_outputs"
# Cache populated by compute_validation_frf.py (single-branch single-point FRF).
VALIDATION_CACHE = WORKSPACE.cache / "validation_frf"

# Excitation amplitude used by both pipelines (= tree_*.json "amplitude").
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
    """Make ``pickle.load`` find generate_all_figures's classes.

    The pareto cache pickles were written from inside that script, so unpickle
    needs the same class objects available in ``__main__``.
    """
    scripts_dir = REPO / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import generate_all_figures as vp
    main = sys.modules["__main__"]
    for name in dir(vp):
        obj = getattr(vp, name)
        if isinstance(obj, type):
            main.__dict__[name] = obj


def load_sim_displacement_frf(cache_dir: Path, tree_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-tree-mean displacement compliance ``H_x = mean(|U_tip|)/F`` from
    the pareto cache pickle (units: m·N⁻¹).

    Note: this is the **averaged** FRF over all branch tips in ``ux``, NOT
    comparable to a single-accelerometer measurement. For validation use
    :func:`load_validation_frf` instead.
    """
    path = cache_dir / f"tree_{tree_n}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Pareto cache missing: {path}. "
            "Run scripts/generate_all_figures.py first."
        )
    with path.open("rb") as fh:
        record = pickle.load(fh)
    freqs = np.asarray(record.freqs, dtype=float)
    mags = np.asarray(record.mags, dtype=float) / F_EXCITATION_N
    return freqs, mags


def load_validation_frf(
    *, tree_n: int,
    input_branch: str, input_node: str, input_comp: str,
    output_branch: str, output_station: str, output_comp: str,
    prefer_calibrated: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Single-branch single-point displacement compliance from the validation
    cache produced by ``compute_validation_frf.py``.

    If a ``*_calibrated.npz`` companion file exists (produced by passing
    ``--from-calibration`` to compute_validation_frf.py), the **linear**
    curve is read from the uncalibrated cache and the **nonlinear** curve
    is read from the calibrated cache — giving "baseline linear vs
    calibrated nonlinear" on a single plot.

    Returns ``(freqs_hz, H_x_lin, H_x_nl, is_calibrated)`` with units m·N⁻¹.
    """
    stem = (f"tree_{tree_n}_{input_branch}_{input_node}_{input_comp}"
            f"_to_{output_branch}_{output_station}_{output_comp}")
    default_path = VALIDATION_CACHE / f"{stem}.npz"
    calib_path = VALIDATION_CACHE / f"{stem}_calibrated.npz"

    if not default_path.exists():
        raise FileNotFoundError(
            f"Validation FRF cache missing: {default_path}. Run "
            f"`python scripts/compute_validation_frf.py "
            f"--tree {tree_n} --input-branch {input_branch} "
            f"--input-node {input_node} --input-comp {input_comp} "
            f"--output-station {output_station} --output-comp {output_comp}` "
            f"first."
        )
    d_def = np.load(default_path)
    amp = float(d_def["amplitude"])
    freqs = np.asarray(d_def["freqs"], dtype=float)
    H_lin = np.asarray(d_def["mag_lin"], dtype=float) / amp

    if prefer_calibrated and calib_path.exists():
        d_cal = np.load(calib_path)
        H_nl = np.asarray(d_cal["mag_nl"], dtype=float) / float(d_cal["amplitude"])
        return freqs, H_lin, H_nl, True
    H_nl = np.asarray(d_def["mag_nl"], dtype=float) / amp
    return freqs, H_lin, H_nl, False


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
        "--measured-dir",
        default=str(WORKSPACE.outputs / "hammer_test" / "tree1_p1"),
        help="Directory containing frf_<station>.csv (default tree1_p1).",
    )
    parser.add_argument("--station", choices=("root", "mid", "tip"), default="tip")
    parser.add_argument("--component", choices=("X", "Y", "Z"), default="Z")
    parser.add_argument("--tree", type=int, default=1)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument(
        "--no-verification-copy", action="store_true",
        help="Skip writing the same figure to workspace/outputs/verification/.",
    )
    parser.add_argument("--k1", type=float, default=40000.0,
                        help="Linear support stiffness k_1 [N·m⁻¹] for the back-estimate.")
    parser.add_argument("--m", type=float, default=2.0,
                        help="Effective modal mass [kg].")
    parser.add_argument("--zeta-lin", type=float, default=0.05,
                        help="Linear damping ratio ζ_lin.")
    parser.add_argument("--A-resp", type=float, default=5.0e-3,
                        help="Modal response amplitude at resonance [m].")

    # --- single-branch validation FRF source (default mode, apples-to-apples)
    parser.add_argument(
        "--use-pareto-cache", action="store_true",
        help="Use whole-tree-mean FRF from the pareto cache (legacy, NOT "
        "physically comparable to a single accelerometer). Default off.",
    )
    parser.add_argument(
        "--input-branch", default="left_leader",
        help="Validation FRF: excitation branch_id (default left_leader, "
        "i.e. level-1 branch 1 for tree_1).",
    )
    parser.add_argument("--input-node", default="root",
                        choices=("root", "mid", "tip"))
    parser.add_argument("--input-comp", default="ux",
                        choices=("ux", "uy", "uz"),
                        help="Excitation direction in the global frame. The "
                        "field accelerometer's 'Z' on a vertical-leaning leader "
                        "branch lies in the lateral plane → default ux.")
    parser.add_argument("--output-branch", default=None,
                        help="Default: same as --input-branch (drive-point FRF).")
    parser.add_argument("--output-station", default="tip",
                        choices=("root", "mid", "tip"))
    parser.add_argument("--output-comp", default="ux",
                        choices=("ux", "uy", "uz"))
    args = parser.parse_args()

    measured_dir = REPO / args.measured_dir
    freqs_meas, cols = load_measured_frf(measured_dir, args.station)
    H_accel = cols[f"H_{args.component}_mag_ms2_per_N"]
    coherence = cols[f"coherence_{args.component}"]
    H_x_meas = measured_displacement_frf(freqs_meas, H_accel)

    # Simulation FRF — by default use the SINGLE-BRANCH cache (apples-to-apples
    # with one accelerometer) instead of the pareto whole-tree mean.
    is_calibrated = False
    if args.use_pareto_cache:
        _register_pareto_pickle_classes()
        f_sim_lin, Hx_sim_lin = load_sim_displacement_frf(PARETO_CACHE_LIN, args.tree)
        f_sim_nl, Hx_sim_nl = load_sim_displacement_frf(PARETO_CACHE_NL, args.tree)
    else:
        out_branch = args.output_branch or args.input_branch
        f_sim_lin, Hx_sim_lin, Hx_sim_nl, is_calibrated = load_validation_frf(
            tree_n=args.tree,
            input_branch=args.input_branch, input_node=args.input_node,
            input_comp=args.input_comp,
            output_branch=out_branch, output_station=args.output_station,
            output_comp=args.output_comp,
        )
        f_sim_nl = f_sim_lin
        if is_calibrated:
            print("[validation] using calibrated nonlinear curve "
                  "(_calibrated.npz)")

    # Summary tables for the k3/c2 back-estimate (still useful for the printout).
    shift_rows = load_summary_rows(WORKSPACE.outputs / "verification/summary_frequency_shift.csv")
    c2_rows = load_summary_rows(WORKSPACE.outputs / "verification/summary_c2_suppression.csv")
    # Peak frequencies are taken from the loaded curves directly so they match
    # whatever FRF source (pareto whole-tree mean vs single-branch validation)
    # the user picked. The k3/c2 back-estimate optionally pulls cubic-only
    # peak data from the diagnostic summary CSV when available.
    f_lin = float(f_sim_lin[np.argmax(Hx_sim_lin)])
    f_full = float(f_sim_nl[np.argmax(Hx_sim_nl)])
    peak_lin_sim = float(np.max(Hx_sim_lin))
    peak_nl_sim = float(np.max(Hx_sim_nl))

    if args.tree in shift_rows and args.tree in c2_rows:
        c = c2_rows[args.tree]
        f_cubic = c["f_cubic_only_Hz"]
        est = estimate_k3_c2(
            f_lin_hz=f_lin, f_cubic_only_hz=f_cubic, f_full_hz=f_full,
            peak_cubic_only=c["peak_mag_cubic_only"], peak_full=c["peak_mag_full"],
            k1=args.k1, m=args.m, A_resp_m=args.A_resp, zeta_lin=args.zeta_lin,
        )
    else:
        est = None
        f_cubic = f_lin

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
    nl_label = (r"Nonlinear sim (calibrated $\beta, k_3, c_2$)"
                if not args.use_pareto_cache and is_calibrated
                else r"Nonlinear sim  ($k_3 \neq 0,\ c_2 \neq 0$)")
    ax_h.semilogy(
        f_sim_nl[mask_sim_nl], Hx_sim_nl[mask_sim_nl],
        color=ACCENT, linewidth=1.6, linestyle="-",
        label=nl_label,
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
        rf"tree {args.tree}: $f_{{\rm lin}}={f_lin:.2f}\,$Hz $\rightarrow$ "
        rf"$f_{{\rm full}}={f_full:.2f}\,$Hz "
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
        verif_dir = WORKSPACE.outputs / "verification"
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
