"""Sobol-based uncertainty source tracing on T3's recommended startup frequency.

Implements Section 4.7 / Fig 15 / Table 6 of the manuscript: a 12-dimensional
Saltelli sample is propagated through the calibrated FE forward operator,
the dominant in-band resonance frequency f_r ∈ [3, 20] Hz is treated as
the scalar output Z, and SALib computes the first-order (S1) and total
(ST) Sobol indices.

The 12 inputs (manuscript Eq. 8 + the four trunk/branch geometric scales):
    1.  E_factor               multiplies xylem Young's modulus (dominant)
    2.  rho_factor             multiplies xylem density (dominant)
    3.  zeta_1                 first-mode damping ratio
    4.  zeta_2                 second-mode damping ratio
    5.  log10_kc               trunk clamp support stiffness, log10(N/m)
    6.  log10_cc               trunk clamp support damping,   log10(N·s/m)
    7.  log10_kf               fruit-attachment stiffness proxy via pith E
    8.  log10_cf               fruit-attachment damping ratio
    9.  d_tr_root_factor       multiplies trunk root outer radius
   10.  L_tr_factor            multiplies trunk length
   11.  d_br_factor            multiplies every non-trunk branch radius
   12.  L_br_factor            multiplies every non-trunk branch length

Forward model: one FRF sweep (0.5–25 Hz, 30 freq points) per sample,
extract the dominant peak in [3, 20] Hz. Each evaluation ≈ 1.5–2 s.

Saltelli sample size N produces N·(2D+2) = 26 N evaluations (with second-
order on). The manuscript uses N = 512 → 13312 evals ≈ 7 h. Default here
is N = 64 (1664 evals ≈ 50 min) so the script can be validated before
committing to the full run; pass ``--n-base 512`` for the publication-grade
version.

Outputs:
  workspace/outputs/calibration/sobol_t3_samples.npz   raw samples + outputs (resumable)
  workspace/outputs/calibration/sobol_t3_indices.csv   S1, ST per parameter
  workspace/outputs/calibration/fig15a_sobol_t3.{png,pdf}   bar plot of S1 vs ST
  workspace/outputs/calibration/sobol_t3_summary.txt   human-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard_fem.workspace import display_path, example_trees_dir, workspace_paths


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter problem definition
# ─────────────────────────────────────────────────────────────────────────────
PARAM_NAMES = [
    "E_factor",        # xylem Young's modulus multiplier
    "rho_factor",      # xylem density multiplier
    "zeta_1",          # first mode damping ratio
    "zeta_2",          # second mode damping ratio
    "log10_kc",        # clamp support stiffness [N/m], log10
    "log10_cc",        # clamp support damping [N·s/m], log10
    "log10_kf",        # fruit-attach stiffness (pith E proxy), log10 [Pa]
    "log10_cf",        # fruit-attach damping ratio, log10
    "d_tr_root_factor",
    "L_tr_factor",
    "d_br_factor",
    "L_br_factor",
]
PARAM_BOUNDS = {
    "E_factor":         [0.7, 1.4],
    "rho_factor":       [0.85, 1.15],
    "zeta_1":           [0.02, 0.20],
    "zeta_2":           [0.02, 0.20],
    "log10_kc":         [4.0, 6.0],
    "log10_cc":         [1.0, 3.0],
    "log10_kf":         [8.0, 10.0],   # pith E ∈ [10⁸, 10¹⁰] Pa
    "log10_cf":         [-2.0, 0.0],   # damping ratio ∈ [10⁻², 1]
    "d_tr_root_factor": [0.7, 1.3],
    "L_tr_factor":      [0.85, 1.15],
    "d_br_factor":      [0.7, 1.3],
    "L_br_factor":      [0.85, 1.15],
}
PARAM_DESCRIPTIONS = {  # human-friendly names for plots / paper
    "E_factor": r"$E$",
    "rho_factor": r"$\rho$",
    "zeta_1": r"$\zeta_1$",
    "zeta_2": r"$\zeta_2$",
    "log10_kc": r"$k_c$",
    "log10_cc": r"$c_c$",
    "log10_kf": r"$k_f$",
    "log10_cf": r"$c_f$",
    "d_tr_root_factor": r"$d_{\rm tr,root}$",
    "L_tr_factor": r"$L_{\rm tr}$",
    "d_br_factor": r"$\bar{d}_{\rm br}$",
    "L_br_factor": r"$\bar{L}_{\rm br}$",
}


def build_problem():
    return {
        "num_vars": len(PARAM_NAMES),
        "names": PARAM_NAMES,
        "bounds": [PARAM_BOUNDS[n] for n in PARAM_NAMES],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Model mutators
# ─────────────────────────────────────────────────────────────────────────────
def apply_sample(model_json: dict, params: dict) -> dict:
    """Return a new model dict with the parameter sample applied in-place."""
    m = json.loads(json.dumps(model_json))  # deep copy

    # 1-2. xylem E and ρ scaling
    for mat in m["materials"]:
        if mat["id"] == "xylem_default":
            mat["youngs_modulus"] *= params["E_factor"]
            mat["density"] *= params["rho_factor"]

    # 7. pith E proxy for fruit-attachment stiffness
    for mat in m["materials"]:
        if mat["id"] == "pith_default":
            mat["youngs_modulus"] = 10.0 ** params["log10_kf"]

    # 3-4. Rayleigh α, β from (ζ_1, ζ_2). Use modes at f1=10 Hz, f2=20 Hz
    # as fiducial frequencies; the relationship is approximate but adequate
    # for a sensitivity study.
    f1, f2 = 10.0, 20.0
    w1, w2 = 2.0 * math.pi * f1, 2.0 * math.pi * f2
    z1, z2 = params["zeta_1"], params["zeta_2"]
    denom = (w2 ** 2 - w1 ** 2)
    if abs(denom) < 1e-9:
        denom = 1e-9
    alpha = 2.0 * (z1 * w1 * w2 ** 2 - z2 * w2 * w1 ** 2) / denom
    beta = 2.0 * (z2 * w2 - z1 * w1) / denom
    m["analysis"]["rayleigh_alpha"] = float(alpha)
    m["analysis"]["rayleigh_beta"] = float(beta)

    # 5-6. clamp stiffness, damping
    if m["clamps"]:
        m["clamps"][0]["support_stiffness"] = 10.0 ** params["log10_kc"]
        m["clamps"][0]["support_damping"] = 10.0 ** params["log10_cc"]

    # 8. fruit attachment damping ratio
    m["fruit_distribution_policy"]["attachment_damping_ratio"] = (
        10.0 ** params["log10_cf"]
    )

    # 9-10. trunk geometry — scale root radius and length
    trunk = next(b for b in m["branches"] if b["id"] == "trunk")
    if trunk["stations"]:
        trunk["stations"][0]["outer_radius"] *= params["d_tr_root_factor"]
    # scale trunk length by scaling z-coordinates of points and end
    L = params["L_tr_factor"]
    trunk["end"] = [trunk["end"][0], trunk["end"][1], trunk["end"][2] * L]
    trunk["points"] = [
        [p[0], p[1], p[2] * L] for p in trunk["points"]
    ]

    # 11-12. average branch geometry — scale every non-trunk branch
    d_br = params["d_br_factor"]
    L_br = params["L_br_factor"]
    for b in m["branches"]:
        if b["id"] == "trunk":
            continue
        # radii at each station
        for st in b.get("stations", []):
            if "outer_radius" in st:
                st["outer_radius"] *= d_br
        # length: scale (end - start) by L_br
        s, e = b["start"], b["end"]
        new_end = [s[i] + (e[i] - s[i]) * L_br for i in range(3)]
        b["end"] = new_end
        new_pts = []
        for p in b.get("points", []):
            new_pts.append([s[i] + (p[i] - s[i]) * L_br for i in range(3)])
        b["points"] = new_pts

    return m


# ─────────────────────────────────────────────────────────────────────────────
#  Forward evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_resonance(model_json: dict, *, freq_band=(0.5, 25.0), n_freq=30,
                       in_band=(3.0, 20.0)):
    """Load a model from a dict, run FRF sweep, return dominant peak in
    [in_band] Hz on the whole-tree-mean canopy displacement.

    Returns NaN if FE fails or no peak found.
    """
    # We load via a temp JSON file because load_orchard_model takes a path
    import tempfile
    from orchard_fem.io.loaders.orchard import load_orchard_model
    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(model_json, fh)
        tmp_path = fh.name
    try:
        model = load_orchard_model(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    swept = replace(
        model,
        analysis=replace(
            model.analysis,
            frequency_start_hz=float(freq_band[0]),
            frequency_end_hz=float(freq_band[1]),
            frequency_steps=int(n_freq),
        ),
    )
    try:
        exp = solve_embedded_beam_frequency_response_experiment(
            swept, polynomial_degree=1,
        )
    except Exception:
        return float("nan")
    res = exp.result
    freqs = np.array([p.frequency_hz for p in res.points])
    name_to_idx = {n: i for i, n in enumerate(res.observation_names)}
    tip_obs = [n for n in res.observation_names if n.endswith("_tip_ux")]
    if not tip_obs:
        return float("nan")
    mags = np.array([
        float(np.mean([p.observation_magnitudes[name_to_idx[n]] for n in tip_obs]))
        for p in res.points
    ])
    mask = (freqs >= in_band[0]) & (freqs <= in_band[1])
    if not mask.any() or not np.isfinite(mags[mask]).any():
        return float("nan")
    idx_local = int(np.nanargmax(mags[mask]))
    f_r = float(freqs[mask][idx_local])
    return f_r


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=int, default=3)
    parser.add_argument(
        "--n-base", type=int, default=64,
        help="Saltelli base sample size N (default 64 → 1664 evals ≈ 50 min). "
        "Use --n-base 512 for the manuscript's 13312 evals (~7 h).",
    )
    parser.add_argument(
        "--no-second-order", action="store_true",
        help="Skip second-order Sobol indices (cuts evals from N(2D+2) "
        "to N(D+2): with D=12, factor ~1.86 speedup).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-dir", default=str(workspace_paths().outputs / "calibration"),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing samples.npz (continue forward evaluations).",
    )
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"sobol_t{args.tree}_samples.npz"
    indices_path = out_dir / f"sobol_t{args.tree}_indices.csv"
    summary_path = out_dir / f"sobol_t{args.tree}_summary.txt"
    fig_stem = out_dir / f"fig15a_sobol_t{args.tree}"

    # Load baseline model JSON (we deep-copy + mutate it per sample)
    base_json_path = example_trees_dir() / f"tree_{args.tree}.json"
    with base_json_path.open() as fh:
        base_json = json.load(fh)
    print(f"Loaded baseline {base_json_path.name}")

    problem = build_problem()
    calc_second = not args.no_second_order

    # Sample
    print(f"\nSaltelli sampling: N={args.n_base}, D={problem['num_vars']}, "
          f"second_order={calc_second}")
    from SALib.sample import sobol as sobol_sample
    X = sobol_sample.sample(
        problem, args.n_base,
        calc_second_order=calc_second, seed=args.seed,
    )
    print(f"  {X.shape[0]} samples generated")

    # Forward evaluations
    if args.resume and samples_path.exists():
        d = np.load(samples_path)
        Y = d["Y"].copy()
        if Y.shape[0] != X.shape[0]:
            print(f"[resume] mismatch ({Y.size} vs {X.shape[0]}); restarting")
            Y = np.full(X.shape[0], np.nan)
        else:
            done = int(np.isfinite(Y).sum())
            print(f"[resume] {done}/{Y.size} samples already evaluated")
    else:
        Y = np.full(X.shape[0], np.nan)

    print(f"\nForward sweep: {(~np.isfinite(Y)).sum()} remaining evaluations …")
    t_total = time.time()
    for i in range(X.shape[0]):
        if np.isfinite(Y[i]):
            continue
        params = dict(zip(PARAM_NAMES, X[i]))
        m = apply_sample(base_json, params)
        t0 = time.time()
        Y[i] = evaluate_resonance(m)
        dt = time.time() - t0
        if (i + 1) % 20 == 0 or i < 3:
            n_done = int(np.isfinite(Y).sum())
            elapsed = time.time() - t_total
            eta = elapsed / max(n_done, 1) * (X.shape[0] - n_done)
            print(f"  [{i+1:4d}/{X.shape[0]}] f_r={Y[i]:.2f} Hz  "
                  f"({dt:.1f}s, ETA {eta/60:.1f} min)")
            # Save periodically so we can resume on interruption
            np.savez(samples_path, X=X, Y=Y, names=np.array(PARAM_NAMES))

    np.savez(samples_path, X=X, Y=Y, names=np.array(PARAM_NAMES))
    print(f"\nWrote: {display_path(samples_path)}")

    valid = np.isfinite(Y)
    print(f"Valid evaluations: {valid.sum()}/{Y.size} "
          f"({(~valid).sum()} failed)")
    if (~valid).any():
        # Replace NaNs with median for SALib (it can't handle NaN)
        Y_clean = np.where(valid, Y, np.nanmedian(Y))
    else:
        Y_clean = Y

    # Sobol analysis
    print("\nComputing Sobol indices …")
    from SALib.analyze import sobol as sobol_analyze
    Si = sobol_analyze.analyze(
        problem, Y_clean,
        calc_second_order=calc_second, seed=args.seed,
        print_to_console=False,
    )

    # Write indices CSV
    with indices_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "S1", "S1_conf", "ST", "ST_conf"])
        for j, name in enumerate(PARAM_NAMES):
            w.writerow([
                name,
                f"{Si['S1'][j]:.4f}", f"{Si['S1_conf'][j]:.4f}",
                f"{Si['ST'][j]:.4f}", f"{Si['ST_conf'][j]:.4f}",
            ])
    print(f"Wrote: {display_path(indices_path)}")

    # Summary text
    order = np.argsort(-Si["ST"])
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Sobol indices — tree {args.tree}\n")
        fh.write(f"N_base = {args.n_base}, D = {problem['num_vars']}, "
                 f"total evals = {X.shape[0]}\n")
        fh.write(f"Valid = {valid.sum()}/{Y.size}\n")
        fh.write("Output: dominant in-band resonance frequency f_r ∈ [3, 20] Hz\n")
        fh.write(f"  Y stats: mean={np.nanmean(Y):.2f} Hz, "
                 f"std={np.nanstd(Y):.2f} Hz, "
                 f"range=[{np.nanmin(Y):.1f}, {np.nanmax(Y):.1f}]\n\n")
        fh.write("Ranked by total effect S_T:\n")
        for rank, j in enumerate(order, 1):
            fh.write(f"  {rank:2d}. {PARAM_NAMES[j]:18s}  "
                     f"S1={Si['S1'][j]:+.3f} ± {Si['S1_conf'][j]:.3f}  "
                     f"ST={Si['ST'][j]:+.3f} ± {Si['ST_conf'][j]:.3f}\n")
    print(f"Wrote: {display_path(summary_path)}")

    # Bar plot (Fig 15a)
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    width = 0.38
    pos = np.arange(len(PARAM_NAMES))
    s1 = Si["S1"][order]
    st = Si["ST"][order]
    s1c = Si["S1_conf"][order]
    stc = Si["ST_conf"][order]
    labels = [PARAM_DESCRIPTIONS[PARAM_NAMES[j]] for j in order]
    ax.bar(pos - width/2, s1, width, color="#7FA7C9",
           yerr=s1c, error_kw={"lw": 0.8, "capsize": 2},
           edgecolor="#333", linewidth=0.6, label=r"$S_1$ (first-order)")
    ax.bar(pos + width/2, st, width, color="#2166AC",
           yerr=stc, error_kw={"lw": 0.8, "capsize": 2},
           edgecolor="#333", linewidth=0.6, label=r"$S_T$ (total)")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"Sobol index for $f_r$")
    ax.set_title(rf"Sobol sensitivity — tree {args.tree}  "
                 rf"(N = {args.n_base}, {X.shape[0]} evals)")
    ax.axhline(0, color="black", lw=0.7)
    ax.grid(True, axis="y", which="major", linewidth=0.6, color="#d0d0d0")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(fig_stem.with_suffix(".png"), dpi=150)
    fig.savefig(fig_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Fig 15a: {display_path(fig_stem)}.{{png,pdf}}")

    # Console summary
    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed/60:.1f} min")
    print("\nTop 5 total-effect indices:")
    for rank, j in enumerate(order[:5], 1):
        print(f"  {rank}. {PARAM_NAMES[j]:18s}  ST = {Si['ST'][j]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
