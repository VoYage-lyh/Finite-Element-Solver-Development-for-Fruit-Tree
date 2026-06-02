"""Render Fig 14 + Table 5 — fixed-frequency posterior-predictive validation.

Reads:
  - ``results/calibration/fixed_freq_segments.csv``     measured hold segments
  - ``cache/calibration/tree_<N>_..._fit.npz``          calibrated (β, k3, c2)
  - ``trees/tree_<N>.json``                             FE model

Pipeline:
  1. For each measured segment (drive_hz, rms_steady), forward-evaluate the
     **calibrated** nonlinear FE at the segment's drive frequency, on the
     single-branch single-point observation that the field accelerometer
     covers. This gives a per-frequency displacement-compliance |H_x|(f).
  2. Convert to a predicted acceleration RMS:
        a_pred,RMS(f) = (2π f)² · α · |H_x|(f) / √2
     where α is a single scalar absorbing the unknown shaker force
     amplitude. α is fit by least-squares in log space across all segments.
  3. Posterior predictive band: draw ``--n-post`` perturbed parameter sets
     around the calibrated point (β, k3, c2 each ± ``--post-sigma`` % in log
     space), forward-evaluate, retain 5%–95% percentile envelope.
  4. Plot Fig 14 (RMS vs drive frequency) + per-segment relative-error bar
     panel. Write Table 5 (segment_csv) and Table 5-summary (csv).

This avoids the "exact 6/10/14/18/22 Hz" assumption — segments are taken
as found in the data. Branches not instrumented stay out of the comparison;
the figure thus reflects the user's measurement scope, not the whole tree.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard_fem.io.loaders.orchard import load_orchard_model


def _override_excitation(model, *, branch_id, node, comp, amplitude):
    new_excitation = replace(
        model.excitation,
        target_branch_id=branch_id, target_node=node,
        target_component=comp, amplitude=amplitude, target_s=None,
    )
    return replace(model, excitation=new_excitation)


def forward_disp_compliance(model, freqs_hz, output_obs):
    """Run a single FE sweep at the supplied frequencies; return |U/F| [m/N]."""
    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    swept = replace(
        model,
        analysis=replace(
            model.analysis,
            frequency_start_hz=float(freqs_hz[0]),
            frequency_end_hz=float(freqs_hz[-1]),
            frequency_steps=int(freqs_hz.size),
        ),
    )
    exp = solve_embedded_beam_frequency_response_experiment(swept, polynomial_degree=1)
    res = exp.result
    if output_obs not in res.observation_names:
        raise RuntimeError(f"Observation {output_obs!r} not in model.")
    idx = res.observation_names.index(output_obs)
    f_out = np.array([p.frequency_hz for p in res.points])
    mag = np.array([p.observation_magnitudes[idx] for p in res.points])
    amp_in = float(model.excitation.amplitude)
    return f_out, mag / amp_in  # m/N


def predicted_accel_rms(freqs_hz, Hx_mpn, alpha):
    """a_pred,RMS(f) = (2π f)² · α · |H_x|(f) / √2 for a sinusoidal input
    of force amplitude α."""
    omega = 2.0 * math.pi * freqs_hz
    return alpha * omega ** 2 * Hx_mpn / math.sqrt(2.0)


def fit_alpha(freqs, Hx, rms_meas):
    """Least-squares scale factor in log space, then convert back."""
    pred_unit = (2.0 * math.pi * freqs) ** 2 * Hx / math.sqrt(2.0)
    valid = (pred_unit > 0) & np.isfinite(rms_meas) & (rms_meas > 0)
    log_ratio = np.log(rms_meas[valid]) - np.log(pred_unit[valid])
    return float(np.exp(np.mean(log_ratio)))


def patched_nonlinear(beta, k3_scale, c2_scale):
    """Context-manager-like helper to patch joints downscales."""
    import orchard_fem.fenicsx.joints as j
    orig_k3 = j._K3_DOWNSCALE
    orig_c2 = j._C2_DOWNSCALE
    j._K3_DOWNSCALE = orig_k3 * float(k3_scale)
    j._C2_DOWNSCALE = orig_c2 * float(c2_scale)
    return j, orig_k3, orig_c2


def _configure_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=int, default=1)
    parser.add_argument(
        "--segments-csv", default="results/calibration/fixed_freq_segments.csv",
    )
    parser.add_argument(
        "--calibration-cache",
        default="cache/calibration/tree_1_left_leader_root_ux_fit.npz",
    )
    parser.add_argument("--input-branch", default="left_leader")
    parser.add_argument("--input-node", default="root",
                        choices=("root", "mid", "tip"))
    parser.add_argument("--input-comp", default="ux",
                        choices=("ux", "uy", "uz"))
    parser.add_argument("--output-branch", default=None)
    parser.add_argument("--output-station", default="tip",
                        choices=("root", "mid", "tip"))
    parser.add_argument("--output-comp", default="ux",
                        choices=("ux", "uy", "uz"))
    parser.add_argument(
        "--init-amplitude", type=float, default=200.0,
        help="Initial shaker force amplitude in N for the sim (default 200, a "
        "typical clamped-shaker continuous-output magnitude). The script then "
        "self-iterates: at each round it re-fits the observed amplitude α and "
        "re-runs the sim at F = α so |H_x| reflects the *actual* drive level "
        "(matters for the nonlinear k₃Δu³ + c₂|v|v terms, which only kick in "
        "at realistic amplitudes — F = 10 N is essentially the linear limit).",
    )
    parser.add_argument(
        "--alpha-iter", type=int, default=3,
        help="Self-consistent α↔F iterations (default 3).",
    )
    parser.add_argument(
        "--alpha-tol-pct", type=float, default=5.0,
        help="Stop α iteration when |Δα/α| ≤ this percentage (default 5%).",
    )
    parser.add_argument("--n-post", type=int, default=40,
                        help="Posterior predictive samples (default 40).")
    parser.add_argument("--post-sigma-pct", type=float, default=25.0,
                        help="Log-space ±σ for β / k3_scale / c2_scale (default 25%).")
    parser.add_argument("--fmin-segment", type=float, default=5.0,
                        help="Drop measured segments with drive freq below this "
                        "value (default 5 Hz) — typically setup transients or "
                        "operator handling noise rather than true steady drive.")
    parser.add_argument("--fmax-segment", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--name", default=None,
        help="Output filename suffix (e.g. --name sensor1 → "
        "fig14_sensor1.{png,pdf} + table5_sensor1.csv). Defaults to the "
        "segments-CSV stem.",
    )
    args = parser.parse_args()

    out_dir = REPO / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve output suffix from --name or segments-CSV stem
    if args.name:
        suffix = args.name
    else:
        # e.g. fixed_freq_segments_sensor1.csv → "sensor1"
        stem = Path(args.segments_csv).stem
        suffix = stem.replace("fixed_freq_segments_", "") if stem.startswith(
            "fixed_freq_segments_") else stem

    # ------------------------------ load segments
    seg_path = REPO / args.segments_csv
    with seg_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    seg_drive = np.array([float(r["drive_hz"]) for r in rows])
    seg_rms = np.array([float(r["rms_steady"]) for r in rows])
    order = np.argsort(seg_drive)
    seg_drive = seg_drive[order]
    seg_rms = seg_rms[order]
    print(f"Loaded {seg_drive.size} measured segments, "
          f"freq range {seg_drive.min():.1f}–{seg_drive.max():.1f} Hz")
    keep = (seg_drive >= args.fmin_segment) & (seg_drive <= args.fmax_segment)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  dropped {n_dropped} segment(s) outside "
              f"[{args.fmin_segment:.1f}, {args.fmax_segment:.1f}] Hz "
              f"(setup transients / out-of-band)")
    seg_drive = seg_drive[keep]
    seg_rms = seg_rms[keep]

    # ------------------------------ load calibration
    cal = np.load(REPO / args.calibration_cache)
    beta_opt = float(cal["beta_opt"])
    k3_scale_opt = float(cal["k3_scale_opt"])
    c2_scale_opt = float(cal["c2_scale_opt"])
    print(f"Calibrated: β={beta_opt:.3e}, k3×{k3_scale_opt:.2f}, c2×{c2_scale_opt:.2f}")

    # ------------------------------ load + override model
    base_model_raw = load_orchard_model(str(REPO / "trees" / f"tree_{args.tree}.json"))
    output_branch = args.output_branch or args.input_branch
    output_obs = f"obs_{output_branch}_{args.output_station}_{args.output_comp}"
    print(f"Forward target: excitation @ {args.input_branch}/{args.input_node}/{args.input_comp},"
          f" observation @ {output_obs}")

    # Use measured-segment frequencies (sorted, deduplicated within 0.1 Hz)
    # as the eval grid.
    eval_freqs = np.unique(np.round(seg_drive, 1))
    print(f"Evaluating sim at {eval_freqs.size} unique freq points "
          f"({eval_freqs.min():.1f}–{eval_freqs.max():.1f} Hz)")

    # ------------------------------ self-consistent α ↔ F iteration
    # |H_x|(F) depends on F via the k3 / c2 nonlinear terms, so matching the
    # measured RMS must be done at the *actual* drive amplitude. Loop: run
    # sim at current F → fit α from data → set F = α, re-run, until α
    # converges (or max-iter reached).
    F_current = float(args.init_amplitude)
    Hx_median = None
    alpha = None
    print(f"\nSelf-consistent α ↔ F iteration "
          f"(tol = ±{args.alpha_tol_pct:.1f}%, max {args.alpha_iter} rounds):")
    j_mod, orig_k3, orig_c2 = patched_nonlinear(
        beta_opt, k3_scale_opt, c2_scale_opt,
    )
    try:
        for it in range(1, args.alpha_iter + 1):
            mdl = _override_excitation(
                base_model_raw,
                branch_id=args.input_branch, node=args.input_node,
                comp=args.input_comp, amplitude=F_current,
            )
            mdl = replace(
                mdl, analysis=replace(mdl.analysis, rayleigh_beta=beta_opt),
            )
            t0 = time.time()
            f_sim, Hx_now = forward_disp_compliance(mdl, eval_freqs, output_obs)
            dt = time.time() - t0
            Hx_at_seg_iter = np.interp(seg_drive, f_sim, Hx_now)
            alpha = fit_alpha(seg_drive, Hx_at_seg_iter, seg_rms)
            delta_pct = abs(alpha - F_current) / F_current * 100.0
            print(f"  iter {it}: F = {F_current:6.1f} N → α = {alpha:6.1f} N  "
                  f"(Δ = {delta_pct:5.1f}%, sweep {dt:.1f}s)")
            Hx_median = Hx_now
            if delta_pct <= args.alpha_tol_pct:
                print("  → converged.")
                break
            F_current = alpha
    finally:
        j_mod._K3_DOWNSCALE = orig_k3
        j_mod._C2_DOWNSCALE = orig_c2

    F_drive = float(alpha)
    Hx_median_at_seg = np.interp(seg_drive, eval_freqs, Hx_median)
    pred_median = predicted_accel_rms(seg_drive, Hx_median_at_seg, F_drive)
    print(f"\nConverged shaker force amplitude F = {F_drive:.1f} N\n"
          f"(real-world clamped shakers typically deliver 100–500 N continuous.)")

    # ------------------------------ posterior predictive samples
    rng = np.random.default_rng(args.seed)
    log_sigma = math.log(1.0 + args.post_sigma_pct / 100.0)
    samples_pred = np.empty((args.n_post, seg_drive.size))
    print(f"\nPosterior predictive: {args.n_post} samples at F = {F_drive:.1f} N "
          f"(log σ = ±{args.post_sigma_pct:.0f}% on β, k3, c2)…")
    for i in range(args.n_post):
        beta_i = beta_opt * math.exp(rng.normal(0.0, log_sigma))
        k3_i = k3_scale_opt * math.exp(rng.normal(0.0, log_sigma))
        c2_i = c2_scale_opt * math.exp(rng.normal(0.0, log_sigma))
        mi = _override_excitation(
            base_model_raw,
            branch_id=args.input_branch, node=args.input_node,
            comp=args.input_comp, amplitude=F_drive,
        )
        mi = replace(mi, analysis=replace(mi.analysis, rayleigh_beta=beta_i))
        j_mod._K3_DOWNSCALE = orig_k3 * k3_i
        j_mod._C2_DOWNSCALE = orig_c2 * c2_i
        try:
            _, Hx_i = forward_disp_compliance(mi, eval_freqs, output_obs)
        finally:
            j_mod._K3_DOWNSCALE = orig_k3
            j_mod._C2_DOWNSCALE = orig_c2
        Hx_at_seg = np.interp(seg_drive, eval_freqs, Hx_i)
        samples_pred[i] = predicted_accel_rms(seg_drive, Hx_at_seg, F_drive)
        if (i + 1) % 10 == 0:
            print(f"  …{i+1}/{args.n_post}")

    pred_q05 = np.percentile(samples_pred, 5, axis=0)
    pred_q95 = np.percentile(samples_pred, 95, axis=0)

    # Coverage check
    in_band = (seg_rms >= pred_q05) & (seg_rms <= pred_q95)
    coverage = float(in_band.mean())
    print(f"\nCoverage: {in_band.sum()}/{seg_drive.size} measured RMS "
          f"inside 90% predictive band ({coverage*100:.0f}%)")

    rel_err = (seg_rms - pred_median) / pred_median * 100.0
    print(f"Relative error: mean abs {np.mean(np.abs(rel_err)):.1f}%, "
          f"max {np.max(np.abs(rel_err)):.1f}%")

    # ------------------------------ write Table 5
    tbl_path = out_dir / f"table5_fixed_freq_{suffix}.csv"
    with tbl_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drive_hz", "measured_rms", "posterior_median",
                    "q05_ms2", "q95_ms2", "rel_err_pct", "covered"])
        for i in range(seg_drive.size):
            w.writerow([
                f"{seg_drive[i]:.2f}", f"{seg_rms[i]:.3f}",
                f"{pred_median[i]:.3f}",
                f"{pred_q05[i]:.3f}", f"{pred_q95[i]:.3f}",
                f"{rel_err[i]:+.1f}", "yes" if in_band[i] else "no",
            ])
    print(f"\nTable 5: {tbl_path.relative_to(REPO)}")

    # ------------------------------ Fig 14
    _configure_paper_style()
    import matplotlib.pyplot as plt
    PRIMARY = "#2166AC"
    ACCENT = "#B2182B"

    fig, (ax_h, ax_e) = plt.subplots(
        2, 1, figsize=(6.8, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_h.fill_between(
        seg_drive, pred_q05, pred_q95,
        color=ACCENT, alpha=0.18, linewidth=0,
        label=r"90% posterior predictive band",
    )
    ax_h.plot(
        seg_drive, pred_median,
        color=ACCENT, linewidth=1.6, marker="s", markersize=4.4,
        markerfacecolor="white", markeredgecolor=ACCENT, markeredgewidth=1.2,
        label="posterior median",
    )
    ax_h.errorbar(
        seg_drive, seg_rms,
        yerr=0.05 * seg_rms,  # 5% measurement uncertainty bar
        fmt="o", color="black", markersize=5.0, capsize=2.5,
        markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.6,
        label="measured RMS",
    )
    ax_h.set_ylabel(r"steady-state RMS [m$\cdot$s$^{-2}$]")
    ax_h.set_title(
        rf"Fixed-frequency validation — tree {args.tree}, "
        rf"{output_branch}/{args.output_station}/{args.output_comp},  "
        rf"$F = {F_drive:.0f}$ N;  coverage = {coverage*100:.0f}%"
    )
    ax_h.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax_h.grid(True, which="minor", linewidth=0.4, color="#ececec")
    ax_h.legend(loc="upper right", fontsize=9)

    bar_colors = [PRIMARY if abs(e) <= 15 else ACCENT for e in rel_err]
    ax_e.bar(seg_drive, rel_err, color=bar_colors, edgecolor="#333", width=0.35)
    ax_e.axhline(0, color="black", lw=0.6)
    ax_e.axhline(15, color="#999", lw=0.8, linestyle="--")
    ax_e.axhline(-15, color="#999", lw=0.8, linestyle="--")
    ax_e.set_ylabel("rel. error [%]")
    ax_e.set_xlabel(r"drive frequency $f$ [Hz]")
    ax_e.grid(True, which="major", linewidth=0.6, color="#d0d0d0")

    fig.tight_layout(pad=0.4)
    stem = out_dir / f"fig14_{suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Fig 14: {stem.relative_to(REPO)}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
