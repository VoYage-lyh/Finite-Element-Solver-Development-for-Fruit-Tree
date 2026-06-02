"""Brute-force locate the field accelerometers in the FE topology.

Given two fixed-frequency hold-segment records (typical: two simultaneous
sensors on different branches), iterate over every (branch, station,
component) observation point declared in ``tree_<n>.json`` and score how
well the calibrated single-branch FE predicts each sensor's measured RMS
sequence. The output is a coverage- and error-ranked CSV plus the top-1
Fig 14 for each sensor.

Efficient version: **one FE sweep** at the calibrated (β, k_3, c_2) and
the agreed-on shaker force amplitude F. The FE returns displacement
amplitudes at every observation point in the model; we just extract each
candidate and score it. Total compute ≈ N_freqs × ~30 ms + scoring ≈
sub-second per sensor regardless of how many candidates we scan.

Usage::

    python scripts/scan_sensor_locations.py \\
        --sensor sensor1=results/calibration/fixed_freq_segments_sensor1.csv \\
        --sensor sensor2=results/calibration/fixed_freq_segments_sensor2.csv \\
        --top 8

The shaker amplitude F is held fixed at ``--F`` (default 210 N — the
sensor-1 fitted value). To re-fit F jointly, run the iterate-loop
externally first or accept the supplied value as a working assumption.
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


def patched_nonlinear(k3_scale, c2_scale):
    import orchard_fem.fenicsx.joints as j
    orig_k3 = j._K3_DOWNSCALE
    orig_c2 = j._C2_DOWNSCALE
    j._K3_DOWNSCALE = orig_k3 * float(k3_scale)
    j._C2_DOWNSCALE = orig_c2 * float(c2_scale)
    return j, orig_k3, orig_c2


def run_full_sweep(model, freqs_hz):
    """One FE sweep returning |U|(f) at *every* observation point."""
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
    f_out = np.array([p.frequency_hz for p in res.points])
    n_freq = f_out.size
    n_obs = len(res.observation_names)
    mag_all = np.empty((n_freq, n_obs))
    for i, p in enumerate(res.points):
        mag_all[i] = p.observation_magnitudes
    return f_out, res.observation_names, mag_all


def score_candidate(seg_drive, seg_rms, freqs_sim, U_mag, F_drive):
    """For one candidate observation column |U|(f), compute predicted RMS at
    each measured segment and return (coverage_at_50pct_band, mean_abs_err,
    pred_rms_at_each_segment).

    Note: we don't have proper 90% CI here (no posterior). We use a
    placeholder ±30% band as a coarse coverage proxy."""
    U_at = np.interp(seg_drive, freqs_sim, U_mag)
    omega = 2.0 * math.pi * seg_drive
    pred_rms = omega ** 2 * U_at / math.sqrt(2.0)  # |U| already scales with F
    # F was set as model.excitation.amplitude, so U already proportional.
    rel_err = (seg_rms - pred_rms) / np.maximum(pred_rms, 1e-12)
    abs_log_err = np.abs(np.log(np.maximum(seg_rms, 1e-12) /
                                np.maximum(pred_rms, 1e-12)))
    # Coverage placeholder: count fraction within ±30% (i.e. |log ratio| ≤ ln 1.3)
    covered = (abs_log_err <= math.log(1.3)).mean()
    mean_abs_err = float(np.mean(np.abs(rel_err)) * 100.0)
    rms_log_err = float(np.sqrt(np.mean(abs_log_err ** 2)))
    return covered, mean_abs_err, pred_rms, rms_log_err


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=int, default=1)
    parser.add_argument(
        "--sensor", action="append", required=True,
        help="Sensor spec 'name=path/to/segments.csv'. Repeat for each sensor.",
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
    parser.add_argument(
        "--F", type=float, default=210.0,
        help="Shaker force amplitude (N), held fixed during the scan "
        "(default 210 N from sensor-1 self-consistent fit).",
    )
    parser.add_argument("--fmin-segment", type=float, default=5.0)
    parser.add_argument("--fmax-segment", type=float, default=30.0)
    parser.add_argument("--top", type=int, default=8,
                        help="Top-N candidates per sensor to print.")
    args = parser.parse_args()

    # ------------------------------ parse sensors
    sensors: dict[str, dict] = {}
    for spec in args.sensor:
        if "=" not in spec:
            raise SystemExit(f"--sensor expects name=path, got {spec!r}")
        name, p = spec.split("=", 1)
        path = REPO / p
        with path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        d = np.array([float(r["drive_hz"]) for r in rows])
        r = np.array([float(r["rms_steady"]) for r in rows])
        keep = (d >= args.fmin_segment) & (d <= args.fmax_segment)
        d, r = d[keep], r[keep]
        order = np.argsort(d)
        sensors[name] = {"drive": d[order], "rms": r[order]}
        print(f"[{name}] {d.size} kept segments, "
              f"f range {d.min():.1f}–{d.max():.1f} Hz, "
              f"RMS range {r.min():.1f}–{r.max():.1f} m·s⁻²")

    # ------------------------------ load calibration
    cal = np.load(REPO / args.calibration_cache)
    beta_opt = float(cal["beta_opt"])
    k3_scale = float(cal["k3_scale_opt"])
    c2_scale = float(cal["c2_scale_opt"])
    print(f"\nCalibrated: β={beta_opt:.3e}, k3×{k3_scale:.2f}, c2×{c2_scale:.2f}")
    print(f"Shaker force F = {args.F:.0f} N (fixed for the scan).")

    # ------------------------------ unique eval frequencies (union of sensors)
    all_freqs = np.unique(np.round(
        np.concatenate([s["drive"] for s in sensors.values()]), 1,
    ))
    print(f"\nUnique segment frequencies: {all_freqs.size} "
          f"({all_freqs.min():.1f}–{all_freqs.max():.1f} Hz)")

    # ------------------------------ load model, override excitation + β
    model = load_orchard_model(str(REPO / "trees" / f"tree_{args.tree}.json"))
    model = _override_excitation(
        model, branch_id=args.input_branch, node=args.input_node,
        comp=args.input_comp, amplitude=args.F,
    )
    model = replace(model, analysis=replace(model.analysis, rayleigh_beta=beta_opt))

    # ------------------------------ ONE FE sweep (gets every observation)
    j_mod, orig_k3, orig_c2 = patched_nonlinear(k3_scale, c2_scale)
    try:
        t0 = time.time()
        f_sim, obs_names, U_all = run_full_sweep(model, all_freqs)
        print(f"FE sweep ({all_freqs.size} freqs, "
              f"{len(obs_names)} observations) → {time.time()-t0:.1f} s")
    finally:
        j_mod._K3_DOWNSCALE = orig_k3
        j_mod._C2_DOWNSCALE = orig_c2

    # ------------------------------ score every (branch, station, comp)
    # Only retain obs_<...> rows (skip excitation_*, etc.)
    candidate_idx = [
        i for i, n in enumerate(obs_names)
        if n.startswith("obs_") and any(n.endswith(c) for c in ("_ux", "_uy", "_uz"))
    ]
    print(f"Scoring {len(candidate_idx)} candidate observation points × "
          f"{len(sensors)} sensors …")

    rows_out: list[dict] = []
    for idx in candidate_idx:
        obs = obs_names[idx]
        U_col = U_all[:, idx]
        rec: dict = {"observation": obs}
        for sname, sdata in sensors.items():
            cov, abs_err, pred, log_rmse = score_candidate(
                sdata["drive"], sdata["rms"], f_sim, U_col, args.F,
            )
            rec[f"{sname}_coverage_pm30"] = cov
            rec[f"{sname}_mean_abs_err_pct"] = abs_err
            rec[f"{sname}_log_rmse"] = log_rmse
        # Combined score: average log RMSE across sensors (lower is better)
        rec["combined_log_rmse"] = float(np.mean([
            rec[f"{s}_log_rmse"] for s in sensors
        ]))
        rows_out.append(rec)

    # ------------------------------ rank and print
    rows_out.sort(key=lambda r: r["combined_log_rmse"])

    print()
    print("=" * 96)
    print(f"Top {args.top} observation points (ranked by mean log-RMSE across sensors):")
    print("-" * 96)
    headers = ["rank", "observation", "combined log-RMSE"]
    for s in sensors:
        headers.extend([f"{s} cov", f"{s} err%"])
    print(("{:<5}{:<40}{:>18}" + "{:>10}{:>9}" * len(sensors)).format(*headers))
    print("-" * 96)
    for rank, r in enumerate(rows_out[: args.top], 1):
        line_vals = [rank, r["observation"], f"{r['combined_log_rmse']:.3f}"]
        for s in sensors:
            line_vals.append(f"{r[f'{s}_coverage_pm30']*100:.0f}%")
            line_vals.append(f"{r[f'{s}_mean_abs_err_pct']:.0f}")
        print(("{:<5}{:<40}{:>18}" + "{:>10}{:>9}" * len(sensors)).format(*line_vals))
    print("=" * 96)

    # ------------------------------ best per-sensor
    print(f"\nBest single observation per sensor (lowest log-RMSE for that sensor):")
    best_per_sensor = {}
    for sname in sensors:
        best = min(rows_out, key=lambda r: r[f"{sname}_log_rmse"])
        best_per_sensor[sname] = best
        print(f"  {sname}: {best['observation']}  "
              f"(cov={best[f'{sname}_coverage_pm30']*100:.0f}%, "
              f"err={best[f'{sname}_mean_abs_err_pct']:.0f}%, "
              f"log-RMSE={best[f'{sname}_log_rmse']:.3f})")

    # ------------------------------ write full ranking CSV
    out_dir = REPO / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sensor_location_scan.csv"
    fieldnames = ["observation", "combined_log_rmse"]
    for s in sensors:
        fieldnames.extend([f"{s}_coverage_pm30", f"{s}_mean_abs_err_pct",
                           f"{s}_log_rmse"])
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nFull ranking: {csv_path.relative_to(REPO)}")

    # ------------------------------ emit suggested commands
    print("\nSuggested follow-up commands (render Fig 14 with best output):")
    for sname, best in best_per_sensor.items():
        # parse obs name back to branch/station/comp
        n = best["observation"][len("obs_"):]
        comp = n[-2:]  # ux / uy / uz
        rest = n[:-3]   # strip _ux
        for station in ("root", "mid", "tip"):
            if rest.endswith("_" + station):
                branch = rest[:-(len(station) + 1)]
                print(
                    f"  python scripts/render_fixed_freq_validation.py "
                    f"--segments-csv results/calibration/fixed_freq_segments_{sname}.csv "
                    f"--output-branch {branch} --output-station {station} "
                    f"--output-comp {comp} --name {sname}_best"
                )
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
