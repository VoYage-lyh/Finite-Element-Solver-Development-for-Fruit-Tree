"""Compute a single-branch, single-point FRF for hammer-test validation.

Unlike ``generate_all_figures.py``'s ``_frf_tip_mean`` — which averages the
displacement amplitudes of **all** branch tips in the **ux** direction (a
"whole-tree" averaged FRF that pareto sweep needs) — this script:

* overrides the excitation to be at a specific branch + node + component
  (matching the hammer-impact point and direction);
* extracts the response at a specific branch + station + component
  (matching the single accelerometer channel used in the field test);
* runs the sweep twice — once with the local nonlinear correction stripped
  ($k_3 = 0$, $c_2 = 0$) and once with the manuscript-calibrated nonlinear
  links injected — so the same point-to-point FRF can be compared in both
  modelling fidelities.

Result is cached at ``cache/validation_frf/tree_<n>_<input>_to_<output>.npz``
so re-runs are sub-second.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard_fem.io.loaders.orchard import load_orchard_model

CACHE_DIR = REPO / "cache" / "validation_frf"


def _strip_nonlinear(model):
    """Return a copy with all nonlinear contributions disabled."""
    new_clamps = [
        replace(c, cubic_stiffness=0.0, quadratic_damping=0.0)
        for c in model.clamps
    ]
    new_analysis = replace(
        model.analysis,
        auto_nonlinear_levels=[],
        auto_nonlinear_cubic_scale=0.0,
        auto_nonlinear_randomize=False,
    )
    return replace(model, clamps=new_clamps, analysis=new_analysis)


def _override_excitation(model, *, branch_id, node, comp, amplitude):
    new_excitation = replace(
        model.excitation,
        target_branch_id=branch_id,
        target_node=node,
        target_component=comp,
        amplitude=amplitude,
        target_s=None,  # use named node, not arc-length
    )
    return replace(model, excitation=new_excitation)


def _frf_single_point(
    model, f_min, f_max, steps,
    *, output_obs_name,
):
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
    if output_obs_name not in res.observation_names:
        suggestions = [n for n in res.observation_names
                       if output_obs_name.split("_")[1] in n][:8]
        raise RuntimeError(
            f"Observation {output_obs_name!r} not found. "
            f"Examples: {suggestions}"
        )
    idx = res.observation_names.index(output_obs_name)
    mags = np.array([p.observation_magnitudes[idx] for p in res.points])
    return freqs, mags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=int, default=1, help="Tree id, 1..5.")
    parser.add_argument(
        "--input-branch", default="left_leader",
        help="Excitation branch_id (default left_leader = level-1 branch 1).",
    )
    parser.add_argument(
        "--input-node", choices=("root", "mid", "tip"), default="root",
        help="Excitation station along the branch (default root).",
    )
    parser.add_argument(
        "--input-comp", choices=("ux", "uy", "uz"), default="uz",
        help="Excitation direction (default uz, matching the hammer's Z channel).",
    )
    parser.add_argument(
        "--output-branch", default=None,
        help="Response branch_id (default: same as --input-branch).",
    )
    parser.add_argument(
        "--output-station", choices=("root", "mid", "tip"), default="tip",
    )
    parser.add_argument(
        "--output-comp", choices=("ux", "uy", "uz"), default="uz",
    )
    parser.add_argument("--fmin", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--amplitude", type=float, default=10.0,
        help="Excitation amplitude [N] (default 10 to match pareto pipeline).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cache and recompute.",
    )
    parser.add_argument(
        "--from-calibration", default=None,
        help="Path to a calibration .npz produced by "
        "calibrate_from_measured_frf.py. If supplied, the optimised "
        "(rayleigh_beta, k3_scale, c2_scale) are applied to the NONLINEAR "
        "sweep and the cache is saved with a `_calibrated` suffix.",
    )
    parser.add_argument(
        "--rayleigh-beta", type=float, default=None,
        help="Override Rayleigh β for the nonlinear sweep (alternative to "
        "--from-calibration). Linear sweep keeps the JSON's β.",
    )
    parser.add_argument("--k3-scale", type=float, default=1.0)
    parser.add_argument("--c2-scale", type=float, default=1.0)
    args = parser.parse_args()

    out_branch = args.output_branch or args.input_branch
    output_obs = f"obs_{out_branch}_{args.output_station}_{args.output_comp}"

    # Resolve calibration overrides
    calibrated = False
    beta_override = args.rayleigh_beta
    k3_scale = float(args.k3_scale)
    c2_scale = float(args.c2_scale)
    if args.from_calibration:
        d = np.load(args.from_calibration)
        beta_override = float(d["beta_opt"])
        k3_scale = float(d["k3_scale_opt"])
        c2_scale = float(d["c2_scale_opt"])
        calibrated = True
        print(f"Loaded calibration from {args.from_calibration}: "
              f"β={beta_override:.3e}, k3×{k3_scale:.2f}, c2×{c2_scale:.2f}")
    elif (beta_override is not None) or (k3_scale != 1.0) or (c2_scale != 1.0):
        calibrated = True

    suffix = "_calibrated" if calibrated else ""
    cache_path = (
        CACHE_DIR
        / f"tree_{args.tree}_{args.input_branch}_{args.input_node}_{args.input_comp}"
        f"_to_{out_branch}_{args.output_station}_{args.output_comp}{suffix}.npz"
    )

    if cache_path.exists() and not args.force:
        d = np.load(cache_path)
        freqs = d["freqs"]
        mag_lin = d["mag_lin"]
        mag_nl = d["mag_nl"]
        print(f"[cache] hit {cache_path.relative_to(REPO)}")
    else:
        model_path = REPO / "trees" / f"tree_{args.tree}.json"
        print(f"Loading {model_path}")
        model = load_orchard_model(str(model_path))

        # Override excitation: match the hammer test setup.
        model = _override_excitation(
            model,
            branch_id=args.input_branch,
            node=args.input_node,
            comp=args.input_comp,
            amplitude=args.amplitude,
        )
        print(f"Input  : {args.input_branch}@{args.input_node} {args.input_comp}, "
              f"{args.amplitude} N")
        print(f"Output : {output_obs}")

        print(f"Linear sweep ({args.fmin}–{args.fmax} Hz, {args.steps} pts) ...")
        t0 = time.time()
        freqs, mag_lin = _frf_single_point(
            _strip_nonlinear(model),
            args.fmin, args.fmax, args.steps,
            output_obs_name=output_obs,
        )
        print(f"  linear     {time.time() - t0:.1f} s, "
              f"peak |U|={mag_lin.max():.3e} m @ {freqs[mag_lin.argmax()]:.2f} Hz")

        # Apply calibration overrides (β, k3_scale, c2_scale) to the
        # nonlinear sweep only — keeps the linear baseline reproducible.
        nl_model = model
        if beta_override is not None:
            nl_model = replace(
                nl_model,
                analysis=replace(nl_model.analysis, rayleigh_beta=float(beta_override)),
            )
            print(f"  override rayleigh_beta = {beta_override:.3e}")
        import orchard_fem.fenicsx.joints as _joints
        orig_k3 = _joints._K3_DOWNSCALE
        orig_c2 = _joints._C2_DOWNSCALE
        _joints._K3_DOWNSCALE = orig_k3 * k3_scale
        _joints._C2_DOWNSCALE = orig_c2 * c2_scale
        try:
            print("Nonlinear sweep ...")
            t0 = time.time()
            _, mag_nl = _frf_single_point(
                nl_model,
                args.fmin, args.fmax, args.steps,
                output_obs_name=output_obs,
            )
        finally:
            _joints._K3_DOWNSCALE = orig_k3
            _joints._C2_DOWNSCALE = orig_c2
        print(f"  nonlinear  {time.time() - t0:.1f} s, "
              f"peak |U|={mag_nl.max():.3e} m @ {freqs[mag_nl.argmax()]:.2f} Hz")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            freqs=freqs, mag_lin=mag_lin, mag_nl=mag_nl,
            input_branch=args.input_branch, input_node=args.input_node,
            input_comp=args.input_comp,
            output_branch=out_branch, output_station=args.output_station,
            output_comp=args.output_comp,
            amplitude=args.amplitude,
        )
        print(f"[cache] wrote {cache_path.relative_to(REPO)}")

    print()
    print(f"Result: freqs {freqs.min():.2f}–{freqs.max():.2f} Hz, "
          f"{freqs.size} points")
    print(f"  linear    peak {mag_lin.max():.3e} @ "
          f"{freqs[mag_lin.argmax()]:.2f} Hz")
    print(f"  nonlinear peak {mag_nl.max():.3e} @ "
          f"{freqs[mag_nl.argmax()]:.2f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
