"""Clamp energy-reach map: how far each grip's excitation actually reaches.

Motivation
----------
The harvest recommendation excites at ONE clamp and sweeps frequency to try to
shed fruit across the whole tree. But excitation energy attenuates with distance
(damping + structural path), so a grip on the left of the tree cannot move — let
alone detach fruit at — the far-right tips, no matter the frequency. This script
makes that concrete: for every candidate clamp it drives a harmonic-displacement
FRF sweep at that grip and records each branch tip's **peak response amplitude**,
then normalises by the strongest-responding branch (so the driven region ≈ 1 and
distant branches decay toward 0). Two views:

  * a clamp×branch heatmap (both axes ordered left→right by position) — a grip
    lights up its own region and fades far away; and
  * normalised response vs. clamp→branch distance — the spatial attenuation.

This shows why one clamp's coverage caps out and motivates the multi-clamp
schedule (``compute_multiclamp_harvest_schedule``).

Example
-------
    python scripts/clamp_energy_reach.py trees/tree_3.json --out results/accuracy_study
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from orchard_fem.domain import ExcitationKind
from orchard_fem.io import load_orchard_model
from orchard_fem.topology import ObservationPoint
from orchard_fem.visualization.model_scene import hierarchical_labels
from orchard_fem.workflows.harvest_recommendation import (
    RecommendationOptions,
    candidate_clamp_labels,
    generate_linear_fruits,
)


def _pt(model, branch_id, s):
    p = model.require_branch(branch_id).path.point_at(s)
    return np.array([p.x, p.y, p.z], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_json", type=Path)
    parser.add_argument("--band", type=str, default="2,15", help="FRF sweep band 'min,max' [Hz].")
    parser.add_argument("--steps", type=int, default=14, help="FRF sweep steps.")
    parser.add_argument("--degree", type=int, default=2, help="FE element order (2 = converged).")
    parser.add_argument("--max-clamps", type=int, default=16)
    parser.add_argument("--drive-mm", type=float, default=10.0, help="Clamp displacement amplitude [mm].")
    parser.add_argument("--rayleigh-beta", type=float, default=None,
                        help="Override stiffness-proportional damping β (model default ~1e-4 ≈ 0.25%% "
                             "ζ — far below real green-wood 5–15%%; try 0.006 ≈ ~8%% ζ at 8 Hz).")
    parser.add_argument("--out", type=Path, default=Path("results/accuracy_study"))
    args = parser.parse_args()

    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )

    fmin, fmax = (float(x) for x in args.band.split(","))
    model = load_orchard_model(str(args.model_json))
    if args.rayleigh_beta is not None:
        model = replace(model, analysis=replace(model.analysis, rayleigh_beta=args.rayleigh_beta))
    policy = replace(model.fruit_policy, detachment_displacement_m=0.002)
    model = replace(model, fruit_policy=policy, fruits=generate_linear_fruits(model, policy, 0.05))
    # Absolute detachment-reach uses the SAME physics as the schedule: a fruit
    # sheds when its inertia force m·ω²·|u| reaches the detachment force F_detach.
    m_fruit = float(getattr(policy, "mean_fruit_mass_kg", None) or 0.05)
    f_detach = float(getattr(policy, "mean_detachment_force_N", None) or 5.0)

    hlabels = hierarchical_labels(json.loads(args.model_json.read_text())["branches"])
    branches = sorted({f.branch_id for f in model.fruits}, key=lambda b: _pt(model, b, 1.0)[0])
    bidx = {b: i for i, b in enumerate(branches)}

    clamps = candidate_clamp_labels(model, RecommendationOptions())[: args.max_clamps]
    clamps.sort(key=lambda cl: _pt(model, cl.split("@")[0], float(cl.split("@")[1]))[0])

    # one observation per branch tip (full translational magnitude)
    obs = [ObservationPoint(observation_id=f"reach_{b}", target_type="branch", target_id=b,
                            target_node="tip", target_components=["ux", "uy", "uz"]) for b in branches]
    analysis = replace(model.analysis, frequency_start_hz=fmin, frequency_end_hz=fmax,
                       frequency_steps=int(args.steps))

    reach = np.zeros((len(clamps), len(branches)), dtype=float)
    distances: list[tuple[float, float]] = []

    for ci, clamp in enumerate(clamps):
        cb, cs = clamp.split("@")
        clamp_pt = _pt(model, cb, float(cs))
        print(f"[{ci + 1}/{len(clamps)}] FRF reach for clamp {hlabels.get(cb, cb)}@{cs}…", flush=True)
        exc = replace(model.excitation, kind=ExcitationKind.HARMONIC_DISPLACEMENT,
                      target_branch_id=cb, target_s=float(cs), target_component="ux",
                      amplitude=args.drive_mm / 1000.0)
        mc = replace(model, excitation=exc, analysis=analysis, observations=obs)
        res = solve_embedded_beam_frequency_response_experiment(mc, polynomial_degree=args.degree).result
        col = {n: i for i, n in enumerate(res.observation_names)}
        for b in branches:
            cols = [col[f"reach_{b}_{c}"] for c in ("ux", "uy", "uz") if f"reach_{b}_{c}" in col]
            # detachment ratio r = m·ω²·|u| / F_detach, peak over the sweep (absolute,
            # NOT normalised — so distance attenuation is visible)
            r_peak = 0.0
            for p in res.points:
                u = float(np.sqrt(sum(p.observation_magnitudes[c] ** 2 for c in cols)))
                omega = 2.0 * np.pi * p.frequency_hz
                r_peak = max(r_peak, m_fruit * omega * omega * u / f_detach)
            reach[ci, bidx[b]] = r_peak
            distances.append((float(np.linalg.norm(_pt(model, b, 1.0) - clamp_pt)), r_peak))

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.model_json.stem
    fig, (axh, axd) = plt.subplots(1, 2, figsize=(15, 6))

    shown = np.clip(reach, 1e-2, None)
    im = axh.imshow(shown, aspect="auto", cmap="magma",
                    norm=LogNorm(vmin=1e-2, vmax=max(1.0, shown.max())))
    axh.set_xticks(range(len(branches)))
    axh.set_xticklabels([hlabels.get(b, b) for b in branches], rotation=90, fontsize=7)
    axh.set_yticks(range(len(clamps)))
    axh.set_yticklabels([f"{hlabels.get(c.split('@')[0], c.split('@')[0])}@{c.split('@')[1]}" for c in clamps],
                        fontsize=7)
    axh.set_xlabel("branch tip (left → right)")
    axh.set_ylabel("clamp / grip (left → right)")
    axh.set_title(f"{stem}: detachment drive r = m·ω²·|u| / F_detach  (drive {args.drive_mm:g} mm)\n"
                  f"r ≥ 1 ⇒ sheds (× marks);  dark ⇒ energy doesn't reach that branch")
    fig.colorbar(im, ax=axh, label="detachment ratio r", fraction=0.046)
    ys, xs = np.where(reach >= 1.0)
    axh.scatter(xs, ys, s=10, marker="x", color="cyan", linewidths=0.7)

    dd = np.array(distances)
    axd.scatter(dd[:, 0], np.clip(dd[:, 1], 1e-2, None), s=10, alpha=0.35)
    axd.axhline(1.0, color="red", ls="--", lw=1, label="r = 1 (detachment threshold)")
    axd.set_yscale("log")
    axd.set_xlabel("clamp → branch-tip distance [m]")
    axd.set_ylabel("detachment ratio r (log)")
    axd.set_title("Energy reach decays with distance\n(points below r=1 cannot be shed from that grip)")
    axd.grid(True, which="both", alpha=0.3)
    axd.legend()

    beta_tag = f"_beta{args.rayleigh_beta:g}" if args.rayleigh_beta is not None else "_betamodel"
    fig.suptitle(f"clamp energy reach — {stem}, β={args.rayleigh_beta if args.rayleigh_beta is not None else 'model(~1e-4)'}",
                 y=1.02, fontsize=9)
    fig.tight_layout()
    out_png = args.out / f"clamp_energy_reach_{stem}{beta_tag}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    per_clamp = (reach >= 1.0).sum(axis=1)
    union = (reach >= 1.0).any(axis=0).sum()
    print(f"\nFruit branches: {len(branches)}  (drive {args.drive_mm:g} mm; r≥1 ⇒ detaches)")
    print(f"Best single clamp sheds: {per_clamp.max()}/{len(branches)} "
          f"({per_clamp.max() / len(branches):.0%})")
    print(f"Union over ALL clamps:   {union}/{len(branches)} ({union / len(branches):.0%})"
          f"  ← multi-clamp ceiling")
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
