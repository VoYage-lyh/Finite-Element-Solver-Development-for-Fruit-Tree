"""End-to-end CLI: Bayesian calibration → Pareto recommendation → Sobol → figures.

Run with::

    orchard-fem recommend trees/tree_3.json \
        --material-test data/material.json \
        --measured-frf data/T3_frf.csv \
        --measured-modal 6.05 11.21 18.38 \
        --candidate-clamps trunk:0.35,0.50,0.65 \
        --output-dir results/T3_recommendation/

Writes a JSON summary + posterior samples (NPZ) + paper figures.

This command keeps the heavy imports (matplotlib, emcee, SALib) lazy so that
``orchard-fem --help`` stays fast even on minimal installations.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path


def _dump_json(obj, path: Path) -> None:
    """Tolerant JSON dump that handles dataclasses, numpy scalars, and tuples."""
    import numpy as np

    def default(o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"Cannot serialise {type(o).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=default)


def _parse_candidate_clamps(spec: str) -> list[tuple[str, float]]:
    """Parse ``"trunk:0.35,0.50,0.65"`` → ``[("trunk", 0.35), ...]``."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--candidate-clamps expects 'branch_id:s1,s2,...' (got {spec!r})"
        )
    branch_id, s_list = spec.split(":", 1)
    return [(branch_id.strip(), float(s.strip())) for s in s_list.split(",")]


def _load_measured_frf(csv_path: Path):
    """Load a 2- or 3-column CSV: freq, magnitude[, phase_deg]."""
    import numpy as np

    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"FRF CSV must have ≥ 2 columns (freq, magnitude); got {data.shape}."
        )
    return data[:, 0], data[:, 1]


def _handle_recommend(args: argparse.Namespace, application) -> int:
    """Driver for the end-to-end recommend workflow."""
    # All heavy imports live here so the CLI dispatch stays cheap.
    import numpy as np

    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.io.material_test import (
        BranchMaterialTestSummary, priors_from_material_test,
    )
    from orchard_fem.calibration import (
        BayesianLikelihood, ForwardResult,
        cache_forward, cache_pareto,
        run_emcee_calibration, gelman_rubin, effective_sample_size,
    )
    from orchard_fem.recommendation import (
        HarvestObjective, propagate_posterior_to_pareto,
    )
    from orchard_fem.sensitivity import (
        SobolInputDef, run_sobol_analysis,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model + material test → priors ─────────────────────────────
    model = load_orchard_model(str(args.model_json))
    material_summary = BranchMaterialTestSummary.from_json(args.material_test)
    priors = priors_from_material_test(material_summary)
    print(f"[recommend] {len(priors)} priors generated from material test "
          f"(n={material_summary.n_samples}).")

    # ── 2. Load measurements ───────────────────────────────────────────────
    frf_freqs, frf_mag = _load_measured_frf(Path(args.measured_frf))
    modal_freqs = np.asarray(args.measured_modal, dtype=float)
    likelihood = BayesianLikelihood(
        modal_frequencies_hz=modal_freqs,
        frf_frequencies_hz=frf_freqs,
        frf_log_magnitudes=np.log(frf_mag),
        modal_sigma_rel=args.modal_sigma_rel,
        frf_sigma_log=args.frf_sigma_log,
    )

    # ── 3. Build the forward operator (FEM closure) ────────────────────────
    forward_operator = application.build_recommend_forward_operator(
        model, frf_freqs,
    )
    if args.cache_size > 0:
        forward_operator = cache_forward(forward_operator, cache_size=args.cache_size)
        print(f"[recommend] Forward-operator LRU cache enabled "
              f"(size={args.cache_size}).")

    # ── 4. Bayesian calibration ────────────────────────────────────────────
    print(f"[recommend] Running emcee: {args.walkers} walkers × {args.steps} steps "
          f"(burn={args.burn})")
    posterior = run_emcee_calibration(
        forward_operator, priors, likelihood,
        n_walkers=args.walkers,
        n_steps=args.steps,
        n_burn=args.burn,
        seed=args.seed,
        progress=args.progress,
    )
    rhat = gelman_rubin(posterior.samples)
    ess = effective_sample_size(posterior.samples)
    diagnostics = {
        "acceptance_fraction": posterior.acceptance_fraction,
        "r_hat": {n: float(r) for n, r in zip(posterior.parameter_names, rhat)},
        "ess": {n: float(e) for n, e in zip(posterior.parameter_names, ess)},
    }
    print(f"[recommend]   acceptance={posterior.acceptance_fraction:.3f}, "
          f"max R̂={float(np.nanmax(rhat)):.3f}, min ESS={float(np.nanmin(ess)):.0f}")

    # Save posterior to NPZ for downstream reuse
    np.savez_compressed(
        output_dir / "posterior.npz",
        parameter_names=np.asarray(posterior.parameter_names),
        samples=posterior.samples,
        log_posterior=posterior.log_posterior,
    )

    # ── 5. Pareto recommendation per candidate clamp point ─────────────────
    flat = posterior.flatten()
    K = min(args.pareto_samples, flat.shape[0])
    posterior_thin = [
        dict(zip(posterior.parameter_names, flat[i]))
        for i in np.linspace(0, flat.shape[0] - 1, K).astype(int)
    ]

    pareto_evaluator = application.build_pareto_evaluator(model, forward_operator)
    if args.cache_size > 0:
        pareto_evaluator = cache_pareto(pareto_evaluator,
                                        cache_size=args.cache_size * 8)
    f_grid = np.arange(args.freq_min, args.freq_max + 1e-9, args.freq_step)
    A_grid = np.arange(args.amp_min, args.amp_max + 1e-9, args.amp_step)
    recommendations = {}
    for spec in args.candidate_clamps:
        for branch_id, target_s in _parse_candidate_clamps(spec):
            clamp_label = f"{branch_id}@{target_s:.2f}"
            print(f"[recommend] Pareto sweep at {clamp_label} "
                  f"(|f|={f_grid.size}, |A|={A_grid.size})")
            rec = propagate_posterior_to_pareto(
                posterior_thin, clamp_label, f_grid, A_grid, pareto_evaluator,
                credible_alpha=0.90,
                coverage_min=args.coverage_min,
                stress_max=args.stress_max,
            )
            recommendations[clamp_label] = rec

    # Pick the best clamp by maximising coverage median minus stress penalty
    # and a frequency CI-tightness bonus.
    def _score(k):
        rec = recommendations[k]
        ci_span = rec.frequency_hz_ci[1] - rec.frequency_hz_ci[0]
        return rec.detachment_coverage_median - 0.05 * ci_span
    best_key = max(recommendations, key=_score)
    print(f"[recommend] Best clamp point: {best_key}")

    # ── 6. Sobol sensitivity at the best clamp ─────────────────────────────
    sobol_inputs = application.build_sobol_inputs(model, priors)
    sobol_forward = application.build_sobol_forward(
        model, pareto_evaluator, f_grid, A_grid,
        clamp_label=best_key,
        constraints=dict(coverage_min=args.coverage_min, stress_max=args.stress_max),
    )
    print(f"[recommend] Running Sobol: N_base={args.sobol_base} "
          f"→ {args.sobol_base * (2 * len(sobol_inputs) + 2)} evaluations")
    sobol_result = run_sobol_analysis(
        sobol_forward, sobol_inputs,
        n_base=args.sobol_base,
        bootstrap_resamples=args.sobol_bootstrap,
        progress=args.progress,
    )
    print(sobol_result.summary_table(top_k=8))

    # ── 7. Figures ─────────────────────────────────────────────────────────
    if args.figures:
        from orchard_fem.visualization.paper_figures import (
            plot_posterior_corner, plot_posterior_predictive_frf,
            plot_cross_tree_recommendations, plot_sobol_bars, savefig,
        )
        savefig(plot_posterior_corner(posterior), output_dir / "fig_corner.png")
        # posterior-predictive FRF
        pred_mag = np.zeros((K, frf_freqs.size))
        for i, params in enumerate(posterior_thin):
            pred_mag[i] = np.abs(forward_operator(params).frf_complex)
        savefig(
            plot_posterior_predictive_frf(
                frf_freqs, frf_mag, pred_mag,
                modal_frequencies_hz=list(modal_freqs),
            ),
            output_dir / "fig_posterior_frf.png",
        )
        savefig(
            plot_cross_tree_recommendations(recommendations),
            output_dir / "fig_clamp_recommendations.png",
        )
        savefig(plot_sobol_bars(sobol_result), output_dir / "fig_sobol.png")

    # ── 8. Summary JSON ────────────────────────────────────────────────────
    summary = {
        "model": str(args.model_json),
        "n_samples": material_summary.n_samples,
        "diagnostics": diagnostics,
        "recommendations": {
            k: asdict(v) for k, v in recommendations.items()
        },
        "best_clamp": best_key,
        "sobol": {
            "first_order": {
                n: idx.as_tuple() for n, idx in sobol_result.first_order.items()
            },
            "total_effect": {
                n: idx.as_tuple() for n, idx in sobol_result.total_effect.items()
            },
            "n_evaluations": sobol_result.n_evaluations,
            "output_mean": sobol_result.output_mean,
            "output_variance": sobol_result.output_variance,
        },
    }
    _dump_json(summary, output_dir / "summary.json")
    print(f"[recommend] Wrote {output_dir/'summary.json'}")
    return 0


def register_recommend_command(
    subparsers: argparse._SubParsersAction,
    application,
) -> None:
    parser = subparsers.add_parser(
        "recommend",
        help=(
            "End-to-end Bayesian calibration → Pareto recommendation → Sobol "
            "→ paper figures for a single tree."
        ),
    )
    parser.add_argument("model_json", type=Path,
                        help="Path to the orchard model JSON file.")
    parser.add_argument(
        "--material-test", type=Path, required=True,
        help="Path to a BranchMaterialTestSummary JSON (see io.material_test).",
    )
    parser.add_argument(
        "--measured-frf", type=Path, required=True,
        help="CSV with columns freq_hz, magnitude[, phase_deg].",
    )
    parser.add_argument(
        "--measured-modal", type=float, nargs="+", required=True,
        metavar="HZ",
        help="Measured natural frequencies (one number per mode, e.g. 6.05 11.21 18.38).",
    )
    parser.add_argument(
        "--candidate-clamps", type=str, nargs="+", required=True,
        metavar="SPEC",
        help="Candidate clamp points, one or more 'branch_id:s1,s2,...' specs.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for summary.json, posterior.npz, and figures.",
    )

    # Noise levels
    parser.add_argument("--modal-sigma-rel", type=float, default=0.03)
    parser.add_argument("--frf-sigma-log", type=float, default=0.15)

    # MCMC settings
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--burn", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)

    # Pareto sweep
    parser.add_argument("--freq-min", type=float, default=4.0)
    parser.add_argument("--freq-max", type=float, default=24.0)
    parser.add_argument("--freq-step", type=float, default=0.5)
    parser.add_argument("--amp-min", type=float, default=5.0)
    parser.add_argument("--amp-max", type=float, default=30.0)
    parser.add_argument("--amp-step", type=float, default=2.5)
    parser.add_argument("--pareto-samples", type=int, default=400,
                        help="Number of posterior samples used in Pareto CI.")
    parser.add_argument("--coverage-min", type=float, default=None,
                        help="Minimum acceptable detachment coverage (∈[0,1]).")
    parser.add_argument("--stress-max", type=float, default=None,
                        help="Maximum acceptable trunk peak stress [Pa].")

    # Sobol
    parser.add_argument("--sobol-base", type=int, default=256,
                        help="Saltelli N_base; total evals = N_base*(2k+2).")
    parser.add_argument("--sobol-bootstrap", type=int, default=200)

    # Performance
    parser.add_argument("--cache-size", type=int, default=1024,
                        help="LRU cache size for forward-operator memoisation "
                             "(0 disables caching; default 1024).")

    # Misc
    parser.add_argument("--figures", action="store_true", default=True,
                        help="Render paper figures (default on).")
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    parser.add_argument("--progress", action="store_true", default=False,
                        help="Show tqdm progress bars for MCMC/Sobol loops.")

    parser.set_defaults(handler=partial(_handle_recommend, application=application))
