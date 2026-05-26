"""End-to-end self-consistency check with **multi-clamp** Pareto search.

For each tree we:
    1. Run a wide FRF sweep to find the dominant in-band local-max resonance.
    2. Tighten the detachment displacement to 2 mm.
    3. Enumerate candidate clamps:
           * Trunk heights at s = 0.25, 0.40, 0.55, 0.70, 0.85
           * Root (s = 0) of every branch joined directly to the trunk
       For each clamp, run a Pareto sweep over (f, A) around the resonance
       and extract a knee.
    4. Select the **best clamp** as the one whose knee is closest to the
       ideal (coverage = 1, σ = 0) point in normalised objective space.

Outputs (in ``results/``), organised by figure type:

* ``results/pareto/tree_<n>.{png,pdf}``    — multi-clamp Pareto + best knee
* ``results/frf/tree_<n>.{png,pdf}``       — FRF sweep with resonance marker
* ``results/response/tree_<n>.{png,pdf}``  — side-view detachment map
* ``results/pareto/all_trees.{png,pdf}``   — best-knee Pareto per tree
* ``results/frf/all_trees.{png,pdf}``      — all 5 FRFs stacked
* ``results/response/all_trees.{png,pdf}`` — side-view, 5 panels
* ``results/summary/knees.{png,pdf}``      — best (clamp, f, A, act, σ) per tree
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ────────────────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────────────────
_FEASIBLE_BAND_HZ = (3.0, 20.0)

# Per-tree legend placement for the multi-clamp Pareto figure. Override when
# the default "lower right" gets crowded by the front. Anything missing
# defaults to "lower right".
_PARETO_LEGEND_LOC = {
    "tree_1": "upper left",
}
_TRUNK_CLAMP_S = (0.25, 0.40, 0.55, 0.70, 0.85)
_AMPLITUDE_GRID_MM = (5.0, 10.0, 15.0, 20.0, 30.0)

# Sanity ceiling on the trunk peak bending stress. Wood typically fails in
# tension at 50–100 MPa; anything above ~100 MPa is well outside the linear
# elastic regime our beam model assumes, and in practice tends to be the
# fingerprint of a numerically pathological (clamp, f, A) point (e.g.
# anti-resonance with near-singular dynamic stiffness, or penalty-Dirichlet
# rounding artefacts amplified by sharp local resonance). Such points are
# dropped from the feasible Pareto set so they cannot become the knee.
_STRESS_SANITY_CEILING_PA = 100.0e6


# ────────────────────────────────────────────────────────────────────────────
#  Publication-grade matplotlib style
# ────────────────────────────────────────────────────────────────────────────
def _apply_pub_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fontsize": 12,
        "legend.edgecolor": "#cccccc",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_both(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))


# ────────────────────────────────────────────────────────────────────────────
#  Resonance detection
# ────────────────────────────────────────────────────────────────────────────
def _find_in_band_resonance(
    freqs: np.ndarray,
    mags: np.ndarray,
    band: tuple[float, float],
    *,
    prominence_ratio: float = 0.10,
) -> tuple[int, bool]:
    """Return ``(peak_index, has_genuine_local_max)`` inside *band*."""
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        raise RuntimeError(
            f"No FRF samples in {band[0]}–{band[1]} Hz; widen the sweep."
        )

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

    # Fallback: curvature inflection.
    curvature = np.zeros_like(mags)
    curvature[1:-1] = log_mags[2:] - 2.0 * log_mags[1:-1] + log_mags[:-2]
    band_idx = np.flatnonzero(in_band)
    interior = band_idx[(band_idx > 0) & (band_idx < mags.size - 1)]
    if interior.size == 0:
        peak_idx = int(band_idx[int(np.argmax(mags[in_band]))])
        return peak_idx, False
    peak_idx = int(interior[int(np.argmin(curvature[interior]))])
    return peak_idx, False


# ────────────────────────────────────────────────────────────────────────────
#  Workflow data containers
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class ClampResult:
    clamp_label: str        # internal label, e.g. "left_scaffold@0.00"
    display_label: str      # reader-friendly, e.g. "B1 root"
    front: object           # ParetoFront
    knee: object            # ParetoKnee
    # Per-(f, A) fruit-level outcomes on the best clamp's grid — populated
    # only for the chosen best clamp, consumed by the greedy multi-stage
    # sequence in ``_greedy_sequence``. ``None`` for non-best clamps.
    grid_outcomes: dict | None = None


@dataclass
class TreeResult:
    label: str
    n_fruits: int
    freqs: np.ndarray
    mags: np.ndarray
    f_resonance: float
    clamps: list[ClampResult] = field(default_factory=list)
    best_idx: int = 0
    # Kept for downstream visualisation (response map). Not used by Pareto.
    model: object = None
    theta: dict = field(default_factory=dict)
    label_map: dict = field(default_factory=dict)

    @property
    def best(self) -> ClampResult:
        return self.clamps[self.best_idx]


# ────────────────────────────────────────────────────────────────────────────
#  Per-tree workflow
# ────────────────────────────────────────────────────────────────────────────
def _coarse_frf_sweep(model, f_min: float, f_max: float, steps: int):
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
    name_to_idx = {n: i for i, n in enumerate(res.observation_names)}
    tip_obs = [n for n in res.observation_names if n.endswith("_tip_ux")]
    mags = np.zeros_like(freqs)
    for j, p in enumerate(res.points):
        mags[j] = float(np.mean([p.observation_magnitudes[name_to_idx[n]]
                                  for n in tip_obs]))
    return freqs, mags


def _candidate_clamps(model) -> list[str]:
    """Return clamp labels for trunk heights + each trunk-child branch root."""
    labels = [f"trunk@{s:.2f}" for s in _TRUNK_CLAMP_S]
    trunk_node = model.topology.require_node("trunk")
    for bid in trunk_node.child_branch_ids:
        labels.append(f"{bid}@0.00")
    return labels


def _build_hierarchical_label_map(model) -> dict[str, str]:
    """Return ``{branch_id: hierarchical_label}`` matching ``view-tree``.

    Trunk → ``T``; primary branches → ``1, 2, …`` ordered left→right by the
    branch endpoint's x-coordinate; secondaries → ``1.1, 1.2, …``; and so on.
    Mirrors :func:`orchard_fem.visualization.model_scene.hierarchical_labels`
    so figure callouts stay consistent with the 3D scene plot.
    """
    children_by_parent: dict[str | None, list[str]] = {}
    for n_id in model.topology.nodes:
        node = model.topology.require_node(n_id)
        parent = node.parent_branch_id
        children_by_parent.setdefault(parent, []).append(n_id)

    def _end_x(branch_id: str) -> float:
        branch = next(b for b in model.branches if b.branch_id == branch_id)
        return float(branch.path.end.x)

    out: dict[str, str] = {}

    def _assign(parent_id: str | None, parent_label: str) -> None:
        kids = sorted(children_by_parent.get(parent_id, []), key=_end_x)
        for i, kid in enumerate(kids, start=1):
            kid_label = f"{parent_label}.{i}" if parent_label else str(i)
            out[kid] = kid_label
            _assign(kid, kid_label)

    for root_id in children_by_parent.get(None, []):
        out[root_id] = "T"
        _assign(root_id, "")

    return out


def _pretty_clamp_label(raw: str, label_map: dict[str, str]) -> str:
    """Translate ``branch_id@s`` to ``HierarchicalLabel@position`` notation.

    ``s = 0`` becomes ``root``, ``s = 1`` becomes ``tip``, ``s = 0.5`` becomes
    ``mid``; intermediate fractions are rendered as ``XX%``. For example,
    ``left_scaffold@0.00`` (with hierarchical label ``1``) → ``B1 root``;
    ``trunk@0.25`` → ``T @ 25%``.
    """
    if "@" in raw:
        bid, s_str = raw.split("@", 1)
        s = float(s_str)
    else:
        bid, s = raw, 0.0

    hier = label_map.get(bid, bid)
    if hier == "T":
        prefix = "T"
    else:
        prefix = f"B{hier}"

    if abs(s) < 1.0e-6:
        suffix = "root"
    elif abs(s - 1.0) < 1.0e-6:
        suffix = "tip"
    elif abs(s - 0.5) < 1.0e-6:
        suffix = "mid"
    else:
        suffix = f"{s * 100:.0f}%"

    return f"{prefix} @ {suffix}"


def _knee_distance_to_ideal(
    knees: list, *, sigma_norm: float
) -> np.ndarray:
    """Normalised distance from each knee to the ideal (cov=1, σ=0) point.

    The coverage axis is already in [0, 1]. The stress axis is normalised by
    ``sigma_norm`` — typically the max stress observed across all knees of
    this tree — so the two axes carry comparable weight.
    """
    distances = np.zeros(len(knees))
    for i, k in enumerate(knees):
        cov = float(k.detachment_coverage)
        sigma_n = float(k.trunk_max_stress) / max(sigma_norm, 1.0e-12)
        distances[i] = float(np.sqrt((1.0 - cov) ** 2 + sigma_n ** 2))
    return distances


def _evaluate_tree(model_path: Path, label: str) -> TreeResult:
    from orchard_fem.calibration.fenicsx_bridge import (
        build_fenicsx_pareto_evaluator,
    )
    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.recommendation.pareto import pareto_front_from_grid

    print(f"\n[{label}] loading {model_path.name} …")
    model = load_orchard_model(str(model_path))

    # Replace the policy-driven fruit list with a dense linear distribution:
    # every non-trunk branch carries one fruit per 5 % of its arc length
    # (20 fruits per branch). The attachment stiffness varies linearly along
    # the branch — stiffer near the root (older, woodier stalks) and softer
    # at the tip (young, easily-snapped peduncles).
    new_policy = replace(
        model.fruit_policy,
        detachment_displacement_m=0.002,
    )
    dense_fruits = _generate_linear_fruits(model, new_policy, spacing=0.05)
    model = replace(model, fruits=dense_fruits, fruit_policy=new_policy)

    fruit_branches = {f.branch_id for f in model.fruits}
    print(f"[{label}]   fruits={len(model.fruits)} on "
          f"{len(fruit_branches)} branches "
          f"(linear density: every 5% of arc length)")

    print(f"[{label}] FRF sweep 0.5–30 Hz …")
    t0 = time.time()
    freqs, mags = _coarse_frf_sweep(model, 0.5, 30.0, 60)
    peak_idx, has_local_max = _find_in_band_resonance(
        freqs, mags, _FEASIBLE_BAND_HZ,
    )
    f_resonance = float(freqs[peak_idx])
    detect_type = "local max" if has_local_max else "curvature inflection"
    print(f"[{label}]   sweep {time.time() - t0:.1f} s   "
          f"resonance ({detect_type}) @ {f_resonance:.2f} Hz "
          f"({mags[peak_idx] / model.excitation.amplitude * 1e3:.3f} mm/N)")

    theta = {
        "E": float(model.materials[0].youngs_modulus),
        "rho": float(model.materials[0].density),
    }
    # Switch the model to displacement excitation: the eccentric-cam shaker
    # used in real harvesters imposes a strict displacement, not a force.
    from orchard_fem.domain import ExcitationKind
    model = replace(
        model,
        excitation=replace(model.excitation, kind=ExcitationKind.HARMONIC_DISPLACEMENT),
    )
    # amplitude_unit="mm" → A_grid values are interpreted as millimetres of
    # imposed displacement at the clamp.
    evaluator = build_fenicsx_pareto_evaluator(
        model, amplitude_unit="mm", coverage_mode="branch",
    )

    f_grid = [
        max(0.5, f_resonance - 2.0),
        max(0.5, f_resonance - 1.0),
        f_resonance,
        f_resonance + 1.0,
        f_resonance + 2.0,
    ]
    A_grid = list(_AMPLITUDE_GRID_MM)

    clamp_labels = _candidate_clamps(model)
    label_map = _build_hierarchical_label_map(model)
    print(f"[{label}] {len(clamp_labels)} candidate clamps × |f|={len(f_grid)} × "
          f"|A|={len(A_grid)} = {len(clamp_labels) * len(f_grid) * len(A_grid)} "
          f"FE solves")

    clamps: list[ClampResult] = []
    t0 = time.time()
    for clamp_label in clamp_labels:
        try:
            front = pareto_front_from_grid(
                clamp_label, f_grid, A_grid, evaluator, theta,
                stress_max=_STRESS_SANITY_CEILING_PA,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}]   skip {clamp_label}: {exc}")
            continue
        if front.non_dominated_index.size == 0:
            continue
        clamps.append(ClampResult(
            clamp_label=clamp_label,
            display_label=_pretty_clamp_label(clamp_label, label_map),
            front=front,
            knee=front.knee,
        ))
    print(f"[{label}]   {len(clamps)} clamps produced a knee "
          f"({time.time() - t0:.1f} s)")

    if not clamps:
        raise RuntimeError(f"No feasible clamp for {label}.")

    # Best-clamp selection: knee closest to the ideal (cov = 1, σ = 0) point
    # after normalising σ by the max σ over knees of this tree.
    knees = [c.knee for c in clamps]
    sigma_norm = max(float(k.trunk_max_stress) for k in knees)
    distances = _knee_distance_to_ideal(knees, sigma_norm=sigma_norm)
    best_idx = int(np.argmin(distances))
    best = clamps[best_idx]
    print(f"[{label}] best clamp: {best.display_label} ({best.clamp_label})  "
          f"f*={best.knee.frequency_hz:.2f} Hz, "
          f"A*={best.knee.amplitude:.1f} mm, "
          f"activation={best.knee.detachment_coverage:.2f}, "
          f"σ={best.knee.trunk_max_stress / 1e6:.3f} MPa")

    # Pre-compute fruit-level outcomes on the best clamp's (f, A) grid for the
    # downstream greedy multi-stage sequence. Only the best clamp gets this
    # treatment (saving 8× the cost of evaluating every candidate clamp).
    print(f"[{label}] tabulating fruit outcomes on best-clamp (f, A) grid …")
    t0 = time.time()
    best.grid_outcomes = _build_best_clamp_grid(
        model, theta, best, f_grid, A_grid,
    )
    print(f"[{label}]   grid outcomes built ({time.time() - t0:.1f} s, "
          f"{len(best.grid_outcomes)} entries)")

    return TreeResult(
        label=label,
        n_fruits=len(model.fruits),
        freqs=freqs,
        mags=mags,
        f_resonance=f_resonance,
        clamps=clamps,
        best_idx=best_idx,
        model=model,
        theta=theta,
        label_map=label_map,
    )


def _load_or_evaluate(model_path: Path, label: str, *,
                       cache_dir: Path, force_recompute: bool) -> "TreeResult":
    """Load a cached TreeResult, falling back to a fresh FE evaluation.

    The cache file lives at ``<cache_dir>/<label>.pkl``. Re-run the script
    with ``--force`` to invalidate every cached tree, or just delete a
    single ``<label>.pkl`` to recompute one tree.
    """
    import pickle

    cache_file = cache_dir / f"{label}.pkl"
    if cache_file.exists() and not force_recompute:
        with open(cache_file, "rb") as fh:
            result = pickle.load(fh)
        print(f"[{label}] loaded from cache: "
              f"{cache_file.relative_to(REPO)}")
        # Cache-schema migration: older caches don't carry the best-clamp
        # (f, A) outcomes needed by the greedy sequence. Patch in place.
        if not getattr(result.best, "grid_outcomes", None):
            print(f"[{label}]   migrating cache: adding best-clamp grid …")
            f_res = result.f_resonance
            f_grid = [max(0.5, f_res - 2.0),
                      max(0.5, f_res - 1.0),
                      f_res, f_res + 1.0, f_res + 2.0]
            A_grid = list(_AMPLITUDE_GRID_MM)
            t0 = time.time()
            result.best.grid_outcomes = _build_best_clamp_grid(
                result.model, result.theta, result.best, f_grid, A_grid,
            )
            print(f"[{label}]   migrated ({time.time() - t0:.1f} s)")
            with open(cache_file, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result
    result = _evaluate_tree(model_path, label=label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = cache_file.stat().st_size / 1024.0
    print(f"[{label}] cached → {cache_file.relative_to(REPO)} ({size_kb:.0f} KB)")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-clamp Pareto recommendation for 5 sample trees.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cached TreeResults and recompute every tree.",
    )
    parser.add_argument(
        "--only-figures", action="store_true",
        help="Refuse to run any FE — fail loudly if any cache is missing. "
             "Use this when only the figure styling has changed.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory (relative to repo root, or absolute) where figures are "
             "written. Defaults to 'results'. Use 'results_nonlinear' to keep "
             "outputs from the randomized Duffing-link pipeline separate from "
             "the linear-baseline figures.",
    )
    args = parser.parse_args()

    _apply_pub_style()
    output_dir_arg = Path(args.output_dir)
    results_root = output_dir_arg if output_dir_arg.is_absolute() else REPO / output_dir_arg
    out_pareto = results_root / "pareto"
    out_frf = results_root / "frf"
    out_response = results_root / "response"
    out_summary = results_root / "summary"
    cache_dir = REPO / "cache" / f"verify_pareto_{results_root.name}"
    for d in (out_pareto, out_frf, out_response, out_summary):
        d.mkdir(parents=True, exist_ok=True)
    results_label = results_root.name

    results: list[TreeResult] = []
    for n in (1, 2, 3, 4, 5):
        model_path = REPO / "trees" / f"tree_{n}.json"
        if not model_path.exists():
            print(f"[skip] {model_path} not found")
            continue
        label = f"tree_{n}"
        if args.only_figures and not (cache_dir / f"{label}.pkl").exists():
            print(f"[error] --only-figures requested but cache missing for "
                  f"{label}. Run without --only-figures first.")
            return 1
        result = _load_or_evaluate(
            model_path, label,
            cache_dir=cache_dir,
            force_recompute=args.force,
        )
        results.append(result)

        _save_pareto_multi_clamp(result, out_pareto / f"pareto_tree_{n}")
        _save_frf(result, out_frf / f"frf_tree_{n}")

        outcomes = _compute_fruit_outcomes_at_best(result)
        _save_tree_response_map(
            result, outcomes,
            out_response / f"response_tree_{n}",
        )
        print(f"[tree_{n}] figures → "
              f"{results_label}/pareto/pareto_tree_{n}.{{png,pdf}} + "
              f"{results_label}/frf/frf_tree_{n}.{{png,pdf}} + "
              f"{results_label}/response/response_tree_{n}.{{png,pdf}}")

    # Strategy A: greedy multi-stage sequence on each tree's best clamp.
    out_sequence = results_root / "sequence"
    out_sequence.mkdir(parents=True, exist_ok=True)
    results_with_stages: list[tuple[TreeResult, list[dict]]] = []
    for n, result in zip((1, 2, 3, 4, 5), results):
        stages = _greedy_sequence(result, target=0.95, max_stages=5)
        results_with_stages.append((result, stages))
        _save_sequence_panel(
            result, stages,
            out_sequence / f"sequence_tree_{n}",
        )
        if stages:
            print(f"[tree_{n}] sequence: {len(stages)} stages → "
                  f"{results_label}/sequence/sequence_tree_{n}.{{png,pdf}}")

    if len(results) >= 2:
        _save_all_pareto_overlay(results, out_pareto / "pareto_all_trees")
        _save_all_frf_overlay(results, out_frf / "frf_all_trees")
        _save_knees_summary(results, out_summary / "summary_knees")
        _save_sequence_coverage(
            results_with_stages, out_summary / "summary_sequence_coverage",
        )
        print(f"\n[summary] cross-tree figures → "
              f"{results_label}/pareto/pareto_all_trees.{{png,pdf}} + "
              f"{results_label}/frf/frf_all_trees.{{png,pdf}} + "
              f"{results_label}/summary/summary_knees.{{png,pdf}} + "
              f"{results_label}/summary/summary_sequence_coverage.{{png,pdf}}")

    _print_recommendation_table(results)
    _print_sequence_table(results_with_stages)
    print(f"\n[done] processed {len(results)} tree(s).")
    return 0


# ────────────────────────────────────────────────────────────────────────────
#  Dense fruit distribution along every branch
# ────────────────────────────────────────────────────────────────────────────
def _generate_linear_fruits(model, policy, spacing: float = 0.05):
    """Generate fruits at every ``spacing`` fraction of every non-trunk branch.

    Mass per fruit is drawn from the policy mean ± Gaussian residual
    (``mass_residual_cv``). The attachment stiffness varies **linearly along
    each branch**: ``k(s) = k_mean × (1.5 − s)`` (1.5×k at the root, 0.5×k at
    the tip), giving the physical "young tip wood snaps first" behaviour.

    A trunk branch is excluded — fruit are borne on scaffolds and shoots,
    not on the trunk itself.
    """
    import random
    from orchard_fem.domain.entities import FruitAttachment

    rng = random.Random(policy.seed)

    # Derive per-fruit mean parameters from the policy aggregates.
    # ``mean_detachment_force_N`` is a derived quantity exposed by the policy
    # generator; if absent, fall back to a default fruit mass of 0.05 kg.
    mean_mass = getattr(policy, "mean_fruit_mass_kg", None)
    if mean_mass is None:
        # Derive from total weight if explicit value missing.
        mean_mass = 0.05
    mean_detach_force = (
        getattr(policy, "mean_detachment_force_N", None) or 5.0
    )
    d_detach = float(policy.detachment_displacement_m)
    k_mean = mean_detach_force / d_detach
    zeta = float(getattr(policy, "attachment_damping_ratio", 0.05))
    target_component = str(
        getattr(policy, "attachment_component", "uz")
    )

    fruits: list = []
    fruit_idx = 0
    n_per_branch = max(int(round(1.0 / spacing)), 1)

    for branch in model.branches:
        if branch.branch_id == "trunk":
            continue
        for i in range(n_per_branch):
            s = (i + 1) * spacing                  # 0.05, 0.10, …, 1.00
            if s > 1.0:
                break
            mass_jitter = rng.gauss(0.0, float(getattr(
                policy, "mass_residual_cv", 0.10,
            )))
            mass = max(mean_mass * (1.0 + mass_jitter), 0.005)
            k = max(k_mean * (1.5 - s), 0.10 * k_mean)
            damping = 2.0 * zeta * math.sqrt(mass * k)
            fruits.append(FruitAttachment(
                fruit_id=f"fr_{branch.branch_id}_{int(s*100):03d}",
                branch_id=branch.branch_id,
                location_s=float(s),
                mass=float(mass),
                stiffness=float(k),
                damping=float(damping),
                target_component=target_component,
            ))
            fruit_idx += 1
    return fruits


# ────────────────────────────────────────────────────────────────────────────
#  Tree response map — per-fruit detachment at the best (clamp, f, A)
# ────────────────────────────────────────────────────────────────────────────
def _compute_fruit_outcomes(
    model, theta: dict, clamp_label: str,
    frequency_hz: float, amplitude_mm: float,
) -> list[dict]:
    """Re-solve the FE problem at the given (clamp, f, A) and return a list of
    per-fruit dicts with detachment status and 3D position."""
    from orchard_fem.calibration.fenicsx_bridge import (
        _apply_theta_to_model, _parse_clamp_label,
    )
    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    from orchard_fem.topology import ObservationPoint

    cloned = _apply_theta_to_model(model, theta)
    branch_id, target_s = _parse_clamp_label(clamp_label)
    cloned = replace(
        cloned,
        excitation=replace(
            cloned.excitation,
            target_branch_id=branch_id,
            target_s=(target_s if target_s is not None
                      else cloned.excitation.target_s),
            amplitude=float(amplitude_mm) * 1.0e-3,
            driving_frequency_hz=float(frequency_hz),
        ),
        analysis=replace(
            cloned.analysis,
            frequency_start_hz=float(frequency_hz),
            frequency_end_hz=float(frequency_hz) + 1.0e-6,
            frequency_steps=1,
        ),
    )

    extras: list[ObservationPoint] = []
    fruit_keys: list[tuple[str, object]] = []
    for fruit in cloned.fruits:
        oid = f"__viz_fruit_{fruit.fruit_id}"
        extras.append(ObservationPoint(
            observation_id=oid,
            target_type="fruit",
            target_id=fruit.fruit_id,
            target_node="tip",
            target_components=[fruit.target_component],
        ))
        fruit_keys.append((oid, fruit))
    cloned = replace(cloned, observations=list(cloned.observations) + extras)

    exp = solve_embedded_beam_frequency_response_experiment(
        cloned, polynomial_degree=1,
    )
    point = exp.result.points[0]
    name_to_idx = {n: i for i, n in enumerate(exp.result.observation_names)}

    d_detach = 0.010
    if cloned.fruit_policy is not None:
        d_detach = float(cloned.fruit_policy.detachment_displacement_m)

    omega = 2.0 * np.pi * float(frequency_hz)
    outcomes: list[dict] = []
    for obs_id, fruit in fruit_keys:
        idx = name_to_idx.get(obs_id)
        if idx is None:
            continue
        u_mag = float(point.observation_magnitudes[idx])
        inertia = fruit.mass * omega * omega * u_mag
        detach_force = fruit.stiffness * d_detach
        branch = next(b for b in cloned.branches if b.branch_id == fruit.branch_id)
        pos = branch.path.point_at(float(fruit.location_s))
        outcomes.append({
            "fruit_id": fruit.fruit_id,
            "branch_id": fruit.branch_id,
            "x": float(pos.x),
            "y": float(pos.y),
            "z": float(pos.z),
            "detached": bool(inertia >= detach_force),
            "inertia_N": float(inertia),
            "detach_force_N": float(detach_force),
        })
    return outcomes


def _compute_fruit_outcomes_at_best(result: TreeResult) -> list[dict]:
    """Re-solve at the best recommended (clamp, f, A) — thin wrapper."""
    k = result.best.knee
    return _compute_fruit_outcomes(
        result.model, result.theta, result.best.clamp_label,
        k.frequency_hz, k.amplitude,
    )


def _build_best_clamp_grid(
    model, theta: dict, best: ClampResult,
    f_grid: list[float], A_grid: list[float],
) -> dict:
    """Tabulate fruit-level outcomes on every (f, A) cell of the best clamp.

    Returns ``{(f, A): {"detached_branches": set[str], "n_detached": int,
    "sigma_mpa": float}}`` — compact enough to ship inside the cached
    ``TreeResult`` (~6 KB total for 5 trees).
    """
    front = best.front
    objectives = front.objectives
    freqs_all = front.frequencies_hz
    amps_all = front.amplitudes
    # Build (f, A) → σ lookup from the Pareto candidate cloud.
    sigma_at = {}
    for i in range(objectives.shape[0]):
        key = (float(freqs_all[i]), float(amps_all[i]))
        sigma_at[key] = float(objectives[i, 1]) / 1.0e6  # to MPa

    grid: dict = {}
    for f in f_grid:
        for A in A_grid:
            outcomes_list = _compute_fruit_outcomes(
                model, theta, best.clamp_label, f, A,
            )
            grid[(float(f), float(A))] = {
                "detached_branches": {
                    o["branch_id"] for o in outcomes_list if o["detached"]
                },
                "n_detached": sum(1 for o in outcomes_list if o["detached"]),
                "n_total_fruits": len(outcomes_list),
                "sigma_mpa": sigma_at.get((float(f), float(A)), float("nan")),
            }
    return grid


# ────────────────────────────────────────────────────────────────────────────
#  Strategy A — Greedy multi-stage sequence
# ────────────────────────────────────────────────────────────────────────────
def _greedy_sequence(
    result: TreeResult, *, target: float = 0.95, max_stages: int = 5,
) -> list[dict]:
    """Build a sequence of (f, A) work points on the *best* clamp.

    At each stage we pick the (f, A) cell of the pre-tabulated grid that
    maximises ``|new branches| / σ`` (new branches per unit of trunk stress
    paid). Stop when the cumulative branch activation reaches ``target``,
    when no further (f, A) introduces new branches, or after ``max_stages``.
    """
    grid = result.best.grid_outcomes
    if not grid:
        return []

    fruit_branches = {f.branch_id for f in result.model.fruits}
    n_total = max(len(fruit_branches), 1)

    activated: set[str] = set()
    stages: list[dict] = []
    for stage in range(max_stages):
        best_score = 0.0
        best_choice = None
        for (f, A), info in grid.items():
            new = info["detached_branches"] - activated
            sigma = info["sigma_mpa"]
            if not new or not (sigma > 0):
                continue
            score = len(new) / sigma                # new branches per MPa
            if score > best_score:
                best_score = score
                best_choice = (f, A, new, sigma, info)
        if best_choice is None:
            break
        f, A, new, sigma, info = best_choice
        activated |= new
        coverage = len(activated) / n_total
        stages.append({
            "stage": stage + 1,
            "f_hz": f,
            "A_mm": A,
            "new_branches": sorted(new),
            "n_new_branches": len(new),
            "cumulative_branches": sorted(activated),
            "coverage": coverage,
            "sigma_mpa": sigma,
            "n_detached_fruits": info["n_detached"],
        })
        if coverage >= target:
            break
    return stages


def _branch_polyline_xz(branch, n: int = 30):
    """Return (xs, zs) polyline for *branch* — side view (x-z) projection."""
    ss = np.linspace(0.0, 1.0, n)
    pts = [branch.path.point_at(float(s)) for s in ss]
    xs = np.array([p.x for p in pts])
    zs = np.array([p.z for p in pts])
    return xs, zs


def _draw_tree_response(
    ax, model, outcomes, best_clamp_label, label_map,
    *, show_branch_labels: bool = True, fontsize: float = 9,
):
    """Render a single side-view tree-response panel onto *ax*."""
    from orchard_fem.calibration.fenicsx_bridge import _parse_clamp_label

    activated = {o["branch_id"] for o in outcomes if o["detached"]}
    fruit_branches = {o["branch_id"] for o in outcomes}

    for branch in model.branches:
        xs, zs = _branch_polyline_xz(branch)
        if branch.branch_id == "trunk":
            color, lw = "#444444", 2.6
        elif branch.branch_id in activated:
            color, lw = "#1B7837", 2.2
        elif branch.branch_id in fruit_branches:
            color, lw = "#E08214", 1.6
        else:
            color, lw = "#bbbbbb", 1.0
        ax.plot(xs, zs, color=color, linewidth=lw,
                solid_capstyle="round", zorder=2)

        if show_branch_labels:
            hier = label_map.get(branch.branch_id, "")
            if hier == "T":
                continue
            if "." not in hier and hier:  # primary branches only
                ax.text(xs[-1], zs[-1] + 0.04, f"B{hier}",
                        color="#222222", fontsize=fontsize,
                        ha="center", va="bottom", fontweight="bold")

    for o in outcomes:
        if o["detached"]:
            ax.scatter([o["x"]], [o["z"]],
                       color="#1B7837", s=55, marker="o",
                       edgecolors="white", linewidths=0.8, zorder=4)
        else:
            ax.scatter([o["x"]], [o["z"]],
                       color="#cccccc", s=26, marker="o",
                       edgecolors="#888888", linewidths=0.5, zorder=3)

    bid, s_clamp = _parse_clamp_label(best_clamp_label)
    if s_clamp is None:
        s_clamp = 0.5
    branch = next(b for b in model.branches if b.branch_id == bid)
    pos = branch.path.point_at(float(s_clamp))
    ax.scatter([pos.x], [pos.z],
               color="#B2182B", s=320, marker="*",
               edgecolors="white", linewidths=1.4, zorder=10)


def _save_tree_response_map(
    result: TreeResult, outcomes: list[dict], stem: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    _draw_tree_response(ax, result.model, outcomes,
                        result.best.clamp_label, result.label_map)

    n_total = len(outcomes)
    n_det = sum(1 for o in outcomes if o["detached"])
    fruit_branches = {o["branch_id"] for o in outcomes}
    activated = {o["branch_id"] for o in outcomes if o["detached"]}

    # Title: two lines, no percentages.
    ax.set_title(
        f"{result.label}\n"
        f"{n_det}/{n_total} fruits detached  •  "
        f"{len(activated)}/{len(fruit_branches)} branches activated",
        fontsize=15,
    )

    handles = [
        Line2D([], [], color="#1B7837", linewidth=2.2, label="Activated branch"),
        Line2D([], [], color="#E08214", linewidth=1.6,
               label="Not activated branch"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#1B7837",
               markeredgecolor="white", markersize=10, label="Fruit detached"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#cccccc",
               markeredgecolor="#888888", markersize=8, label="Fruit retained"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#B2182B",
               markeredgecolor="white", markersize=15,
               label=f"Clamp ({result.best.display_label})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=12, framealpha=0.95)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Working-parameter box in the lower-left, matching the Pareto figure style.
    k = result.best.knee
    ax.text(
        0.02, 0.02,
        f"$f^* = {k.frequency_hz:.1f}$ Hz\n"
        f"$A^* = {k.amplitude:.1f}$ mm\n"
        f"clamp: {result.best.display_label}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=13, color="#B2182B",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#B2182B", lw=0.9, alpha=0.95),
        zorder=10,
    )

    fig.tight_layout(pad=0.4)
    _save_both(fig, stem)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
#  Figures (per-tree, multi-clamp)
# ────────────────────────────────────────────────────────────────────────────
_CLAMP_PALETTE = [
    "#2166AC", "#1B7837", "#762A83", "#E08214",
    "#5AAE61", "#9970AB", "#FDB863", "#80CDC1",
    "#D6604D", "#4393C3",
]


def _save_pareto_multi_clamp(result: TreeResult, stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    best = result.best
    best_sigma_max = max(c.knee.trunk_max_stress / 1e6 for c in result.clamps)
    best_cov_max = max(c.knee.detachment_coverage for c in result.clamps)

    sigma_ceiling_mpa = _STRESS_SANITY_CEILING_PA / 1.0e6

    sigma_lo = float("inf")
    sigma_hi = 0.0
    for i, c in enumerate(result.clamps):
        front = c.front
        cov = -front.objectives[:, 0]
        sigma = front.objectives[:, 1] / 1.0e6
        # Display-layer guard: drop physically-impossible points (e.g.
        # numerical artefacts from anti-resonance / near-singular solves).
        sane_mask = sigma <= sigma_ceiling_mpa
        nd = front.non_dominated_index
        nd_sane = np.array([k for k in nd if sane_mask[k]], dtype=int)
        if nd_sane.size == 0:
            continue
        order = np.argsort(cov[nd_sane])
        color = _CLAMP_PALETTE[i % len(_CLAMP_PALETTE)]
        is_best = c is best

        ax.plot(cov[nd_sane][order], sigma[nd_sane][order],
                color=color, linewidth=1.6 if is_best else 1.1,
                alpha=0.85 if is_best else 0.55, zorder=3 if is_best else 2)
        ax.scatter(cov[nd_sane], sigma[nd_sane],
                   color=color,
                   s=70 if is_best else 42,
                   edgecolors="white", linewidths=0.9,
                   label=f"{c.display_label} (best)" if is_best
                         else c.display_label,
                   zorder=4 if is_best else 3)

        pos = sigma[nd_sane][sigma[nd_sane] > 0]
        if pos.size:
            sigma_lo = min(sigma_lo, float(pos.min()))
        sigma_hi = max(sigma_hi, float(sigma[nd_sane].max()))

    # Mark the best knee
    bk = best.knee
    bx = bk.detachment_coverage
    by = bk.trunk_max_stress / 1.0e6
    ax.scatter([bx], [by], s=380, marker="o",
               facecolor="none", edgecolor="#B2182B", linewidth=2.6,
               zorder=8)

    ax.set_yscale("log")
    ax.set_xlabel("Branch activation")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa, log]")
    ax.set_title(
        f"Multi-clamp Pareto — {result.label} "
        f"({result.n_fruits} fruits, resonance {result.f_resonance:.1f} Hz)"
    )
    ax.set_xlim(0.0, 1.0)
    if np.isfinite(sigma_lo) and sigma_lo > 0:
        ax.set_ylim(sigma_lo * 0.5, sigma_hi * 1.5)

    # Per-tree legend/annotation placement (override default lower-right when
    # the points crowd that corner).
    legend_loc = _PARETO_LEGEND_LOC.get(result.label, "lower right")
    ax.legend(loc=legend_loc, fontsize=12,
              ncol=2 if len(result.clamps) > 5 else 1)

    # Knee annotation: by default left-top; if legend already there, drop it
    # below the legend (still left side).
    if legend_loc == "upper left":
        ann_xy = (0.02, 0.62)
    else:
        ann_xy = (0.02, 0.98)
    ax.text(
        ann_xy[0], ann_xy[1],
        f"$f^* = {bk.frequency_hz:.1f}$ Hz\n"
        f"$A^* = {bk.amplitude:.1f}$ mm\n"
        f"branch activation $= {bk.detachment_coverage:.2f}$\n"
        f"$\\sigma_{{\\max}} = {by:.2f}$ MPa\n"
        f"clamp: {best.display_label}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, color="#B2182B",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#B2182B", lw=0.9, alpha=0.95),
        zorder=10,
    )

    fig.tight_layout(pad=0.4)
    _save_both(fig, stem)
    plt.close(fig)


def _save_frf(result: TreeResult, stem: Path) -> None:
    import matplotlib.pyplot as plt

    freqs, mags = result.freqs, result.mags
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    ax.axvspan(
        _FEASIBLE_BAND_HZ[0], _FEASIBLE_BAND_HZ[1],
        facecolor="#cfe0ee", alpha=0.35, zorder=0,
        label=f"feasible band ({_FEASIBLE_BAND_HZ[0]:.0f}–"
              f"{_FEASIBLE_BAND_HZ[1]:.0f} Hz)",
    )

    ax.semilogy(
        freqs, mags * 1.0e3,
        color="#2166AC", linewidth=1.4,
        marker="o", markersize=3.2,
        markerfacecolor="#2166AC", markeredgecolor="white", markeredgewidth=0.5,
        zorder=2,
    )

    ax.axvline(result.f_resonance, color="#B2182B",
               linestyle="--", linewidth=1.0, alpha=0.85, zorder=3)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title(f"FRF sweep — {result.label} (F = 10 N at trunk mid)")
    ax.set_xlim(0.0, freqs.max())
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")
    ax.legend(loc="upper right", fontsize=10)

    y_min, y_max = ax.get_ylim()
    label_y = y_min * (y_max / y_min) ** 0.06
    ax.text(
        result.f_resonance + 0.4, label_y,
        f"resonance ≈ {result.f_resonance:.1f} Hz",
        color="#B2182B", fontsize=11,
        ha="left", va="bottom", zorder=5,
    )

    fig.tight_layout(pad=0.4)
    _save_both(fig, stem)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
#  Figures (cross-tree comparison)
# ────────────────────────────────────────────────────────────────────────────
_TREE_PALETTE = [
    "#2166AC",  # tree_1 blue
    "#1B7837",  # tree_2 green
    "#B2182B",  # tree_3 red
    "#762A83",  # tree_4 purple
    "#E08214",  # tree_5 orange
]


def _save_all_pareto_overlay(results: list[TreeResult], stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    sigma_ceiling_mpa = _STRESS_SANITY_CEILING_PA / 1.0e6
    cov_max_all = 0.0
    for i, r in enumerate(results):
        front = r.best.front
        cov = -front.objectives[:, 0]
        sigma = front.objectives[:, 1] / 1.0e6
        nd = front.non_dominated_index
        if nd.size == 0:
            continue
        sane_mask = sigma <= sigma_ceiling_mpa
        nd_sane = np.array([k for k in nd if sane_mask[k]], dtype=int)
        if nd_sane.size == 0:
            continue
        order = np.argsort(cov[nd_sane])
        color = _TREE_PALETTE[i % len(_TREE_PALETTE)]

        ax.plot(cov[nd_sane][order], sigma[nd_sane][order],
                color=color, linewidth=1.4, alpha=0.65, zorder=2)
        ax.scatter(cov[nd_sane], sigma[nd_sane],
                   color=color, s=68, edgecolors="white", linewidths=0.9,
                   label=f"{r.label} ({r.best.display_label}, "
                         f"resonance {r.f_resonance:.1f} Hz)", zorder=3)
        k = nd[front.knee_index]
        if sane_mask[k]:
            ax.scatter([cov[k]], [sigma[k]],
                       s=230, marker="o",
                       facecolor="none", edgecolor=color, linewidth=2.0,
                       zorder=5)
        cov_max_all = max(cov_max_all, float(cov[nd_sane].max()))

    ax.set_yscale("log")
    ax.set_xlabel("Branch activation")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa, log]")
    ax.set_title("Best-clamp Pareto fronts — 5 trees")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")

    fig.tight_layout(pad=0.4)
    _save_both(fig, stem)
    plt.close(fig)


def _save_all_frf_overlay(results: list[TreeResult], stem: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axvspan(
        _FEASIBLE_BAND_HZ[0], _FEASIBLE_BAND_HZ[1],
        facecolor="#cfe0ee", alpha=0.30, zorder=0,
        label=f"feasible band ({_FEASIBLE_BAND_HZ[0]:.0f}–"
              f"{_FEASIBLE_BAND_HZ[1]:.0f} Hz)",
    )
    for i, r in enumerate(results):
        color = _TREE_PALETTE[i % len(_TREE_PALETTE)]
        ax.semilogy(r.freqs, r.mags * 1.0e3,
                    color=color, linewidth=1.3, alpha=0.85,
                    label=f"{r.label} (resonance {r.f_resonance:.1f} Hz)",
                    zorder=2)
        in_band = (r.freqs >= _FEASIBLE_BAND_HZ[0]) & \
                  (r.freqs <= _FEASIBLE_BAND_HZ[1])
        ax.scatter([r.f_resonance], [r.mags[r.freqs.tolist().index(r.f_resonance)] * 1e3],
                   color=color, s=70, edgecolors="white", linewidths=0.8,
                   zorder=4)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title("FRF sweeps across 5 trees — markers at in-band resonance")
    ax.set_xlim(0.0, max(r.freqs.max() for r in results))
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(pad=0.4)
    _save_both(fig, stem)
    plt.close(fig)


def _save_sequence_coverage(
    results_with_stages: list[tuple[TreeResult, list[dict]]], stem: Path,
) -> None:
    """Plot cumulative branch activation vs stage index for every tree."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for i, (r, stages) in enumerate(results_with_stages):
        color = _TREE_PALETTE[i % len(_TREE_PALETTE)]
        # Prepend (stage 0, coverage 0) so the curve starts at the origin.
        xs = [0] + [s["stage"] for s in stages]
        ys = [0.0] + [s["coverage"] for s in stages]
        ax.plot(xs, ys, "-o", color=color, lw=1.6,
                markersize=8, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.7,
                label=f"{r.label}")
    ax.axhline(0.95, color="#B2182B", linestyle="--", lw=1.2, alpha=0.85)
    ax.text(0.02, 0.96, "95% target", transform=ax.transAxes,
            color="#B2182B", fontsize=12, va="top", ha="left",
            family="serif")
    ax.set_xlabel("Sequential stage index")
    ax.set_ylabel("Cumulative branch activation")
    ax.set_title("Greedy multi-stage activation coverage")
    ax.set_xlim(0, max((len(s) for _, s in results_with_stages), default=1))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    _save_both(fig, stem)
    plt.close(fig)


_STAGE_PALETTE = ["#3F60A0", "#1B7837", "#E08214", "#762A83", "#A04D6A"]


def _save_sequence_panel(
    result: TreeResult, stages: list[dict], stem: Path,
) -> None:
    """Side-view sequence panel: every branch is coloured by the stage that
    *first* activated it; that colour persists in every later subplot so the
    rightmost subplot shows the cumulative "filled" tree. Each subplot is
    titled just "Stage k"; the working parameters live in a red box at the
    lower-left, mirroring the per-tree response figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not stages:
        return

    # Map every branch to the stage (0-indexed) that first activated it.
    branch_first_stage: dict[str, int] = {}
    for idx, stage in enumerate(stages):
        for b in stage["new_branches"]:
            branch_first_stage.setdefault(b, idx)

    # Clamp position (same across all stages)
    from orchard_fem.calibration.fenicsx_bridge import _parse_clamp_label
    bid, s_clamp = _parse_clamp_label(result.best.clamp_label)
    if s_clamp is None:
        s_clamp = 0.5
    clamp_branch = next(b for b in result.model.branches if b.branch_id == bid)
    clamp_pos = clamp_branch.path.point_at(float(s_clamp))

    import math
    from matplotlib.gridspec import GridSpec

    n = len(stages)
    per_cell_w, per_cell_h = 4.6, 4.6

    # Layout choices:
    #   1–3 stages → 1×n
    #   4 stages   → 2×2
    #   5 stages   → row 1 has 3 panels, row 2 has 2 panels centred
    #   ≥6 stages  → 3 columns, ceil(n/3) rows
    if n <= 3:
        nrows, ncols = 1, n
        use_centred_last_row = False
    elif n == 4:
        nrows, ncols = 2, 2
        use_centred_last_row = False
    elif n == 5:
        nrows, ncols = 2, 3
        use_centred_last_row = True
    else:
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        use_centred_last_row = (n % ncols) != 0

    # Limit the legend to at most ~2-panel widths. Allow up to 4 columns
    # per legend row; total handles = 3 + n (Trunk + n stages + Not act. + Clamp).
    n_handles = 3 + n
    legend_ncol = min(4, n_handles)
    legend_rows = math.ceil(n_handles / legend_ncol)
    legend_height_in = 0.45 + 0.40 * legend_rows  # inches

    fig_w = per_cell_w * ncols
    fig_h = per_cell_h * nrows + legend_height_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Build axes via GridSpec — supports centred last row.
    if use_centred_last_row:
        # Double the grid columns so an offset of 1 cell ≈ half-panel; then
        # place last-row panels starting at column = (2*ncols - 2*n_last)/2
        col_units = ncols * 2
        n_last = n - ncols * (nrows - 1)
        last_start = (col_units - n_last * 2) // 2
        gs = GridSpec(
            nrows, col_units, figure=fig,
            wspace=0.18, hspace=0.42,
        )
        axes_list = []
        for r in range(nrows):
            row_count = ncols if r < nrows - 1 else n_last
            row_start = 0 if r < nrows - 1 else last_start
            for c in range(row_count):
                col_off = row_start + c * 2
                ax = fig.add_subplot(gs[r, col_off:col_off + 2])
                axes_list.append(ax)
    else:
        gs = GridSpec(
            nrows, ncols, figure=fig,
            wspace=0.18, hspace=0.42,
        )
        axes_list = [fig.add_subplot(gs[r, c])
                     for r in range(nrows) for c in range(ncols)
                     if r * ncols + c < n]

    # Track which panels start a new row (for ylabel only on them).
    if use_centred_last_row:
        row_start_indices = [0]
        cumulative = ncols
        while cumulative < n:
            row_start_indices.append(cumulative)
            cumulative += ncols
    else:
        row_start_indices = [r * ncols for r in range(nrows)]

    for ax_idx, (ax, stage) in enumerate(zip(axes_list, stages)):
        for branch in result.model.branches:
            xs, zs = _branch_polyline_xz(branch)
            if branch.branch_id == "trunk":
                color, lw = "#444444", 2.6
            elif branch.branch_id in branch_first_stage and \
                 branch_first_stage[branch.branch_id] <= ax_idx:
                stage_color = _STAGE_PALETTE[
                    branch_first_stage[branch.branch_id] % len(_STAGE_PALETTE)
                ]
                color, lw = stage_color, 2.4
            else:
                color, lw = "#cccccc", 1.2
            ax.plot(xs, zs, color=color, linewidth=lw,
                    solid_capstyle="round", zorder=2)

        ax.scatter([clamp_pos.x], [clamp_pos.z],
                   color="#B2182B", s=260, marker="*",
                   edgecolors="white", linewidths=1.3, zorder=6)

        ax.set_title(f"Stage {stage['stage']}", fontsize=16, fontweight="bold")
        ax.set_xlabel("x [m]")
        if ax_idx in row_start_indices:
            ax.set_ylabel("z [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02, 0.02,
            f"$f = {stage['f_hz']:.1f}$ Hz\n"
            f"$A = {stage['A_mm']:.0f}$ mm\n"
            f"+{stage['n_new_branches']} branches\n"
            f"cum. {stage['coverage'] * 100:.0f}%",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=13, color="#B2182B",
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec="#B2182B", lw=0.9, alpha=0.95),
            zorder=10,
        )

    # Legend — at most 4 columns wide; multi-row if necessary.
    legend_handles = [Line2D([], [], color="#444444", lw=2.6, label="Trunk")]
    for i in range(n):
        legend_handles.append(
            Line2D([], [], color=_STAGE_PALETTE[i % len(_STAGE_PALETTE)],
                   lw=2.4, label=f"Activated at stage {i + 1}")
        )
    legend_handles.append(
        Line2D([], [], color="#cccccc", lw=1.2, label="Not activated")
    )
    legend_handles.append(
        Line2D([], [], marker="*", color="w", markerfacecolor="#B2182B",
               markeredgecolor="white", markersize=15, label="Clamp")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=legend_ncol,
        fontsize=13, framealpha=0.95,
        bbox_to_anchor=(0.5, 0.005),
    )

    # Reserve exactly legend_height_in inches at the bottom.
    gs.tight_layout(
        fig,
        rect=(0.0, legend_height_in / fig_h, 1.0, 1.0),
        pad=0.3, h_pad=1.2, w_pad=0.6,
    )
    _save_both(fig, stem)
    plt.close(fig)


def _save_knees_summary(results: list[TreeResult], stem: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [r.label for r in results]
    f_stars = [r.best.knee.frequency_hz for r in results]
    a_stars = [r.best.knee.amplitude for r in results]
    covs = [r.best.knee.detachment_coverage for r in results]
    sigmas = [r.best.knee.trunk_max_stress / 1.0e6 for r in results]
    clamps = [r.best.display_label for r in results]
    colors = [_TREE_PALETTE[i % len(_TREE_PALETTE)] for i in range(len(results))]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))

    for ax, values, ylabel, title in (
        (axes[0, 0], f_stars, "f* [Hz]", "Knee drive frequency"),
        (axes[0, 1], a_stars, "A* [mm]",  "Knee displacement amplitude"),
        (axes[1, 0], covs,    "Branch activation", "Knee branch activation"),
        (axes[1, 1], sigmas,  r"$\sigma_{\mathrm{max}}$ [MPa]", "Knee trunk stress"),
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.62)
        for bar, val in zip(bars, values):
            txt = f"{val:.2f}" if val < 1.0 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    txt, ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add a footer legend listing the best clamp for each tree.
    legend_lines = "    ".join(
        f"{lbl}: {c}" for lbl, c in zip(labels, clamps)
    )
    fig.text(
        0.5, 0.01,
        f"Best clamps  —  {legend_lines}",
        ha="center", va="bottom",
        fontsize=9.5, color="#444444",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f6f6f6",
                  ec="#cccccc", lw=0.6),
    )

    fig.suptitle("Best-clamp knee recommendations across 5 trees",
                 fontsize=14, y=1.00)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.06, 1.0, 1.0))
    _save_both(fig, stem)
    plt.close(fig)


def _print_recommendation_table(results: list[TreeResult]) -> None:
    if not results:
        return
    print("\n┌────────┬─────────┬─────────────┬─────────┬────────┬────────────┬─────────────┐")
    print("│  Tree  │ Fruits  │ Best clamp  │  f* Hz  │ A* mm  │ Activation │ σ_max [MPa] │")
    print("├────────┼─────────┼─────────────┼─────────┼────────┼────────────┼─────────────┤")
    for r in results:
        b = r.best
        k = b.knee
        print(
            f"│ {r.label:>6} │ {r.n_fruits:>7} │ {b.display_label:<11} │ "
            f"{k.frequency_hz:>7.2f} │ {k.amplitude:>6.0f} │ "
            f"{k.detachment_coverage:>10.2f} │ {k.trunk_max_stress/1e6:>11.3f} │"
        )
    print("└────────┴─────────┴─────────────┴─────────┴────────┴────────────┴─────────────┘")


def _print_sequence_table(
    results_with_stages: list[tuple[TreeResult, list[dict]]],
) -> None:
    if not results_with_stages:
        return
    print("\n  Greedy multi-stage sequence  (single best clamp per tree)")
    print("  ┌────────┬───────┬─────────┬────────┬───────────┬───────────┬──────────┐")
    print("  │  Tree  │ Stage │  f Hz   │ A mm   │ + branch  │ Cum. cov. │ σ [MPa]  │")
    print("  ├────────┼───────┼─────────┼────────┼───────────┼───────────┼──────────┤")
    for r, stages in results_with_stages:
        if not stages:
            print(f"  │ {r.label:>6} │   -   │    -    │   -    │    -      │     -     │    -     │")
            continue
        for i, s in enumerate(stages):
            label = r.label if i == 0 else ""
            print(
                f"  │ {label:>6} │ {s['stage']:>5} │ {s['f_hz']:>7.2f} │ "
                f"{s['A_mm']:>6.1f} │ {s['n_new_branches']:>9d} │ "
                f"{s['coverage']:>9.2f} │ {s['sigma_mpa']:>8.3f} │"
            )
        print("  ├────────┼───────┼─────────┼────────┼───────────┼───────────┼──────────┤")
    print("  └────────┴───────┴─────────┴────────┴───────────┴───────────┴──────────┘")


if __name__ == "__main__":
    raise SystemExit(main())
