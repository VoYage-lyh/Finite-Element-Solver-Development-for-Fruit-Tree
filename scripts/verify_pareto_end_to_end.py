"""End-to-end figures for the 5 sample trees, driven by the PACKAGE pipeline.

This is now a thin figure front-end over the package's harvest pipeline — the
single source of truth — NOT a second, divergent implementation. Per tree it
runs :func:`orchard_fem.workflows.harvest_recommendation.recommend_harvest_parameters`
(P2 elements, modal per-subtree local-mode frequency selection, multi-clamp
Pareto, band-tuned ζ≈6 % damping) and
:func:`~orchard_fem.workflows.harvest_schedule.compute_multiclamp_harvest_schedule`,
then renders the figures from that output. The old standalone logic (mean-FRF
single-peak resonance, single-clamp greedy, P1) was wrong and has been removed.

Outputs default to ``results_nonlinear/`` (``--output-dir`` to change; the old
linear ``results/`` tree is deprecated), organised by figure type:

* ``<out>/pareto/pareto_tree_<n>.{png,pdf}``    — per-clamp Pareto + best knee
* ``<out>/frf/frf_tree_<n>.{png,pdf}``          — FRF sweep, global f₁/f₂ peak markers
* ``<out>/sequence/sequence_tree_<n>.{png,pdf}``— multi-clamp staged sequence
  (per-stage side-view tree response — replaces the dropped standalone response/)
* ``<out>/summary/knees.{png,pdf}`` + ``summary_sequence_coverage.{png,pdf}``
"""
from __future__ import annotations

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
def _find_in_band_peaks(
    freqs: np.ndarray,
    mags: np.ndarray,
    band: tuple[float, float],
    *,
    prominence_ratio: float = 0.10,
    min_separation_hz: float = 2.0,
    max_peaks: int = 4,
) -> list[int]:
    """Return ALL prominent local maxima inside ``band``, sorted by amplitude.

    A local maximum on the log-magnitude trace qualifies if its absolute
    magnitude is at least ``(1 + prominence_ratio)`` × the in-band median.
    Within ``min_separation_hz`` of an already-selected peak no further peak
    is added (otherwise neighbouring grid samples around one mode would
    inflate the count). At most ``max_peaks`` peaks are returned.
    """
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return []
    log_mags = np.log(np.maximum(mags, 1.0e-20))
    is_local_max = np.zeros(mags.size, dtype=bool)
    is_local_max[1:-1] = (
        (log_mags[1:-1] > log_mags[:-2]) & (log_mags[1:-1] > log_mags[2:])
    )
    candidates = is_local_max & in_band
    if not candidates.any():
        return []
    band_median = float(np.median(mags[in_band]))
    cand_idx = np.flatnonzero(candidates)
    prominent = mags[cand_idx] >= band_median * (1.0 + prominence_ratio)
    cand_idx = cand_idx[prominent] if prominent.any() else cand_idx
    cand_idx = cand_idx[np.argsort(-mags[cand_idx])]  # descending magnitude
    selected: list[int] = []
    for idx in cand_idx:
        if any(abs(float(freqs[idx]) - float(freqs[s])) < min_separation_hz
               for s in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= max_peaks:
            break
    return selected


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
    exp = solve_embedded_beam_frequency_response_experiment(swept, polynomial_degree=2)
    res = exp.result
    freqs = np.array([p.frequency_hz for p in res.points])
    name_to_idx = {n: i for i, n in enumerate(res.observation_names)}
    tip_obs = [n for n in res.observation_names if n.endswith("_tip_ux")]
    mags = np.zeros_like(freqs)
    for j, p in enumerate(res.points):
        mags[j] = float(np.mean([p.observation_magnitudes[name_to_idx[n]]
                                  for n in tip_obs]))
    return freqs, mags


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


def _clamp_result_from_rec(cr, label_map: dict[str, str]) -> "ClampResult":
    """Adapt a package ``ClampRecommendation`` into the figure-friendly
    ``ClampResult`` / ``ParetoFront`` used by the savers (no recompute)."""
    from orchard_fem.recommendation.pareto import ParetoFront

    pts = list(cr.points)
    freqs = np.array([p.frequency_hz for p in pts], dtype=float)
    amps = np.array([p.amplitude_mm for p in pts], dtype=float)
    # objectives are the minimisation form the savers read: col0=-coverage, col1=σ[Pa].
    objs = np.array([[-p.coverage, p.trunk_stress_pa, 0.0] for p in pts], dtype=float)
    nd = np.array([i for i, p in enumerate(pts) if p.on_front], dtype=int)
    if nd.size == 0:
        nd = np.array([int(np.argmin(objs[:, 0]))], dtype=int)
    knee_local = next((j for j, i in enumerate(nd) if pts[i].is_knee), 0)
    front = ParetoFront(
        clamp_node=cr.clamp_label, frequencies_hz=freqs, amplitudes=amps,
        objectives=objs, non_dominated_index=nd, knee_index=knee_local,
    )
    return ClampResult(
        clamp_label=cr.clamp_label,
        display_label=_pretty_clamp_label(cr.clamp_label, label_map),
        front=front, knee=front.knee,
    )


def _evaluate_tree(model_path: Path, label: str) -> TreeResult:
    """Run the PACKAGE pipeline (single source of truth) and adapt its output to
    the figure data structures. Replaces the old standalone FRF-sweep + Pareto +
    greedy logic — that pipeline was wrong (P1, mean-FRF single-peak, single
    clamp); the package now does modal per-subtree local modes, P2, realistic
    band-tuned damping, and a multi-clamp schedule.
    """
    from orchard_fem.discretization.damping import (
        rayleigh_from_band_zeta, rayleigh_from_paper_zeta,
    )
    from orchard_fem.domain import ExcitationKind
    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.workflows.harvest_recommendation import (
        RecommendationOptions, generate_linear_fruits, recommend_harvest_parameters,
    )

    print(f"\n[{label}] loading {model_path.name} …")
    model = load_orchard_model(str(model_path))
    opt = RecommendationOptions(duration_s=10.0)   # P2 + modal multi-clamp + measured ζ(f) defaults

    _dmp = (f"flat ζ≈{opt.damping_zeta:.0%}" if opt.damping_zeta is not None
            else "measured ζ(f) 0.35→0.09")
    print(f"[{label}] pipeline (P{opt.polynomial_degree}, {_dmp}, modal "
          f"per-subtree local modes, multi-clamp) …")
    t0 = time.time()
    result = recommend_harvest_parameters(
        model, model_path=str(model_path), options=opt, progress_cb=lambda *_: None,
    )
    print(f"[{label}]   {time.time() - t0:.1f}s  resonance {result.resonance_hz:.2f} Hz, "
          f"{len(result.clamps)} clamps, best #{result.best_clamp_index}")

    # Rebuild the SAME fruited + damped model the pipeline used, for the FRF curve
    # and the per-fruit response map the figures need.
    fmodel = model
    if opt.detachment_displacement_m is not None and fmodel.fruit_policy is not None:
        fmodel = replace(fmodel, fruit_policy=replace(
            fmodel.fruit_policy, detachment_displacement_m=float(opt.detachment_displacement_m)))
    if opt.dense_fruit_spacing is not None:
        fmodel = replace(fmodel, fruits=generate_linear_fruits(
            fmodel, fmodel.fruit_policy, opt.dense_fruit_spacing))
    _dmp_hi = min(opt.band_hz[1], opt.limits.max_freq_hz)
    if opt.damping_zeta is None:
        a, b = rayleigh_from_paper_zeta(opt.band_hz[0], _dmp_hi)
    else:
        a, b = rayleigh_from_band_zeta(opt.damping_zeta, opt.band_hz[0], opt.band_hz[1])
    fmodel = replace(fmodel, analysis=replace(fmodel.analysis, rayleigh_alpha=a, rayleigh_beta=b))

    print(f"[{label}] FRF sweep (figure) …")
    freqs, mags = _coarse_frf_sweep(fmodel, 0.5, 30.0, 60)

    label_map = _build_hierarchical_label_map(model)
    clamps = [_clamp_result_from_rec(cr, label_map)
              for cr in result.clamps if cr.knee is not None]
    if not clamps:
        raise RuntimeError(f"No feasible clamp for {label}.")
    best_label = result.best.clamp_label
    best_idx = next((i for i, c in enumerate(clamps) if c.clamp_label == best_label), 0)

    theta = {"E": float(model.materials[0].youngs_modulus),
             "rho": float(model.materials[0].density)}
    disp_model = replace(fmodel, excitation=replace(
        fmodel.excitation, kind=ExcitationKind.HARMONIC_DISPLACEMENT))
    return TreeResult(
        label=label, n_fruits=len(fmodel.fruits), freqs=freqs, mags=mags,
        f_resonance=float(result.resonance_hz), clamps=clamps, best_idx=best_idx,
        model=disp_model, theta=theta, label_map=label_map,
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
        "--output-dir", default="results_nonlinear",
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
    out_summary = results_root / "summary"
    cache_dir = REPO / "cache" / f"verify_pareto_{results_root.name}"
    for d in (out_pareto, out_frf, out_summary):
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
        # The standalone response/ map is dropped: the single-fixed-parameter
        # harvest it depicted is abandoned. The per-stage tree response now lives
        # only inside the multi-clamp sequence panels below.
        print(f"[tree_{n}] figures → "
              f"{results_label}/pareto/pareto_tree_{n}.{{png,pdf}} + "
              f"{results_label}/frf/frf_tree_{n}.{{png,pdf}}")

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
        cloned, polynomial_degree=2,
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


def _greedy_sequence(
    result: TreeResult, *, target: float = 0.95, max_stages: int = 6,
) -> list[dict]:
    """Multi-clamp staged sequence via the package scheduler (replaces the old
    single-clamp greedy). Builds a branch-outcome grid on the top candidate
    clamps (each on its own local-mode frequencies) and lets the scheduler move
    the grip between energy-reachable regions; returns figure-friendly dicts.
    """
    from orchard_fem.actuator.harvest_bridge import DS5L1Limits
    from orchard_fem.workflows.harvest_schedule import (
        build_branch_outcome_grid, compute_multiclamp_harvest_schedule,
    )

    n_total = max(len({f.branch_id for f in result.model.fruits}), 1)
    a_grid = list(_AMPLITUDE_GRID_MM)
    # Give the scheduler ample re-clamp options (the rig CAN change grip): take the
    # best-covering clamps across distinct subtrees so later stages can reach
    # branches the first grip could not excite.
    cand = sorted(result.clamps, key=lambda c: c.knee.detachment_coverage, reverse=True)[:6]
    grids = {}
    for c in cand:
        f_for = sorted({float(x) for x in c.front.frequencies_hz.tolist()})
        grids[c.clamp_label] = build_branch_outcome_grid(
            result.model, c.clamp_label, f_for, a_grid,
            theta=result.theta, limits=DS5L1Limits(), polynomial_degree=2,
        )
    sched = compute_multiclamp_harvest_schedule(
        grids, n_fruit_branches=n_total, target_coverage=target, max_stages=max_stages,
    )
    return [
        {
            "stage": st.index,
            "f_hz": st.plan.frequency_hz,
            "A_mm": st.plan.stroke_mm / 2.0,
            "clamp": st.plan.excitation_label,
            "new_branches": list(st.new_branches),
            "n_new_branches": len(st.new_branches),
            "cumulative_branches": [],
            "coverage": st.cumulative_coverage,
            "sigma_mpa": st.trunk_stress_pa / 1.0e6,
            "n_detached_fruits": st.n_detached_fruits,
        }
        for st in sched.stages
    ]

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

    # Mark the prominent natural frequencies by GLOBAL peak search on the FRF
    # curve across the whole feasible band (not just the lowest mode): a tree
    # like tree_3 resonates at BOTH ~3.4 Hz (f₁) and a stronger ~8 Hz (f₂).
    peak_idx = _find_in_band_peaks(freqs, mags, _FEASIBLE_BAND_HZ, max_peaks=3)
    peak_idx = sorted(peak_idx, key=lambda i: float(freqs[i]))  # ascending → f₁, f₂, …
    y_min, y_max = ax.get_ylim()
    label_y = y_min * (y_max / y_min) ** 0.06
    if not peak_idx:  # degenerate FRF: fall back to the reported resonance
        ax.axvline(result.f_resonance, color="#B2182B", linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=3)
        ax.text(result.f_resonance + 0.4, label_y,
                f"resonance ≈ {result.f_resonance:.1f} Hz",
                color="#B2182B", fontsize=11, ha="left", va="bottom", zorder=5)
    for rank, idx in enumerate(peak_idx, start=1):
        f_pk = float(freqs[idx])
        ax.axvline(f_pk, color="#B2182B", linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=3)
        ax.plot([f_pk], [mags[idx] * 1.0e3], marker="v", color="#B2182B",
                markersize=7, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        sub = "₁₂₃₄₅"[rank - 1] if rank <= 5 else str(rank)
        ax.text(f_pk + 0.4, label_y, f"f{sub} ≈ {f_pk:.1f} Hz",
                color="#B2182B", fontsize=10.5, ha="left", va="bottom", zorder=5)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title(f"FRF sweep — {result.label} (F = 10 N at trunk mid)")
    ax.set_xlim(0.0, freqs.max())
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")
    ax.legend(loc="upper right", fontsize=10)

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
        # Primary + at most one well-separated secondary peak (≥ 5 Hz away)
        # so only the genuinely different modes are marked, not the sidebands
        # of the same resonance.
        peak_idxs = _find_in_band_peaks(
            r.freqs, r.mags, _FEASIBLE_BAND_HZ,
            prominence_ratio=0.10, min_separation_hz=5.0, max_peaks=2,
        )
        peak_freqs = [float(r.freqs[k]) for k in peak_idxs]
        if peak_freqs:
            primary_f = peak_freqs[0]
            if len(peak_freqs) >= 2:
                label = (f"{r.label} (primary {primary_f:.1f} Hz, "
                         f"secondary {peak_freqs[1]:.1f} Hz)")
            else:
                label = f"{r.label} (resonance {primary_f:.1f} Hz)"
        else:
            label = f"{r.label} (resonance {r.f_resonance:.1f} Hz)"

        ax.semilogy(r.freqs, r.mags * 1.0e3,
                    color=color, linewidth=1.3, alpha=0.85,
                    label=label, zorder=2)
        # Primary peak: solid filled marker (top-of-mag).
        # Secondary peaks: smaller hollow marker so the eye distinguishes them.
        for rank, k in enumerate(peak_idxs):
            f = float(r.freqs[k])
            m_mm = float(r.mags[k]) * 1e3
            if rank == 0:
                ax.scatter([f], [m_mm], color=color, s=70,
                           edgecolors="white", linewidths=0.8, zorder=4)
            else:
                ax.scatter([f], [m_mm], facecolors="none", edgecolors=color,
                           s=44, linewidths=1.4, zorder=4)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Mean canopy-tip displacement [mm]")
    ax.set_title("FRF sweeps across 5 trees — filled = primary, hollow = secondary peaks")
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
