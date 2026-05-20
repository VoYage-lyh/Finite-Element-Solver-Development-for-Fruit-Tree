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
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fontsize": 10,
        "legend.edgecolor": "#cccccc",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
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


def main() -> int:
    _apply_pub_style()
    results_root = REPO / "results"
    out_pareto = results_root / "pareto"
    out_frf = results_root / "frf"
    out_response = results_root / "response"
    out_summary = results_root / "summary"
    for d in (out_pareto, out_frf, out_response, out_summary):
        d.mkdir(parents=True, exist_ok=True)

    results: list[TreeResult] = []
    for n in (1, 2, 3, 4, 5):
        model_path = REPO / "trees" / f"tree_{n}.json"
        if not model_path.exists():
            print(f"[skip] {model_path} not found")
            continue
        result = _evaluate_tree(model_path, label=f"tree_{n}")
        results.append(result)

        _save_pareto_multi_clamp(result, out_pareto / f"tree_{n}")
        _save_frf(result, out_frf / f"tree_{n}")

        outcomes = _compute_fruit_outcomes_at_best(result)
        _save_tree_response_map(
            result, outcomes,
            out_response / f"tree_{n}",
        )
        print(f"[tree_{n}] figures → results/{{pareto,frf,response}}/tree_{n}.{{png,pdf}}")

    if len(results) >= 2:
        _save_all_pareto_overlay(results, out_pareto / "all_trees")
        _save_all_frf_overlay(results, out_frf / "all_trees")
        _save_knees_summary(results, out_summary / "knees")
        _save_all_response_panels(results, out_response / "all_trees")
        print(f"\n[summary] cross-tree figures → "
              f"results/{{pareto,frf,response}}/all_trees.{{png,pdf}} + "
              f"results/summary/knees.{{png,pdf}}")

    _print_recommendation_table(results)
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
def _compute_fruit_outcomes_at_best(result: TreeResult) -> list[dict]:
    """Re-solve the FE problem at the recommended work point and return a list
    of per-fruit dicts with detachment status and 3D position.
    """
    from orchard_fem.calibration.fenicsx_bridge import (
        _apply_theta_to_model, _parse_clamp_label,
    )
    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    from orchard_fem.topology import ObservationPoint

    model = result.model
    theta = result.theta
    best = result.best
    knee = best.knee

    cloned = _apply_theta_to_model(model, theta)
    branch_id, target_s = _parse_clamp_label(best.clamp_label)
    cloned = replace(
        cloned,
        excitation=replace(
            cloned.excitation,
            target_branch_id=branch_id,
            target_s=(target_s if target_s is not None
                      else cloned.excitation.target_s),
            amplitude=float(knee.amplitude) * 1.0e-3,  # mm → m (HARMONIC_DISPLACEMENT)
            driving_frequency_hz=float(knee.frequency_hz),
        ),
        analysis=replace(
            cloned.analysis,
            frequency_start_hz=float(knee.frequency_hz),
            frequency_end_hz=float(knee.frequency_hz) + 1.0e-6,
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

    omega = 2.0 * np.pi * float(knee.frequency_hz)
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
    fruit_pct = 100.0 * n_det / max(n_total, 1)
    branch_pct = 100.0 * len(activated) / max(len(fruit_branches), 1)

    k = result.best.knee
    ax.set_title(
        f"{result.label} response — "
        f"$f^*={k.frequency_hz:.1f}$ Hz, $A^*={k.amplitude:.1f}$ mm, "
        f"clamp: {result.best.display_label}\n"
        f"{n_det}/{n_total} fruits detached ({fruit_pct:.0f}%)  •  "
        f"{len(activated)}/{len(fruit_branches)} branches activated "
        f"({branch_pct:.0f}%)",
        fontsize=12,
    )

    handles = [
        Line2D([], [], color="#444444", linewidth=2.6, label="Trunk"),
        Line2D([], [], color="#1B7837", linewidth=2.2, label="Activated branch"),
        Line2D([], [], color="#E08214", linewidth=1.6,
               label="Fruit-bearing, not activated"),
        Line2D([], [], color="#bbbbbb", linewidth=1.0, label="No fruit"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#1B7837",
               markeredgecolor="white", markersize=10, label="Fruit detached"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#cccccc",
               markeredgecolor="#888888", markersize=8, label="Fruit retained"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#B2182B",
               markeredgecolor="white", markersize=15,
               label=f"Clamp ({result.best.display_label})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m] (height)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_both(fig, stem)
    plt.close(fig)


def _save_all_response_panels(
    results: list[TreeResult], stem: Path,
) -> None:
    """5-panel side-by-side tree-response map for all trees."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.4), squeeze=False)
    axes = axes[0]
    for ax, r in zip(axes, results):
        outcomes = _compute_fruit_outcomes_at_best(r)
        _draw_tree_response(ax, r.model, outcomes,
                            r.best.clamp_label, r.label_map,
                            show_branch_labels=False, fontsize=8)
        activated = {o["branch_id"] for o in outcomes if o["detached"]}
        fruit_branches = {o["branch_id"] for o in outcomes}
        n_det = sum(1 for o in outcomes if o["detached"])
        branch_pct = 100.0 * len(activated) / max(len(fruit_branches), 1)
        ax.set_title(
            f"{r.label}\n"
            f"clamp: {r.best.display_label}\n"
            f"{n_det}/{len(outcomes)} fruits, "
            f"{len(activated)}/{len(fruit_branches)} branches "
            f"({branch_pct:.0f}%)",
            fontsize=10,
        )
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("z [m]", fontsize=10)

    legend_handles = [
        Line2D([], [], color="#1B7837", linewidth=2.2, label="Activated branch"),
        Line2D([], [], color="#E08214", linewidth=1.6, label="Fruit, not activated"),
        Line2D([], [], color="#bbbbbb", linewidth=1.0, label="No fruit"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#1B7837",
               markeredgecolor="white", markersize=9, label="Fruit detached"),
        Line2D([], [], marker="o", color="w", markerfacecolor="#cccccc",
               markeredgecolor="#888888", markersize=7, label="Fruit retained"),
        Line2D([], [], marker="*", color="w", markerfacecolor="#B2182B",
               markeredgecolor="white", markersize=14, label="Best clamp"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=6, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Tree response under best (clamp, f, A) — side view",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
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
    ax.text(
        0.02, 0.98,
        f"$f^* = {bk.frequency_hz:.1f}$ Hz\n"
        f"$A^* = {bk.amplitude:.1f}$ mm\n"
        f"branch activation $= {bk.detachment_coverage:.2f}$\n"
        f"$\\sigma_{{\\max}} = {by:.2f}$ MPa\n"
        f"clamp: {best.display_label}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, color="#B2182B",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="#B2182B", lw=0.8, alpha=0.95),
        zorder=10,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Branch activation (fraction of fruit-bearing branches resonating)")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa, log]")
    ax.set_title(
        f"Multi-clamp Pareto — {result.label} "
        f"({result.n_fruits} fruits, resonance {result.f_resonance:.1f} Hz)"
    )
    ax.set_xlim(0.0, 1.0)
    if np.isfinite(sigma_lo) and sigma_lo > 0:
        ax.set_ylim(sigma_lo * 0.5, sigma_hi * 1.5)
    ax.legend(loc="lower right", fontsize=9,
              ncol=2 if len(result.clamps) > 5 else 1)

    fig.tight_layout()
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

    fig.tight_layout()
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
    ax.set_xlabel("Branch activation (fraction of fruit-bearing branches resonating)")
    ax.set_ylabel(r"Trunk peak stress $\sigma_{\mathrm{max}}$  [MPa, log]")
    ax.set_title("Best-clamp Pareto fronts — 5 trees")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax.grid(True, which="minor", linewidth=0.4, color="#ececec")

    fig.tight_layout()
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

    fig.tight_layout()
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
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
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


if __name__ == "__main__":
    raise SystemExit(main())
