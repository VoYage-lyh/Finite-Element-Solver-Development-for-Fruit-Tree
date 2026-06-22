"""Headless simulation→working-parameter recommendation pipeline.

One call takes a tree model and returns rig-executable working parameters:

    load model → (densify fruits) → coarse FRF sweep → resonance candidates
    → (f, A) grid **clipped to the DS5L1 envelope** → per-clamp Pareto fronts
    (coverage vs. trunk stress) → knee per clamp → best clamp → WorkingPoint

This ports the validated logic of ``scripts/verify_pareto_end_to_end.py`` into
the package, with three changes for interactive/front-end use:

1. **Rig-envelope hard constraint.**  The e2e study scans displacement
   amplitudes up to 30 mm, but the electric cylinder caps the stroke at
   ``S = 2A ≤ max_stroke_mm`` and reciprocation frequency falls with stroke
   (``1/(2f) = 6S/rpm + C``).  Grid points the rig cannot execute are excluded
   from the Pareto search *up front*, so the recommendation is executable by
   construction.
2. **Progress / cancellation hooks** for a GUI worker thread.
3. **Adjustment trace** (:attr:`RecommendationResult.steps`) — every automatic
   decision (grid clipping, resonance pick, infeasible-point counts, knee
   choice) is recorded as a human-readable step so the front-end can show *how*
   the final parameters were reached.

FEniCSx (dolfinx) is imported lazily inside :func:`recommend_harvest_parameters`;
:func:`summarize_orchard_model` and all dataclasses work without it.  The FE
pieces can be substituted (``frf_sweep`` / ``evaluator_factory``) for testing.

The chosen :class:`WorkingPoint` exports to the same params-JSON consumed by
``scripts/run_harvest_on_rig.py`` and the harvest console GUI
(:mod:`orchard_fem.actuator.harvest_console`).
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from orchard_fem.actuator.harvest_bridge import DS5L1Limits

if TYPE_CHECKING:  # annotations only — numpy/pareto stay lazy so the package
    import numpy as np  # (and the CLI) imports without numpy installed

ProgressCb = Callable[[str, float], None]   # (message, fraction 0..1)
CancelCb = Callable[[], bool]


def _noop_progress(_msg: str, _frac: float) -> None:
    pass


def _never_cancel() -> bool:
    return False


# --------------------------------------------------------------------------- #
# Model summary (no FE backend required)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSummary:
    """Front-end-displayable digest of an :class:`OrchardModel`."""

    name: str
    path: str
    n_branches: int
    n_levels: int
    height_m: float
    n_fruits: int
    fruit_mass_total_kg: float
    fruit_mass_mean_kg: float
    detachment_displacement_m: float | None
    n_materials: int
    material_names: tuple[str, ...]
    clamp_branches: tuple[str, ...]
    excitation_kind: str
    excitation_target: str
    frequency_band_hz: tuple[float, float]
    notes: str = ""

    def lines(self) -> list[str]:
        """Key/value lines for plain-text display."""
        return [
            f"Model:        {self.name}",
            f"File:         {self.path}",
            f"Branches:     {self.n_branches}  ({self.n_levels} levels, "
            f"height {self.height_m:.2f} m)",
            f"Fruits:       {self.n_fruits}, total mass {self.fruit_mass_total_kg:.2f} kg "
            f"(mean {self.fruit_mass_mean_kg * 1000:.0f} g)",
            "Detachment:   "
            + (f"{self.detachment_displacement_m * 1000:.1f} mm"
               if self.detachment_displacement_m is not None else "undefined"),
            f"Materials:    {self.n_materials} ({', '.join(self.material_names)})",
            f"Clamp branch: {', '.join(self.clamp_branches) or 'none'}",
            f"Excitation:   {self.excitation_kind} @ {self.excitation_target}",
            f"Analysis band:{self.frequency_band_hz[0]:g}–{self.frequency_band_hz[1]:g} Hz",
        ] + ([f"Notes:        {self.notes}"] if self.notes else [])


def summarize_orchard_model(model: Any, path: str | Path = "") -> ModelSummary:
    """Build a :class:`ModelSummary` from a loaded model (no FE backend)."""
    masses = [float(f.mass) for f in model.fruits]
    d_detach = (
        float(model.fruit_policy.detachment_displacement_m)
        if model.fruit_policy is not None else None
    )
    levels = {int(b.level) for b in model.branches}
    height = max(
        max(float(b.path.start.z), float(b.path.end.z)) for b in model.branches
    )
    exc = model.excitation
    target = exc.target_branch_id + (
        f"@s={exc.target_s:g}" if exc.target_s is not None else f"@{exc.target_node}"
    )
    return ModelSummary(
        name=str(model.metadata.name),
        path=str(path),
        n_branches=len(model.branches),
        n_levels=len(levels),
        height_m=height,
        n_fruits=len(model.fruits),
        fruit_mass_total_kg=sum(masses),
        fruit_mass_mean_kg=(sum(masses) / len(masses)) if masses else 0.0,
        detachment_displacement_m=d_detach,
        n_materials=len(model.materials),
        material_names=tuple(str(m.material_id) for m in model.materials),
        clamp_branches=tuple(c.branch_id for c in model.clamps),
        excitation_kind=str(getattr(exc.kind, "value", exc.kind)),
        excitation_target=target,
        frequency_band_hz=(
            float(model.analysis.frequency_start_hz),
            float(model.analysis.frequency_end_hz),
        ),
        notes=str(getattr(model.metadata, "notes", "") or ""),
    )


# --------------------------------------------------------------------------- #
# Options / results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecommendationOptions:
    """Tunable knobs of the recommendation pipeline (defaults match the e2e study).

    Parameters
    ----------
    band_hz:
        Feasible mechanical band searched for resonances.
    sweep_steps:
        Frequency steps of the coarse FRF sweep over ``sweep_range_hz``.
    trunk_clamp_s / include_child_clamps / primary_clamp_s:
        Candidate clamp positions: trunk arc-length fractions, plus
        ``primary_clamp_s`` along every primary branch (one joined directly to
        the trunk).  Primary defaults are the **root** (``s=0``), **quarter**
        (``s=0.25``) and **mid** (``s=0.5``) — reachable, low spots near the
        trunk; the far tip (``s=1``) is excluded as unreachable for the clamp.
    clamp_labels:
        Explicit clamp-label override (e.g. a front-end selection); when set,
        the automatic enumeration above is skipped.
    amplitude_grid_mm:
        Displacement-amplitude candidates A [mm] (half peak-to-peak at the
        clamp).  Values beyond the rig envelope (``2A > max_stroke``) are
        dropped with a logged step.
    dense_fruit_spacing:
        When set (e.g. ``0.05``), replace the model fruit list with one fruit
        per *spacing* fraction of every non-trunk branch (the e2e convention).
        ``None`` keeps the model's own fruits.
    detachment_displacement_m:
        Override of the fruit-policy detachment displacement (e2e uses 2 mm).
    coverage_mode:
        ``"branch"`` (default) or ``"fruit"`` — see the Pareto evaluator.
    stress_ceiling_pa:
        Hard sanity ceiling on trunk peak stress.
    duration_s:
        Working duration attached to the exported parameters.
    limits:
        Actuator envelope used for the rig-feasibility constraint.
    """

    band_hz: tuple[float, float] = (3.0, 20.0)
    sweep_range_hz: tuple[float, float] = (0.5, 30.0)
    sweep_steps: int = 60
    trunk_clamp_s: tuple[float, ...] = (0.25, 0.55, 0.85)
    include_child_clamps: bool = True
    primary_clamp_s: tuple[float, ...] = (0.0, 0.25, 0.5)   # root / quarter / mid of each primary branch
    clamp_labels: tuple[str, ...] | None = None
    amplitude_grid_mm: tuple[float, ...] = (2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)
    dense_fruit_spacing: float | None = 0.05
    detachment_displacement_m: float | None = 0.002
    coverage_mode: str = "branch"
    stress_ceiling_pa: float = 100.0e6
    duration_s: float = 10.0
    limits: DS5L1Limits = field(default_factory=DS5L1Limits)
    n_jobs: int = -1            # FE solves across processes; <=0 = all cores, 1 = serial


@dataclass(frozen=True)
class WorkingPoint:
    """One evaluated ``(clamp, f, A)`` candidate."""

    clamp_label: str
    frequency_hz: float
    amplitude_mm: float                  # displacement amplitude (half p-p)
    coverage: float
    trunk_stress_pa: float
    rig_feasible: bool
    on_front: bool = False
    is_knee: bool = False

    @property
    def stroke_mm(self) -> float:
        """Rig stroke = peak-to-peak excursion = 2·A."""
        return 2.0 * self.amplitude_mm

    def to_params_json(self, duration_s: float) -> dict:
        """Dict in the schema of ``scripts/run_harvest_on_rig.py --params-json``."""
        return {
            "frequency_hz": self.frequency_hz,
            "displacement_amplitude_m": self.amplitude_mm / 1000.0,
            "duration_s": duration_s,
            "excitation_label": self.clamp_label,
            "coverage": self.coverage,
            "trunk_stress_pa": self.trunk_stress_pa,
        }


@dataclass(frozen=True)
class ClampRecommendation:
    clamp_label: str
    points: tuple[WorkingPoint, ...]
    knee: WorkingPoint | None


@dataclass(frozen=True)
class RecommendationResult:
    """Full pipeline output: candidates, fronts, knees, and the decision trace."""

    model_path: str
    model_name: str
    resonance_hz: float
    secondary_hz: tuple[float, ...]
    frequency_grid_hz: tuple[float, ...]
    amplitude_grid_mm: tuple[float, ...]
    clamps: tuple[ClampRecommendation, ...]
    best_clamp_index: int
    duration_s: float
    steps: tuple[str, ...]               # adjustment/decision trace
    elapsed_s: float

    @property
    def best(self) -> ClampRecommendation:
        return self.clamps[self.best_clamp_index]

    @property
    def recommended(self) -> WorkingPoint | None:
        return self.best.knee if self.clamps else None

    # -- serialisation (GUI export / offline rig PC import) ----------------- #

    def to_json_dict(self) -> dict:
        d = asdict(self)
        return d

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @staticmethod
    def from_json_dict(d: dict) -> "RecommendationResult":
        clamps = tuple(
            ClampRecommendation(
                clamp_label=c["clamp_label"],
                points=tuple(WorkingPoint(**p) for p in c["points"]),
                knee=WorkingPoint(**c["knee"]) if c.get("knee") else None,
            )
            for c in d["clamps"]
        )
        return RecommendationResult(
            model_path=d["model_path"],
            model_name=d["model_name"],
            resonance_hz=float(d["resonance_hz"]),
            secondary_hz=tuple(float(x) for x in d["secondary_hz"]),
            frequency_grid_hz=tuple(float(x) for x in d["frequency_grid_hz"]),
            amplitude_grid_mm=tuple(float(x) for x in d["amplitude_grid_mm"]),
            clamps=clamps,
            best_clamp_index=int(d["best_clamp_index"]),
            duration_s=float(d["duration_s"]),
            steps=tuple(d["steps"]),
            elapsed_s=float(d["elapsed_s"]),
        )

    @staticmethod
    def load_json(path: str | Path) -> "RecommendationResult":
        return RecommendationResult.from_json_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


# --------------------------------------------------------------------------- #
# Building blocks ported from the e2e study
# --------------------------------------------------------------------------- #


def generate_linear_fruits(model: Any, policy: Any, spacing: float = 0.05) -> list:
    """One fruit per *spacing* fraction of every non-trunk branch.

    Stiffness varies linearly along each branch (1.5×k_mean at the root →
    0.5×k_mean at the tip): young tip wood snaps first.  Port of the e2e
    ``_generate_linear_fruits``.
    """
    from orchard_fem.domain.entities import FruitAttachment

    rng = random.Random(policy.seed)
    mean_mass = getattr(policy, "mean_fruit_mass_kg", None) or 0.05
    mean_detach_force = getattr(policy, "mean_detachment_force_N", None) or 5.0
    d_detach = float(policy.detachment_displacement_m)
    k_mean = mean_detach_force / d_detach
    zeta = float(getattr(policy, "attachment_damping_ratio", 0.05))
    component = str(getattr(policy, "attachment_component", "uz"))

    fruits: list = []
    n_per_branch = max(int(round(1.0 / spacing)), 1)
    for branch in model.branches:
        if branch.branch_id == "trunk":
            continue
        for i in range(n_per_branch):
            s = (i + 1) * spacing
            if s > 1.0:
                break
            mass = max(
                mean_mass * (1.0 + rng.gauss(0.0, float(getattr(policy, "mass_residual_cv", 0.10)))),
                0.005,
            )
            k = max(k_mean * (1.5 - s), 0.10 * k_mean)
            fruits.append(FruitAttachment(
                fruit_id=f"fr_{branch.branch_id}_{int(s * 100):03d}",
                branch_id=branch.branch_id,
                location_s=float(s),
                mass=float(mass),
                stiffness=float(k),
                damping=float(2.0 * zeta * math.sqrt(mass * k)),
                target_component=component,
            ))
    return fruits


def find_in_band_resonance(
    freqs: np.ndarray, mags: np.ndarray, band: tuple[float, float],
    *, prominence_ratio: float = 0.10,
) -> tuple[int, bool]:
    """``(peak_index, has_genuine_local_max)`` inside *band* (e2e port)."""
    import numpy as np

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        raise RuntimeError(f"No FRF samples in {band[0]}–{band[1]} Hz; widen the sweep.")
    log_mags = np.log(np.maximum(mags, 1.0e-20))
    is_local_max = np.zeros(mags.size, dtype=bool)
    is_local_max[1:-1] = (log_mags[1:-1] > log_mags[:-2]) & (log_mags[1:-1] > log_mags[2:])
    candidates = is_local_max & in_band
    band_median = float(np.median(mags[in_band]))
    if candidates.any():
        cand_idx = np.flatnonzero(candidates)
        prominent = mags[cand_idx] >= band_median * (1.0 + prominence_ratio)
        if prominent.any():
            cand_idx = cand_idx[prominent]
        return int(cand_idx[int(np.argmax(mags[cand_idx]))]), True
    curvature = np.zeros_like(mags)
    curvature[1:-1] = log_mags[2:] - 2.0 * log_mags[1:-1] + log_mags[:-2]
    band_idx = np.flatnonzero(in_band)
    interior = band_idx[(band_idx > 0) & (band_idx < mags.size - 1)]
    if interior.size == 0:
        return int(band_idx[int(np.argmax(mags[in_band]))]), False
    return int(interior[int(np.argmin(curvature[interior]))]), False


def find_in_band_peaks(
    freqs: np.ndarray, mags: np.ndarray, band: tuple[float, float],
    *, prominence_ratio: float = 0.10, min_separation_hz: float = 5.0, max_peaks: int = 2,
) -> list[int]:
    """Indices of up to *max_peaks* prominent, well-separated in-band local maxima."""
    import numpy as np

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    log_mags = np.log(np.maximum(mags, 1.0e-20))
    is_local_max = np.zeros(mags.size, dtype=bool)
    is_local_max[1:-1] = (log_mags[1:-1] > log_mags[:-2]) & (log_mags[1:-1] > log_mags[2:])
    cand = np.flatnonzero(is_local_max & in_band)
    if cand.size == 0:
        return []
    band_median = float(np.median(mags[in_band]))
    cand = cand[mags[cand] >= band_median * (1.0 + prominence_ratio)]
    cand = cand[np.argsort(mags[cand])[::-1]]
    picked: list[int] = []
    for idx in cand:
        if all(abs(freqs[idx] - freqs[j]) >= min_separation_hz for j in picked):
            picked.append(int(idx))
        if len(picked) >= max_peaks:
            break
    return picked


def candidate_clamp_labels(model: Any, options: RecommendationOptions) -> list[str]:
    """Clamp candidates: explicit override, else trunk fractions + primary root/mid.

    Primary branches are those joined directly to the trunk; the clamp candidates
    on them are ``options.primary_clamp_s`` (default root ``s=0`` and mid
    ``s=0.5``) — reachable spots near the trunk, not the high far tip.
    """
    if options.clamp_labels is not None:
        return list(options.clamp_labels)
    labels = [f"trunk@{s:.2f}" for s in options.trunk_clamp_s]
    if options.include_child_clamps:
        trunk_node = model.topology.require_node("trunk")
        labels += [f"{bid}@{s:.2f}"
                   for bid in trunk_node.child_branch_ids
                   for s in options.primary_clamp_s]
    return labels


def build_frequency_grid(
    resonance_hz: float,
    secondary_hz: list[float],
    band_hz: tuple[float, float],
) -> list[float]:
    """Primary ±2 Hz cluster + ±1 Hz around each secondary, band-guarded (e2e port)."""
    grid: set[float] = {
        max(0.5, resonance_hz - 2.0), max(0.5, resonance_hz - 1.0),
        resonance_hz, resonance_hz + 1.0, resonance_hz + 2.0,
    }
    for f_sec in secondary_hz:
        for df in (-1.0, 0.0, 1.0):
            grid.add(max(0.5, f_sec + df))
    return sorted(f for f in grid if band_hz[0] - 1.5 <= f <= band_hz[1] + 1.5)


def rig_feasible(frequency_hz: float, amplitude_mm: float, limits: DS5L1Limits) -> bool:
    """Whether the rig can execute ``(f, A)``: stroke 2A within limits and f reachable."""
    stroke = 2.0 * amplitude_mm
    if stroke > limits.max_stroke_mm or frequency_hz > limits.max_freq_hz:
        return False
    return limits.seed_rpm(stroke, frequency_hz) is not None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def _default_frf_sweep(model: Any, f_min: float, f_max: float, steps: int):
    """Mean tip-|H| spectrum via the FEniCSx solver (e2e ``_coarse_frf_sweep``)."""
    import numpy as np

    from orchard_fem.fenicsx.frequency_response import (
        solve_embedded_beam_frequency_response_experiment,
    )
    swept = replace(model, analysis=replace(
        model.analysis,
        frequency_start_hz=float(f_min), frequency_end_hz=float(f_max),
        frequency_steps=int(steps),
    ))
    res = solve_embedded_beam_frequency_response_experiment(swept, polynomial_degree=1).result
    freqs = np.array([p.frequency_hz for p in res.points])
    name_to_idx = {n: i for i, n in enumerate(res.observation_names)}
    tip_obs = [n for n in res.observation_names if n.endswith("_tip_ux")]
    mags = np.zeros_like(freqs)
    for j, p in enumerate(res.points):
        mags[j] = float(np.mean([p.observation_magnitudes[name_to_idx[n]] for n in tip_obs]))
    return freqs, mags


def _default_outcome_solver_factory(model: Any, options: RecommendationOptions):
    """Displacement-excitation outcome solver (e2e convention).

    Returns ``solve(params, f, A, clamp) -> HarvestOutcome`` — one FE solve per
    work point yielding coverage + trunk stress + per-branch detachment.
    """
    from orchard_fem.calibration.fenicsx_bridge import build_outcome_solver
    from orchard_fem.domain import ExcitationKind

    disp_model = replace(model, excitation=replace(
        model.excitation, kind=ExcitationKind.HARMONIC_DISPLACEMENT,
    ))
    return build_outcome_solver(
        disp_model, amplitude_unit="mm", coverage_mode=options.coverage_mode,
    )


def recommend_harvest_parameters(
    model: Any,
    *,
    model_path: str | Path = "",
    options: RecommendationOptions | None = None,
    progress_cb: ProgressCb = _noop_progress,
    cancel_cb: CancelCb = _never_cancel,
    frf_sweep: Callable | None = None,
    evaluator_factory: Callable | None = None,
) -> RecommendationResult:
    """Run the full recommendation pipeline on a loaded model.

    Parameters
    ----------
    model:
        Loaded :class:`OrchardModel`.
    options:
        Pipeline knobs (defaults to :class:`RecommendationOptions`).
    progress_cb / cancel_cb:
        GUI hooks: progress messages with a 0–1 fraction; cooperative
        cancellation checked between FE solves (raises ``RuntimeError`` with
        message "cancelled" when triggered).
    frf_sweep / evaluator_factory:
        FE-stage overrides for testing; defaults use the FEniCSx backend.

    Returns
    -------
    RecommendationResult
        Including the decision trace (:attr:`~RecommendationResult.steps`).
    """
    import numpy as np

    from orchard_fem.calibration.fenicsx_bridge import HarvestOutcome
    from orchard_fem.recommendation.pareto import (
        find_knee_min_distance,
        non_dominated_mask,
    )
    from orchard_fem.workflows._solve_pool import (
        resolve_n_jobs,
        solve_outcomes_parallel,
    )

    opt = options or RecommendationOptions()
    sweep = frf_sweep or _default_frf_sweep
    steps: list[str] = []
    t_start = time.time()

    def log(msg: str, frac: float) -> None:
        steps.append(msg)
        progress_cb(msg, frac)

    def check_cancel() -> None:
        if cancel_cb():
            raise RuntimeError("cancelled")

    # -- 1. fruit setup ------------------------------------------------------
    if opt.detachment_displacement_m is not None and model.fruit_policy is not None:
        model = replace(model, fruit_policy=replace(
            model.fruit_policy,
            detachment_displacement_m=float(opt.detachment_displacement_m),
        ))
        log(f"Detachment displacement set to {opt.detachment_displacement_m * 1000:g} mm", 0.02)
    if opt.dense_fruit_spacing is not None:
        if model.fruit_policy is None:
            raise ValueError("dense_fruit_spacing requires a fruit policy on the model.")
        fruits = generate_linear_fruits(model, model.fruit_policy, opt.dense_fruit_spacing)
        model = replace(model, fruits=fruits)
        log(f"Dense fruit placement: 1 fruit per {opt.dense_fruit_spacing * 100:g}% arc "
            f"length, {len(fruits)} total", 0.04)

    # -- 2. coarse FRF sweep → resonance candidates ---------------------------
    check_cancel()
    log(f"FRF sweep {opt.sweep_range_hz[0]:g}–{opt.sweep_range_hz[1]:g} Hz "
        f"({opt.sweep_steps} steps)…", 0.05)
    freqs, mags = sweep(model, opt.sweep_range_hz[0], opt.sweep_range_hz[1], opt.sweep_steps)
    peak_idx, genuine = find_in_band_resonance(freqs, mags, opt.band_hz)
    f_res = float(freqs[peak_idx])
    log(f"Primary resonance {f_res:.2f} Hz "
        f"({'local maximum' if genuine else 'curvature inflection'})", 0.20)
    sec_idx = find_in_band_peaks(freqs, mags, opt.band_hz)
    secondary = [float(freqs[k]) for k in sec_idx if int(k) != peak_idx]
    if secondary:
        log(f"Secondary peak(s) {', '.join(f'{f:.2f}' for f in secondary)} Hz", 0.21)

    # -- 3. (f, A) grid with the rig envelope ---------------------------------
    f_grid = build_frequency_grid(f_res, secondary, opt.band_hz)
    a_all = sorted(opt.amplitude_grid_mm)
    a_grid = [a for a in a_all if 2.0 * a <= opt.limits.max_stroke_mm]
    dropped = [a for a in a_all if a not in a_grid]
    if dropped:
        log(f"Amplitudes dropped {', '.join(f'{a:g}' for a in dropped)} mm: "
            f"stroke 2A exceeds the {opt.limits.max_stroke_mm:g} mm cylinder limit", 0.22)
    if not a_grid:
        raise ValueError("All amplitude candidates exceed the rig stroke limit.")
    n_unreach = sum(1 for f in f_grid for a in a_grid
                    if not rig_feasible(f, a, opt.limits))
    if n_unreach:
        log(f"{n_unreach} (f,A) points outside the cylinder frequency envelope, "
            f"marked non-executable", 0.23)
    log(f"Working-point grid: |f|={len(f_grid)} × |A|={len(a_grid)}", 0.24)

    # -- 4. per-clamp Pareto sweep --------------------------------------------
    # Only rig-feasible (f, A) cells are ever executable, so we solve those alone
    # (unreachable frequencies/strokes are skipped, not solved-then-discarded).
    clamp_labels = candidate_clamp_labels(model, opt)
    work_points = [
        (clamp_label, float(f), float(a))
        for clamp_label in clamp_labels
        for f in f_grid
        for a in a_grid
        if rig_feasible(float(f), float(a), opt.limits)
    ]
    if not work_points:
        raise RuntimeError(
            "No rig-feasible (f, A) work points: every grid cell is outside the "
            "cylinder envelope. Relax the grids or re-clamp."
        )
    theta = {
        "E": float(model.materials[0].youngs_modulus),
        "rho": float(model.materials[0].density),
    }
    # Parallelise only the default FE backend (an injected evaluator may hold an
    # unpicklable closure → run it serially in-process).
    injected = evaluator_factory is not None
    n_jobs = 1 if injected else resolve_n_jobs(opt.n_jobs, len(work_points))
    n_total = len(work_points)
    log(f"{len(clamp_labels)} candidate clamps, {n_total} rig-feasible solves"
        f"{'' if n_jobs == 1 else f' across {n_jobs} processes'}", 0.25)

    n_done = 0

    def _on_done(_clamp: str, _f: float, _a: float) -> None:
        nonlocal n_done
        n_done += 1
        progress_cb(f"FE solves {n_done}/{n_total}", 0.25 + 0.70 * n_done / n_total)

    if n_jobs > 1:
        from orchard_fem.domain import ExcitationKind
        disp_model = replace(model, excitation=replace(
            model.excitation, kind=ExcitationKind.HARMONIC_DISPLACEMENT))
        solver_kw = dict(amplitude_unit="mm", coverage_mode=opt.coverage_mode)
        outcomes = solve_outcomes_parallel(
            disp_model, solver_kw, work_points, theta, n_jobs,
            on_done=_on_done, cancel_cb=cancel_cb,
        )
    else:
        if injected:
            evaluate = evaluator_factory(model, opt)

            def solve(_params, f, a, clamp):
                cov, stress = evaluate(float(f), float(a), clamp)
                return HarvestOutcome(coverage=float(cov), trunk_stress_pa=float(stress),
                                      branch_governing_ratio={}, n_detached_fruits=0)
        else:
            solve = _default_outcome_solver_factory(model, opt)
        outcomes = {}
        for clamp_label, f, a in work_points:
            check_cancel()
            outcomes[(clamp_label, f, a)] = solve(theta, f, a, clamp_label)
            _on_done(clamp_label, f, a)

    # group the flat outcomes back into per-clamp working points
    clamps: list[ClampRecommendation] = []
    for clamp_label in clamp_labels:
        raw = [
            WorkingPoint(
                clamp_label=clamp_label, frequency_hz=float(f), amplitude_mm=float(a),
                coverage=outcomes[(clamp_label, float(f), float(a))].coverage,
                trunk_stress_pa=outcomes[(clamp_label, float(f), float(a))].trunk_stress_pa,
                rig_feasible=True,   # work_points were filtered to rig-feasible cells
            )
            for f in f_grid
            for a in a_grid
            if (clamp_label, float(f), float(a)) in outcomes
        ]
        if not raw:
            continue

        # hard constraints: rig envelope + stress sanity ceiling
        feasible = [p for p in raw
                    if p.rig_feasible and p.trunk_stress_pa <= opt.stress_ceiling_pa]
        if not feasible:
            clamps.append(ClampRecommendation(clamp_label, tuple(raw), None))
            continue
        objs = np.array([[-p.coverage, p.trunk_stress_pa] for p in feasible])
        nd_mask = non_dominated_mask(objs)
        nd_idx = np.flatnonzero(nd_mask)
        knee_local = find_knee_min_distance(objs[nd_idx])
        knee_pos = int(nd_idx[knee_local])
        marked: list[WorkingPoint] = []
        feas_pos = {id(p): i for i, p in enumerate(feasible)}
        for p in raw:
            i = feas_pos.get(id(p))
            on_front = i is not None and bool(nd_mask[i])
            marked.append(replace(
                p, on_front=on_front, is_knee=(i == knee_pos),
            ))
        knee = next(p for p in marked if p.is_knee)
        clamps.append(ClampRecommendation(clamp_label, tuple(marked), knee))

    with_knee = [c for c in clamps if c.knee is not None]
    if not with_knee:
        raise RuntimeError(
            "No executable working point found: every candidate violates the rig "
            "envelope or the stress ceiling. Relax the grids or re-clamp."
        )

    # -- 5. best clamp: knee closest to the ideal (coverage=1, σ=0) -----------
    sigma_norm = max(c.knee.trunk_stress_pa for c in with_knee) or 1.0
    def _distance(c: ClampRecommendation) -> float:
        k = c.knee
        return math.hypot(1.0 - k.coverage, k.trunk_stress_pa / sigma_norm)
    best = min(with_knee, key=_distance)
    best_idx = clamps.index(best)
    k = best.knee
    log(
        f"Recommended working point: clamp {best.clamp_label}, f={k.frequency_hz:.2f} Hz, "
        f"A={k.amplitude_mm:g} mm (stroke {k.stroke_mm:g} mm), "
        f"coverage {k.coverage:.2f}, trunk stress {k.trunk_stress_pa / 1e6:.2f} MPa",
        0.99,
    )

    return RecommendationResult(
        model_path=str(model_path),
        model_name=str(model.metadata.name),
        resonance_hz=f_res,
        secondary_hz=tuple(secondary),
        frequency_grid_hz=tuple(f_grid),
        amplitude_grid_mm=tuple(a_grid),
        clamps=tuple(clamps),
        best_clamp_index=best_idx,
        duration_s=opt.duration_s,
        steps=tuple(steps),
        elapsed_s=time.time() - t_start,
    )


# --------------------------------------------------------------------------- #
# Headless CLI (also exercised by tests)
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(
        description="Recommend rig-executable harvest parameters for a tree JSON.")
    parser.add_argument("model_json")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the RecommendationResult JSON here")
    parser.add_argument("--params-out", type=Path, default=None,
                        help="write the chosen working point as run_harvest_on_rig "
                             "params JSON")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args(argv)

    from orchard_fem.io.loaders import load_orchard_model
    model = load_orchard_model(args.model_json)
    print("\n".join(summarize_orchard_model(model, args.model_json).lines()))

    result = recommend_harvest_parameters(
        model, model_path=args.model_json,
        options=RecommendationOptions(duration_s=args.duration),
        progress_cb=lambda m, f: print(f"[{f * 100:3.0f}%] {m}"),
    )
    if args.out:
        result.save_json(args.out)
        print(f"result → {args.out}")
    if args.params_out and result.recommended:
        args.params_out.write_text(json.dumps(
            result.recommended.to_params_json(result.duration_s),
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"params → {args.params_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
