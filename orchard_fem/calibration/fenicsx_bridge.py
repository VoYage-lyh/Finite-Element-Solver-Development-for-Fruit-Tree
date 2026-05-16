"""FEniCSx bridges that wire the abstract UQ modules to the actual FEM solver.

Two factory functions are provided here, both returning callables that the
Bayesian / Pareto layers can consume without ever importing FEniCSx
themselves:

* :func:`build_fenicsx_forward_operator` →
      ``params_dict -> ForwardResult``  (modal frequencies + complex FRF)
  Used by :func:`orchard_fem.calibration.run_emcee_calibration`.

* :func:`build_fenicsx_pareto_evaluator` →
      ``(params, freq_hz, amplitude, clamp_label) -> HarvestObjective``
  Used by :func:`orchard_fem.recommendation.pareto.pareto_front_from_grid`.
  The new two-objective formulation evaluates:

    1. **detachment_coverage** = fraction of fruits whose dynamic inertia
       force ``m·a`` exceeds the local attachment force ``k_attach·d_detach``,
       with ``a = ω²·|u|`` at the fruit's attachment node.
    2. **trunk_max_stress** = ``E · r_outer · |κ|`` at the trunk root, with
       curvature ``|κ| = √(κ_y²+κ_z²)`` reconstructed from the FE complex
       rotations at two adjacent trunk nodes.

The closures clone the supplied :class:`OrchardModel` per call (Python dataclass
``replace``) so the caller's model object is never mutated. Heavy FEniCSx
imports happen on first invocation, keeping module import cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from orchard_fem.calibration.bayesian_calibration import ForwardResult


# ────────────────────────────────────────────────────────────────────────────
#  Parameter → model mapping
# ────────────────────────────────────────────────────────────────────────────
def _apply_theta_to_model(model, params: dict[str, float]):
    """Return a clone of *model* with θ applied to materials / clamps / fruits.

    Supported keys (any subset):
      E, rho            — applied to all materials matching ``calibrated_materials``
      zeta1, zeta2      — used by :func:`rayleigh_from_modal_damping_hz` later
      k_c, c_c          — applied to every ``ClampBoundaryCondition``
      k_f, c_f          — applied to every ``FruitAttachment``
    """
    # 1. Materials (E, ρ)
    if "E" in params or "rho" in params:
        new_materials = []
        for mat in model.materials:
            kwargs = {}
            if "E" in params:
                kwargs["youngs_modulus"] = float(params["E"])
            if "rho" in params:
                kwargs["density"] = float(params["rho"])
            new_materials.append(replace(mat, **kwargs))
        model = replace(model, materials=new_materials)

    # 2. Clamps (k_c, c_c)
    if ("k_c" in params or "c_c" in params) and model.clamps:
        new_clamps = []
        for clamp in model.clamps:
            kwargs = {}
            if "k_c" in params:
                kwargs["support_stiffness"] = float(params["k_c"])
            if "c_c" in params:
                kwargs["support_damping"] = float(params["c_c"])
            new_clamps.append(replace(clamp, **kwargs))
        model = replace(model, clamps=new_clamps)

    # 3. Fruits (k_f, c_f)
    if ("k_f" in params or "c_f" in params) and model.fruits:
        new_fruits = []
        for fruit in model.fruits:
            kwargs = {}
            if "k_f" in params:
                kwargs["stiffness"] = float(params["k_f"])
            if "c_f" in params:
                kwargs["damping"] = float(params["c_f"])
            new_fruits.append(replace(fruit, **kwargs))
        model = replace(model, fruits=new_fruits)

    return model


def _apply_rayleigh_to_model(model, zeta1: float, zeta2: float,
                              f1_hz: float, f2_hz: float):
    """Inject (α, β) into ``model.analysis`` from two modal anchors."""
    from orchard_fem.dynamics.rayleigh import rayleigh_from_modal_damping_hz

    if f1_hz <= 0.0 or f2_hz <= 0.0 or abs(f1_hz - f2_hz) < 1.0e-6:
        # Bad anchor → leave the existing Rayleigh untouched and return model unchanged.
        return model
    coef = rayleigh_from_modal_damping_hz(
        zeta1=zeta1, zeta2=zeta2, f1_hz=f1_hz, f2_hz=f2_hz,
    )
    return replace(
        model,
        analysis=replace(
            model.analysis,
            rayleigh_alpha=float(coef.alpha),
            rayleigh_beta=float(coef.beta),
        ),
    )


def _set_frequency_grid(model, frequencies_hz: np.ndarray):
    """Configure ``model.analysis`` to sweep exactly *frequencies_hz*.

    Assumes a uniformly-spaced grid; the solver builds its own linspace from
    ``(start, end, steps)``. We pick start/end so the resulting linspace matches.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    if f.size < 2:
        raise ValueError("Need at least two frequency points.")
    return replace(
        model,
        analysis=replace(
            model.analysis,
            frequency_start_hz=float(f[0]),
            frequency_end_hz=float(f[-1]),
            frequency_steps=int(f.size),
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
#  Forward operator for Bayesian calibration
# ────────────────────────────────────────────────────────────────────────────
def build_fenicsx_forward_operator(
    model,
    frf_frequencies_hz: Sequence[float],
    *,
    target_observation_ids: Sequence[str],
    n_modes: int = 3,
    polynomial_degree: int = 1,
):
    """Return a Bayesian forward operator backed by the FEniCSx solver.

    Parameters
    ----------
    model:
        Baseline :class:`OrchardModel` (excitation already pointing at the
        clamp location used during the FRF measurement).
    frf_frequencies_hz:
        Frequency grid where the measured FRF lives — the simulator is set up
        to sweep exactly these frequencies so likelihoods compare point-wise.
    target_observation_ids:
        Observation IDs whose ``ux``-component magnitudes are averaged into a
        scalar FRF for the likelihood. Typically a single observation point on
        the target fruit-bearing branch.
    n_modes:
        Number of modes to request from the modal solver (default 3 per paper).
    polynomial_degree:
        Beam discretisation order forwarded to the solver.

    Returns
    -------
    Callable[[dict[str, float]], ForwardResult]
    """
    freqs_arr = np.asarray(frf_frequencies_hz, dtype=float)
    target_set = set(target_observation_ids)

    def operator(params: dict[str, float]) -> ForwardResult:
        # Lazy imports — FEniCSx is heavy.
        from orchard_fem.fenicsx.frequency_response import (
            solve_embedded_beam_frequency_response_experiment,
        )
        from orchard_fem.fenicsx.modal import (
            solve_embedded_beam_modal_experiment,
        )

        # 1. θ → materials / clamps / fruits
        cloned = _apply_theta_to_model(model, params)

        # 2. Modal analysis (first n_modes natural frequencies)
        modal = solve_embedded_beam_modal_experiment(cloned, num_modes=n_modes)
        modal_freqs = np.array(
            [m.frequency_hz for m in modal.modes[:n_modes]], dtype=float,
        )
        if modal_freqs.size < 2:
            # Cannot derive Rayleigh from one anchor; fall back to model defaults.
            f1, f2 = (modal_freqs[0] if modal_freqs.size else 1.0,
                      modal_freqs[0] * 2.0 if modal_freqs.size else 2.0)
        else:
            f1, f2 = float(modal_freqs[0]), float(modal_freqs[1])

        # 3. Inject Rayleigh derived from (ζ_1, ζ_2) and the first two modes.
        if "zeta1" in params and "zeta2" in params:
            cloned = _apply_rayleigh_to_model(
                cloned,
                zeta1=float(params["zeta1"]),
                zeta2=float(params["zeta2"]),
                f1_hz=f1, f2_hz=f2,
            )

        # 4. Configure & run FRF sweep on exactly the measured grid.
        cloned = _set_frequency_grid(cloned, freqs_arr)
        frf_exp = solve_embedded_beam_frequency_response_experiment(
            cloned, polynomial_degree=polynomial_degree,
        )
        result = frf_exp.result
        names = result.observation_names
        col_indices = [i for i, name in enumerate(names) if name in target_set]
        if not col_indices:
            raise RuntimeError(
                f"None of target_observation_ids {sorted(target_set)} found in "
                f"FRF observation_names {names}."
            )

        # Aggregate magnitude across target observations (mean over selected channels)
        mag = np.array(
            [
                float(np.mean([p.observation_magnitudes[i] for i in col_indices]))
                for p in result.points
            ],
            dtype=float,
        )
        # Wrap as complex so the Bayesian likelihood (which calls np.abs) works.
        H = mag.astype(np.complex128)
        return ForwardResult(
            modal_frequencies_hz=modal_freqs,
            frf_complex=H,
        )

    return operator


# ────────────────────────────────────────────────────────────────────────────
#  Pareto objective evaluator
# ────────────────────────────────────────────────────────────────────────────
def _parse_clamp_label(clamp_label: str) -> tuple[str, float | None]:
    """Parse 'branch_id@s_value' or just 'branch_id'."""
    if "@" in clamp_label:
        branch_id, s_str = clamp_label.split("@", 1)
        return branch_id.strip(), float(s_str)
    return clamp_label.strip(), None


def _trunk_outer_radius(model, branch_id: str) -> float:
    """Return the outermost radius at the trunk's root section (s = 0)."""
    branch = next(b for b in model.branches if b.branch_id == branch_id)
    profiles = branch.section_series.profiles
    if not profiles:
        return 0.05  # fallback
    first = profiles[0]
    # Pick the largest outer-radius across all tissue regions (woody bark dominates).
    max_r = 0.0
    for region in getattr(first, "_regions", getattr(first, "regions", [])):
        outer = max(region.geometry.outer_radii)
        max_r = max(max_r, float(outer))
    return max_r if max_r > 0.0 else 0.05


def _trunk_first_segment_length(model, branch_id: str) -> float:
    """Return Δs between trunk root (s=0) and trunk mid (s=0.5) in metres.

    The downstream ``EmbeddedBeamResponseMapping`` resolver only honours the
    ``target_node`` field on ``ObservationPoint``, not ``target_s``. So to get
    two physically distinct evaluation points we use the named "root" and
    "mid" nodes, which gives ``Δs = L/2``.
    """
    branch = next(b for b in model.branches if b.branch_id == branch_id)
    return float(branch.path.length()) * 0.5


def _augment_observations_for_evaluation(model, trunk_branch_id: str):
    """Append fruit + trunk-stress observations and return (augmented_model,
    fruit_obs_keys, trunk_obs_keys).

    The keys are observation IDs we just inserted; the caller uses them to
    look up the corresponding entries in ``observation_names`` after solving.
    """
    from orchard_fem.topology import ObservationPoint

    extra: list[ObservationPoint] = []
    fruit_keys: list[tuple[str, str]] = []   # (obs_id, fruit_id)
    for fruit in model.fruits:
        oid = f"__pareto_fruit_{fruit.fruit_id}"
        # Fruit augmentation produces one scalar DOF (the attachment-direction
        # response). Only one component slot is needed.
        extra.append(ObservationPoint(
            observation_id=oid,
            target_type="fruit",
            target_id=fruit.fruit_id,
            target_node="tip",
            target_components=[fruit.target_component],
        ))
        fruit_keys.append((oid, fruit.fruit_id))

    # Trunk rotation at root and mid. Use named nodes because the response
    # mapping resolver ignores target_s on branch observations.
    trunk_keys = ("__pareto_trunk_root_rot", "__pareto_trunk_mid_rot")
    extra.append(ObservationPoint(
        observation_id=trunk_keys[0],
        target_type="branch",
        target_id=trunk_branch_id,
        target_node="root",
        target_components=["ry", "rz"],
    ))
    extra.append(ObservationPoint(
        observation_id=trunk_keys[1],
        target_type="branch",
        target_id=trunk_branch_id,
        target_node="mid",
        target_components=["ry", "rz"],
    ))

    augmented = replace(model, observations=list(model.observations) + extra)
    return augmented, fruit_keys, trunk_keys


def _component_complex(
    point,
    observation_names: list[str],
    obs_id: str,
    components: tuple[str, ...],
) -> dict[str, complex]:
    """Pick the complex amplitudes belonging to *obs_id* off the FRF point.

    Observations live in the FRF result as one entry per (obs_id, component).
    The conventional naming inside the FE response mapping is
    ``"{obs_id}::{component}"`` or similar — we match by ``startswith(obs_id)``
    and split off the component suffix.
    """
    out: dict[str, complex] = {}
    cplx = point.observation_complex
    if cplx is None:
        # Fallback: magnitudes only, treat as real-valued.
        cplx = tuple(complex(m, 0.0) for m in point.observation_magnitudes)
    for i, name in enumerate(observation_names):
        if not name.startswith(obs_id):
            continue
        # name pattern: "{obs_id}__{component}" or "{obs_id}_{component}"
        for comp in components:
            if name.endswith(comp):
                out[comp] = cplx[i]
                break
    return out


def build_fenicsx_pareto_evaluator(
    model,
    *,
    polynomial_degree: int = 1,
    amplitude_unit: str = "mm",
    trunk_branch_id: str = "trunk",
):
    """Return a Pareto-objective evaluator backed by the FEniCSx solver.

    Two-objective formulation (matches the redesigned :class:`HarvestObjective`):

    1. **detachment_coverage** — fraction of fruits whose inertia force
       ``m·a`` ≥ attachment force ``k_attach·d_detach``. Each fruit is
       augmented as a single lumped-mass DOF along its attachment direction;
       the observed scalar response ``|u|`` is converted to acceleration via
       ``a = ω²·|u|``.
    2. **trunk_max_stress** — bending stress at the trunk root computed from
       the curvature ``|κ|`` between two adjacent trunk FE nodes (rotation
       components ``ry``, ``rz``) via ``σ = E · r_outer · |κ|``.

    The caller does **not** need to pre-tag observations — the bridge
    auto-augments the model with the necessary observations on each call.

    Parameters
    ----------
    model:
        Baseline :class:`OrchardModel`. Must define fruits to get a meaningful
        coverage; ``trunk_branch_id`` must reference an existing branch.
    polynomial_degree:
        Beam discretisation order.
    amplitude_unit:
        ``"mm"`` (default) or ``"m"`` — selects the conversion factor between
        the Pareto ``A`` scalar and the ``HarmonicExcitation.amplitude`` value.
    trunk_branch_id:
        Branch ID used for stress evaluation (default ``"trunk"``).

    Returns
    -------
    Callable[[dict, float, float, str], HarvestObjective]
    """
    from orchard_fem.recommendation.pareto import HarvestObjective  # avoid cycle

    if amplitude_unit not in ("mm", "m"):
        raise ValueError("amplitude_unit must be 'mm' or 'm'.")
    A_scale = 1.0e-3 if amplitude_unit == "mm" else 1.0

    r_outer = _trunk_outer_radius(model, trunk_branch_id)
    delta_s = _trunk_first_segment_length(model, trunk_branch_id)

    def evaluator(params, frequency_hz, amplitude, clamp_label):
        from orchard_fem.fenicsx.frequency_response import (
            solve_embedded_beam_frequency_response_experiment,
        )

        # 1. θ → model
        cloned = _apply_theta_to_model(model, params)

        # 2. Reroute excitation to the candidate clamp.
        branch_id, target_s = _parse_clamp_label(clamp_label)
        new_exc = replace(
            cloned.excitation,
            target_branch_id=branch_id,
            target_s=target_s if target_s is not None else cloned.excitation.target_s,
            amplitude=float(amplitude) * A_scale,
            driving_frequency_hz=float(frequency_hz),
        )
        cloned = replace(cloned, excitation=new_exc)

        # 3. Single-frequency analysis grid.
        cloned = replace(
            cloned,
            analysis=replace(
                cloned.analysis,
                frequency_start_hz=float(frequency_hz),
                frequency_end_hz=float(frequency_hz) + 1.0e-6,
                frequency_steps=1,
            ),
        )

        # 4. Augment observations and solve.
        cloned, fruit_keys, trunk_keys = _augment_observations_for_evaluation(
            cloned, trunk_branch_id,
        )
        frf_exp = solve_embedded_beam_frequency_response_experiment(
            cloned, polynomial_degree=polynomial_degree,
        )
        point = frf_exp.result.points[0]
        names = frf_exp.result.observation_names
        omega = 2.0 * np.pi * float(frequency_hz)

        # ── 5. Coverage from fruit inertia forces ──────────────────────────
        # Each fruit augmentation produces ONE scalar DOF (the lumped mass
        # along its attachment_component), so the observation appears under
        # the bare ``obs_id`` — no component suffix.
        d_detach = 0.010
        if cloned.fruit_policy is not None:
            d_detach = float(cloned.fruit_policy.detachment_displacement_m)
        n_detached = 0
        n_total = 0
        fruit_lookup = {f.fruit_id: f for f in cloned.fruits}
        name_to_idx = {n: i for i, n in enumerate(names)}
        cplx = point.observation_complex
        if cplx is None:
            cplx = tuple(complex(m, 0.0) for m in point.observation_magnitudes)
        for obs_id, fruit_id in fruit_keys:
            fruit = fruit_lookup.get(fruit_id)
            if fruit is None or fruit.mass <= 0.0 or fruit.stiffness <= 0.0:
                continue
            idx = name_to_idx.get(obs_id)
            if idx is None:
                continue
            u_mag = float(abs(cplx[idx]))
            a_fruit = omega * omega * u_mag                  # m/s²
            inertia = fruit.mass * a_fruit                   # N
            detach_force = fruit.stiffness * d_detach
            n_total += 1
            if inertia >= detach_force:
                n_detached += 1

        coverage = n_detached / n_total if n_total else 0.0

        # ── 6. Trunk max bending stress σ = E · r · |κ| ─────────────────────
        root_rot = _component_complex(point, names, trunk_keys[0], ("ry", "rz"))
        next_rot = _component_complex(point, names, trunk_keys[1], ("ry", "rz"))
        if root_rot and next_rot:
            # Complex rotation differences along the trunk axis
            d_ry = next_rot.get("ry", 0.0) - root_rot.get("ry", 0.0)
            d_rz = next_rot.get("rz", 0.0) - root_rot.get("rz", 0.0)
            kappa_mag = (abs(d_ry) ** 2 + abs(d_rz) ** 2) ** 0.5 / max(delta_s, 1.0e-9)
        else:
            kappa_mag = 0.0
        E = float(params.get("E", 1.0e10))
        stress = E * r_outer * kappa_mag

        return HarvestObjective(
            detachment_coverage=float(coverage),
            trunk_max_stress=float(stress),
        )

    return evaluator


__all__ = [
    "build_fenicsx_forward_operator",
    "build_fenicsx_pareto_evaluator",
]
