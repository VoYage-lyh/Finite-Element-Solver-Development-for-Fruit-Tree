"""Smoke tests for the uncertainty-quantification stack.

Covers the modules added for the parameter-uncertainty harvesting paper:

* Rayleigh α/β ↔ ζ₁/ζ₂ inversion
* Material-test → Bayesian priors
* FRF post-processing (H1 + half-power BW)
* Bayesian calibration (emcee on a cheap analytic forward operator)
* Fixed-frequency posterior-predictive coverage
* Pareto non-dominated sort + knee selection
* Sobol sensitivity on Ishigami benchmark

Heavy dependencies (emcee, SALib) are imported only inside the relevant tests
so the rest of the suite stays runnable on minimal installations.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────────────────
#  Rayleigh
# ────────────────────────────────────────────────────────────────────────────
def test_rayleigh_round_trip_recovers_zetas():
    from orchard_fem.dynamics.rayleigh import rayleigh_from_modal_damping_hz

    target_z1, target_z2 = 0.046, 0.036
    f1, f2 = 6.05, 11.21
    coef = rayleigh_from_modal_damping_hz(
        zeta1=target_z1, zeta2=target_z2, f1_hz=f1, f2_hz=f2,
    )
    assert coef.modal_damping_ratio(2 * math.pi * f1) == pytest.approx(target_z1, abs=1e-9)
    assert coef.modal_damping_ratio(2 * math.pi * f2) == pytest.approx(target_z2, abs=1e-9)


def test_rayleigh_rejects_singular_anchors():
    from orchard_fem.dynamics.rayleigh import rayleigh_from_modal_damping

    with pytest.raises(ValueError):
        rayleigh_from_modal_damping(zeta1=0.04, zeta2=0.03, omega1=10.0, omega2=10.0)


# ────────────────────────────────────────────────────────────────────────────
#  Material test → priors
# ────────────────────────────────────────────────────────────────────────────
def test_priors_from_material_test_shape():
    from orchard_fem.io.material_test import (
        BranchMaterialTestSummary, priors_from_material_test,
    )

    summary = BranchMaterialTestSummary(
        n_samples=30,
        fresh_density_kgm3_mean=765.0,
        fresh_density_kgm3_sd=58.0,
        bending_E_GPa_mean=7.85,
        bending_E_GPa_sd=1.12,
    )
    priors = priors_from_material_test(summary)
    names = [p.name for p in priors]
    assert names == ["E", "rho", "zeta1", "zeta2", "k_c", "c_c", "k_f", "c_f"]
    # E prior is in Pa, mean = 7.85 GPa
    assert priors[0].mean == pytest.approx(7.85e9)
    assert priors[0].sd == pytest.approx(1.12e9)


def test_material_test_summary_round_trips_json(tmp_path: Path):
    from orchard_fem.io.material_test import BranchMaterialTestSummary

    summary = BranchMaterialTestSummary(
        n_samples=12,
        fresh_density_kgm3_mean=750.0,
        fresh_density_kgm3_sd=50.0,
        bending_E_GPa_mean=7.5,
        bending_E_GPa_sd=1.0,
        notes="unit test",
    )
    p = tmp_path / "mat.json"
    summary.to_json(p)
    loaded = BranchMaterialTestSummary.from_json(p)
    assert loaded.n_samples == 12
    assert loaded.notes == "unit test"
    assert loaded.bending_E_GPa_mean == 7.5


# ────────────────────────────────────────────────────────────────────────────
#  FRF post-processing
# ────────────────────────────────────────────────────────────────────────────
def _synthetic_hammer_record(seed: int, fs: float, duration_s: float,
                              modes: list[tuple[float, float]]):
    from orchard_fem.processing import HammerRecord

    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    pulse_n = int(0.003 * fs)
    force = np.zeros(n)
    force[:pulse_n] = 100.0 * np.sin(np.pi * np.arange(pulse_n) / pulse_n)
    response = np.zeros(n)
    for f0, zeta in modes:
        omega = 2 * np.pi * f0
        wd = omega * np.sqrt(1 - zeta * zeta)
        response += np.exp(-zeta * omega * t) * np.sin(wd * t) / omega
    response += 0.0005 * rng.standard_normal(n)
    return HammerRecord(force=force, response=response, sample_rate_hz=fs)


def test_frf_pipeline_recovers_known_peaks():
    pytest.importorskip("scipy")
    from orchard_fem.processing import estimate_h1_average, identify_modes

    records = [
        _synthetic_hammer_record(s, fs=1024.0, duration_s=8.0,
                                  modes=[(5.0, 0.04), (12.0, 0.03)])
        for s in range(5)
    ]
    frf = estimate_h1_average(records, gamma_min=0.5, nperseg=2048)
    modes = identify_modes(frf, n_modes=2, frequency_band_hz=(2.0, 20.0))
    freqs = sorted(m.frequency_hz for m in modes.modes)
    assert freqs[0] == pytest.approx(5.0, abs=0.2)
    assert freqs[1] == pytest.approx(12.0, abs=0.2)


# ────────────────────────────────────────────────────────────────────────────
#  Bayesian calibration
# ────────────────────────────────────────────────────────────────────────────
def _two_mode_forward(params):
    from orchard_fem.calibration import ForwardResult

    E = params["E"]
    f1 = 5.0 * np.sqrt(E / 1.0e10)
    f2 = 12.0 * np.sqrt(E / 1.0e10)
    w = 2 * np.pi * np.linspace(2.0, 25.0, 60)
    Z = 1.0 / ((2 * np.pi * f1) ** 2 - w * w
               + 2j * params["zeta1"] * (2 * np.pi * f1) * w)
    Z += 1.0 / ((2 * np.pi * f2) ** 2 - w * w
                + 2j * params["zeta2"] * (2 * np.pi * f2) * w)
    return ForwardResult(modal_frequencies_hz=np.array([f1, f2]), frf_complex=Z)


def test_bayesian_calibration_recovers_synthetic_truth():
    pytest.importorskip("emcee")
    from orchard_fem.calibration import (
        BayesianLikelihood, BayesianPrior, run_emcee_calibration,
    )

    truth = _two_mode_forward({"E": 1.0e10, "zeta1": 0.04, "zeta2": 0.03})
    rng = np.random.default_rng(0)
    likelihood = BayesianLikelihood(
        modal_frequencies_hz=truth.modal_frequencies_hz,
        frf_frequencies_hz=np.linspace(2, 25, 60),
        frf_log_magnitudes=np.log(np.abs(truth.frf_complex))
        + 0.05 * rng.standard_normal(60),
    )
    priors = [
        BayesianPrior("E", "truncnorm", (4e9, 1.5e10), mean=8e9, sd=2e9),
        BayesianPrior("zeta1", "loguniform", (0.01, 0.10)),
        BayesianPrior("zeta2", "loguniform", (0.01, 0.10)),
    ]
    post = run_emcee_calibration(
        _two_mode_forward, priors, likelihood,
        n_walkers=12, n_steps=400, n_burn=200, seed=1,
    )
    lo, med, hi = post.credible_interval("E", alpha=0.95)
    assert lo <= 1.0e10 <= hi
    lo, med, hi = post.credible_interval("zeta1", alpha=0.95)
    assert lo <= 0.04 <= hi


# ────────────────────────────────────────────────────────────────────────────
#  Fixed-frequency validation
# ────────────────────────────────────────────────────────────────────────────
def test_coverage_report_counts_inside_outside():
    from orchard_fem.validation import (
        FixedFrequencyRecord, check_posterior_coverage,
    )

    records = [
        FixedFrequencyRecord(10.0, 6.0),
        FixedFrequencyRecord(20.0, 100.0),  # way outside
    ]
    rng = np.random.default_rng(0)
    predictive = {
        10.0: rng.normal(6.0, 0.5, size=200),
        20.0: rng.normal(3.0, 0.4, size=200),
    }
    report = check_posterior_coverage(records, predictive, nominal_coverage=0.90)
    assert report.n_records() == 2
    assert report.n_hits() == 1
    assert 0.4 <= report.empirical_coverage <= 0.6


# ────────────────────────────────────────────────────────────────────────────
#  Pareto
# ────────────────────────────────────────────────────────────────────────────
def test_non_dominated_mask_basic():
    from orchard_fem.recommendation import non_dominated_mask

    pts = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 5.0]])
    mask = non_dominated_mask(pts)
    # (5,5) is dominated by (1,5) and (4,2) — exclude it; the rest form a front
    assert mask.tolist() == [True, True, True, True, False]


def test_pareto_knee_selection():
    from orchard_fem.recommendation import find_knee_min_distance

    front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])  # symmetric L
    knee = find_knee_min_distance(front)
    assert knee == 1  # the corner / balanced point


def test_pareto_front_from_grid_smoke():
    from orchard_fem.recommendation import (
        HarvestObjective, pareto_front_from_grid,
    )

    def evaluator(params, f, A, clamp):
        # Coverage peaks near resonance (f≈10), stress grows with A and f.
        coverage = max(0.0, 1.0 - ((f - 10.0) / 6.0) ** 2 - max(0.0, 0.5 - A / 20.0))
        stress = 1.0e7 * (A / 20.0) ** 1.5 * (f / 10.0) ** 1.2
        return HarvestObjective(
            detachment_coverage=min(1.0, coverage),
            trunk_max_stress=stress,
        )
    front = pareto_front_from_grid(
        "trunk", np.arange(4.0, 24.5, 1.0), np.arange(5.0, 32.5, 2.5),
        evaluator, params={},
    )
    assert front.non_dominated_index.size > 0
    knee = front.knee
    assert 0.0 <= knee.detachment_coverage <= 1.0
    assert knee.trunk_max_stress >= 0.0


# ────────────────────────────────────────────────────────────────────────────
#  Sobol
# ────────────────────────────────────────────────────────────────────────────
def test_forward_cache_returns_same_value():
    """LRU cache should return the stored value on repeated identical θ."""
    from orchard_fem.calibration import cache_forward

    calls = {"n": 0}

    def forward(params):
        calls["n"] += 1
        return params["x"] ** 2

    cached = cache_forward(forward, cache_size=16)
    assert cached({"x": 1.5}) == 2.25
    assert cached({"x": 1.5}) == 2.25  # hit
    assert cached({"x": 2.0}) == 4.0   # miss
    assert calls["n"] == 2             # only 2 underlying calls

    stats = cached.cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2


def test_pareto_cache_handles_extra_args():
    """The Pareto cache key must include (f, A, clamp), not just params."""
    from orchard_fem.calibration import cache_pareto
    from orchard_fem.recommendation import HarvestObjective

    calls = {"n": 0}

    def evaluator(params, f, A, clamp):
        calls["n"] += 1
        return HarvestObjective(
            detachment_coverage=min(1.0, f * A / 500.0),
            trunk_max_stress=hash(clamp) % 100 * 1.0e5,
        )

    cached = cache_pareto(evaluator, cache_size=16)
    cached({"E": 1e10}, 10.0, 20.0, "trunk@0.5")        # miss → calls=1
    cached({"E": 1e10}, 10.0, 20.0, "trunk@0.5")        # hit  → calls=1
    cached({"E": 1e10}, 10.0, 20.0, "trunk@0.6")        # miss → calls=2 (new clamp)
    cached({"E": 1e10}, 10.5, 20.0, "trunk@0.5")        # miss → calls=3 (new freq)
    cached({"E": 1.01e10}, 10.0, 20.0, "trunk@0.5")     # miss → calls=4 (new params)
    assert calls["n"] == 4


def test_sobol_recovers_ishigami_indices():
    pytest.importorskip("SALib")
    from orchard_fem.sensitivity import SobolInputDef, run_sobol_analysis

    def ishigami(p, a=7.0, b=0.1):
        return (
            math.sin(p["x1"])
            + a * math.sin(p["x2"]) ** 2
            + b * p["x3"] ** 4 * math.sin(p["x1"])
        )

    inputs = [
        SobolInputDef("x1", (-math.pi, math.pi)),
        SobolInputDef("x2", (-math.pi, math.pi)),
        SobolInputDef("x3", (-math.pi, math.pi)),
    ]
    res = run_sobol_analysis(ishigami, inputs, n_base=512,
                              bootstrap_resamples=80, seed=0)
    # Analytic reference: S1 ≈ [0.31, 0.44, 0.0], ST ≈ [0.56, 0.44, 0.24].
    # Tolerances reflect Monte-Carlo variance at N_base=512.
    assert res.first_order["x2"].value == pytest.approx(0.44, abs=0.10)
    assert res.total_effect["x1"].value == pytest.approx(0.56, abs=0.16)
    assert abs(res.first_order["x3"].value) < 0.12
    # Ordering by total effect should be x1 > x2 > x3
    ranked = [n for n, _ in res.ranked_by_total_effect()]
    assert ranked[0] == "x1"
    assert ranked[-1] == "x3"


# ────────────────────────────────────────────────────────────────────────────
#  End-to-end recommend workflow (no FEniCSx; stub forward operator)
# ────────────────────────────────────────────────────────────────────────────
def test_end_to_end_recommend_with_stub_forward(tmp_path: Path):
    """Run Bayesian + Pareto + Sobol on a cheap analytic forward operator.

    This is the smallest reproduction of the full ``recommend`` workflow:
    a 2-DOF analytic forward → posterior → per-clamp Pareto knee with 90% CI
    → Sobol indices. No FEniCSx is needed, so this test runs in the standard
    pytest environment.
    """
    pytest.importorskip("emcee")
    pytest.importorskip("SALib")

    from orchard_fem.calibration import (
        BayesianLikelihood, BayesianPrior, ForwardResult,
        cache_forward, cache_pareto, run_emcee_calibration,
    )
    from orchard_fem.recommendation import (
        HarvestObjective, propagate_posterior_to_pareto,
    )
    from orchard_fem.sensitivity import SobolInputDef, run_sobol_analysis

    # 1. Cheap analytic forward — a 2-DOF system parameterised by (E, zeta1, zeta2)
    def fwd(params):
        E = params["E"]
        f1 = 5.0 * np.sqrt(E / 1.0e10)
        f2 = 12.0 * np.sqrt(E / 1.0e10)
        w = 2 * np.pi * np.linspace(2.0, 25.0, 30)
        Z = 1.0 / ((2 * np.pi * f1) ** 2 - w * w
                   + 2j * params["zeta1"] * (2 * np.pi * f1) * w)
        Z += 1.0 / ((2 * np.pi * f2) ** 2 - w * w
                    + 2j * params["zeta2"] * (2 * np.pi * f2) * w)
        return ForwardResult(modal_frequencies_hz=np.array([f1, f2]),
                              frf_complex=Z)

    truth = fwd({"E": 1.0e10, "zeta1": 0.04, "zeta2": 0.03})
    rng = np.random.default_rng(0)
    likelihood = BayesianLikelihood(
        modal_frequencies_hz=truth.modal_frequencies_hz,
        frf_frequencies_hz=np.linspace(2, 25, 30),
        frf_log_magnitudes=np.log(np.abs(truth.frf_complex))
        + 0.05 * rng.standard_normal(30),
    )
    priors = [
        BayesianPrior("E", "truncnorm", (4e9, 1.5e10), mean=8e9, sd=2e9),
        BayesianPrior("zeta1", "loguniform", (0.01, 0.10)),
        BayesianPrior("zeta2", "loguniform", (0.01, 0.10)),
    ]

    # 2. Bayesian step (with cache)
    fwd_cached = cache_forward(fwd, cache_size=64)
    post = run_emcee_calibration(
        fwd_cached, priors, likelihood,
        n_walkers=10, n_steps=300, n_burn=150, seed=1,
    )
    lo, _, hi = post.credible_interval("E", alpha=0.95)
    assert lo <= 1.0e10 <= hi

    # 3. Pareto with synthetic objective surface
    def evaluator(params, f, A, clamp):
        peak = 10.0 * (params.get("E", 1e10) / 1e10) ** 0.5
        coverage = min(1.0, math.exp(-((f - peak) / 3.0) ** 2) * (A / 25.0))
        stress = 5.0e6 * (A / 20.0) ** 1.5 * (f / 10.0) ** 1.2
        return HarvestObjective(
            detachment_coverage=coverage,
            trunk_max_stress=stress,
        )
    eval_cached = cache_pareto(evaluator, cache_size=256)
    flat = post.flatten()
    posterior_thin = [
        dict(zip(post.parameter_names, flat[i]))
        for i in np.linspace(0, flat.shape[0] - 1, 60).astype(int)
    ]
    rec = propagate_posterior_to_pareto(
        posterior_thin, "trunk@0.50",
        np.arange(4.0, 24.5, 0.5), np.arange(5.0, 32.5, 2.5),
        eval_cached, credible_alpha=0.90,
    )
    assert 6.0 < rec.frequency_hz_median < 14.0
    # CI quantiles are non-decreasing by construction
    assert rec.frequency_hz_ci[0] <= rec.frequency_hz_median <= rec.frequency_hz_ci[1]

    # 4. Sobol on recommended frequency
    def sobol_target(params):
        return propagate_posterior_to_pareto(
            [params], "trunk@0.50",
            np.arange(4.0, 24.5, 1.0), np.arange(5.0, 32.5, 5.0),
            evaluator, credible_alpha=0.50,
        ).frequency_hz_median

    sobol = run_sobol_analysis(
        sobol_target,
        [SobolInputDef("E", (5e9, 1.5e10)),
         SobolInputDef("zeta1", (0.01, 0.10), log_scale=True),
         SobolInputDef("zeta2", (0.01, 0.10), log_scale=True)],
        n_base=64, bootstrap_resamples=40, seed=0,
    )
    # E dominates the recommended frequency in this synthetic model.
    assert sobol.total_effect["E"].value > sobol.total_effect["zeta1"].value
