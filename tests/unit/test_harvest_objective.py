"""Tests for the time-dependent harvest objective (no FEniCSx required)."""
from __future__ import annotations

import math

import pytest

from orchard_fem.harvest.objective import (
    DetachmentFatigueLaw,
    HarvestObjectiveConfig,
    HarvestParameters,
    StressFatigueLaw,
    evaluate_harvest_objective,
    load_ratios_from_detachment,
    scale_stress_with_amplitude,
)


# ---------------------------------------------------------------------------
# DetachmentFatigueLaw
# ---------------------------------------------------------------------------

def test_immediate_detachment_above_threshold():
    law = DetachmentFatigueLaw()
    assert law.cycles_to_detach(1.5) == 0.0
    assert law.detached(1.5, n_cycles=1.0) is True
    # Zero applied cycles never detaches, even above threshold.
    assert law.detached(1.5, n_cycles=0.0) is False


def test_never_detaches_below_endurance():
    law = DetachmentFatigueLaw(endurance_ratio=0.3)
    assert math.isinf(law.cycles_to_detach(0.2))
    assert law.detached(0.2, n_cycles=1e12) is False


def test_finite_cycles_between_endurance_and_threshold():
    law = DetachmentFatigueLaw(endurance_ratio=0.3, reference_cycles=10.0, exponent=3.0)
    n_at_threshold = law.cycles_to_detach(0.999999)
    n_mid = law.cycles_to_detach(0.6)
    # Curve equals reference_cycles at r→1 and grows as r decreases.
    assert n_at_threshold == pytest.approx(10.0, rel=1e-3)
    assert n_mid > n_at_threshold
    assert math.isfinite(n_mid)


def test_fatigue_law_invalid_params():
    with pytest.raises(ValueError):
        DetachmentFatigueLaw(endurance_ratio=1.0)
    with pytest.raises(ValueError):
        DetachmentFatigueLaw(reference_cycles=0.0)
    with pytest.raises(ValueError):
        StressFatigueLaw(reference_stress=-1.0)


# ---------------------------------------------------------------------------
# Duration accumulation: detached fraction must be monotone in T
# ---------------------------------------------------------------------------

def _safe_config() -> HarvestObjectiveConfig:
    """A config whose stress thresholds never trip (isolates detachment physics)."""
    return HarvestObjectiveConfig(
        branch_ultimate_stress_pa=1e12,
        branch_fatigue=StressFatigueLaw(reference_stress=1e12),
        clamp_fatigue=StressFatigueLaw(reference_stress=1e12),
    )


def test_detached_fraction_monotone_in_duration():
    # Three fruits: one immediate (r>1), one fatigue (endurance<r<1), one never (r<endurance).
    load_ratios = [1.4, 0.6, 0.2]
    cfg = _safe_config()

    fracs = []
    for duration in [0.0, 0.5, 2.0, 10.0, 60.0]:
        params = HarvestParameters(
            frequency_hz=10.0, force_amplitude_n=1.0, duration_s=duration
        )
        res = evaluate_harvest_objective(
            load_ratios, params,
            branch_peak_stress_pa=0.0, clamp_stress_pa=0.0, config=cfg,
        )
        fracs.append(res.detached_fraction)

    # Non-decreasing in T, and saturates below 1.0 (the r<endurance fruit never goes).
    for prev, curr in zip(fracs, fracs[1:]):
        assert curr >= prev - 1e-12
    assert fracs[0] == pytest.approx(0.0)     # T=0 → n=0 → nothing detaches
    assert fracs[-1] == pytest.approx(2 / 3)  # immediate + fatigue fruit; never-fruit stays


def test_zero_duration_only_immediate():
    load_ratios = [1.4, 0.6]
    params = HarvestParameters(frequency_hz=10.0, force_amplitude_n=1.0, duration_s=0.0)
    res = evaluate_harvest_objective(
        load_ratios, params,
        branch_peak_stress_pa=0.0, clamp_stress_pa=0.0, config=_safe_config(),
    )
    # n_cycles = 0 → nothing detaches (immediate fruit needs n>0).
    assert res.n_detached == 0


# ---------------------------------------------------------------------------
# Branch fracture hard constraint
# ---------------------------------------------------------------------------

def test_overstress_makes_infeasible():
    cfg = HarvestObjectiveConfig(branch_ultimate_stress_pa=3.0e7)
    params = HarvestParameters(frequency_hz=10.0, force_amplitude_n=1.0, duration_s=1.0)
    res = evaluate_harvest_objective(
        [1.5, 1.5], params,
        branch_peak_stress_pa=4.0e7,   # exceeds ultimate
        clamp_stress_pa=0.0, config=cfg,
    )
    assert res.branch_fracture is True
    assert res.fracture_mode == "overstress"
    assert res.feasible is False
    assert res.objective == -math.inf


def test_fatigue_fracture_within_duration():
    cfg = HarvestObjectiveConfig(
        branch_ultimate_stress_pa=1e12,   # never instantaneous
        branch_fatigue=StressFatigueLaw(reference_stress=1.0e7, reference_cycles=100.0, exponent=6.0),
    )
    # At σ = reference_stress, N_fracture = 100 cycles. f·T = 10*20 = 200 ≥ 100 → fracture.
    params = HarvestParameters(frequency_hz=10.0, force_amplitude_n=1.0, duration_s=20.0)
    res = evaluate_harvest_objective(
        [1.5], params,
        branch_peak_stress_pa=1.0e7, clamp_stress_pa=0.0, config=cfg,
    )
    assert res.branch_fracture is True
    assert res.fracture_mode == "fatigue"
    assert res.feasible is False


def test_no_fracture_short_duration():
    cfg = HarvestObjectiveConfig(
        branch_ultimate_stress_pa=1e12,
        branch_fatigue=StressFatigueLaw(reference_stress=1.0e7, reference_cycles=100.0, exponent=6.0),
        clamp_fatigue=StressFatigueLaw(reference_stress=1e12),
    )
    # f·T = 10*5 = 50 < 100 → no fatigue fracture.
    params = HarvestParameters(frequency_hz=10.0, force_amplitude_n=1.0, duration_s=5.0)
    res = evaluate_harvest_objective(
        [1.5], params,
        branch_peak_stress_pa=1.0e7, clamp_stress_pa=0.0, config=cfg,
    )
    assert res.feasible is True
    assert res.objective == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clamp-point soft penalty
# ---------------------------------------------------------------------------

def test_clamp_penalty_reduces_objective_but_stays_feasible():
    cfg = HarvestObjectiveConfig(
        branch_ultimate_stress_pa=1e12,
        branch_fatigue=StressFatigueLaw(reference_stress=1e12),
        clamp_fatigue=StressFatigueLaw(reference_stress=1.0e7, reference_cycles=100.0, exponent=6.0),
        clamp_penalty_weight=0.5,
    )
    # Clamp damage = (10*20) / 100 = 2.0; penalty = 0.5 * 2.0 = 1.0.
    params = HarvestParameters(frequency_hz=10.0, force_amplitude_n=1.0, duration_s=20.0)
    res = evaluate_harvest_objective(
        [1.5], params,
        branch_peak_stress_pa=0.0, clamp_stress_pa=1.0e7, config=cfg,
    )
    assert res.feasible is True            # clamp damage is a soft penalty, not a constraint
    assert res.clamp_damage == pytest.approx(2.0)
    assert res.clamp_penalty == pytest.approx(1.0)
    assert res.objective == pytest.approx(1.0 - 1.0)   # detached_fraction - penalty


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, inertia, detach):
        self.inertia_force_n = inertia
        self.detachment_force_n = detach


class _DetachResult:
    def __init__(self, states):
        self.states = states


def test_load_ratios_amplitude_scaling():
    # Reference at 1 N: r = 0.5/1.0 = 0.5. At A=4 N → r = 2.0.
    dr = _DetachResult([_State(inertia=0.5, detach=1.0)])
    ratios = load_ratios_from_detachment(dr, force_amplitude_n=4.0, reference_amplitude_n=1.0)
    assert ratios[0] == pytest.approx(2.0)


def test_stress_amplitude_scaling():
    assert scale_stress_with_amplitude(1.0e6, force_amplitude_n=3.0) == pytest.approx(3.0e6)


def test_fruit_weights_length_mismatch_raises():
    params = HarvestParameters(frequency_hz=10.0, force_amplitude_n=1.0, duration_s=1.0)
    with pytest.raises(ValueError, match="fruit_weights"):
        evaluate_harvest_objective(
            [1.5, 0.5], params,
            branch_peak_stress_pa=0.0, clamp_stress_pa=0.0,
            fruit_weights=[1.0],
        )
