"""Tests for harvest detachment criterion (no FEniCSx required)."""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Minimal model / batch-result stubs
# ---------------------------------------------------------------------------

def _make_fruit(
    fruit_id: str,
    mass: float,
    stiffness: float,
    branch_id: str = "trunk",
    location_s: float = 0.5,
):
    from orchard_fem.domain.entities import FruitAttachment

    return FruitAttachment(
        fruit_id=fruit_id,
        branch_id=branch_id,
        location_s=location_s,
        mass=mass,
        stiffness=stiffness,
        damping=0.0,
    )


def _make_model(fruits):
    """Minimal OrchardModel stub that only needs .fruits and .fruit_policy."""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.fruits = fruits
    model.fruit_policy = None
    return model


def _make_batch_result(freqs_hz, mags):
    """Minimal BatchExcitationResult with a given FRF."""
    from unittest.mock import MagicMock
    from orchard_fem.dynamics.frequency_response import (
        FrequencyResponsePoint,
        FrequencyResponseResult,
    )

    result = FrequencyResponseResult(
        observation_names=["test"],
        points=[
            FrequencyResponsePoint(
                frequency_hz=f,
                excitation_response_magnitude=m,
                observation_magnitudes=[m],
            )
            for f, m in zip(freqs_hz, mags)
        ],
    )
    br = MagicMock()
    br.result = result
    return br


# ---------------------------------------------------------------------------
# compute_detachment_at_frequency
# ---------------------------------------------------------------------------

def test_detachment_basic_threshold():
    """Fruit detaches when inertia force >= k*d."""
    from orchard_fem.harvest.detachment import compute_detachment_at_frequency

    d_detach = 0.01   # 10 mm
    k = 100.0         # N/m  → F_detach = 1 N
    mass = 0.05       # kg

    freq = 5.0   # Hz
    omega = 2.0 * math.pi * freq
    # |H| such that F_inertia = m * ω² * |H| just exceeds F_detach
    H = 1.0 / (mass * omega**2) + 1e-6   # compliance: barely detaches

    fruit = _make_fruit("f1", mass=mass, stiffness=k)
    model = _make_model([fruit])
    br = _make_batch_result([freq], [H])

    result = compute_detachment_at_frequency(
        model, [br], freq, detachment_displacement_m=d_detach
    )
    assert result.n_detached == 1
    assert result.detachment_fraction == pytest.approx(1.0)
    assert result.states[0].detached is True


def test_no_detachment_below_threshold():
    from orchard_fem.harvest.detachment import compute_detachment_at_frequency

    d_detach = 0.01
    k = 1000.0   # N/m  → F_detach = 10 N (hard to detach)
    mass = 0.05

    freq = 2.0
    H = 0.001   # small compliance → small inertia force

    fruit = _make_fruit("f1", mass=mass, stiffness=k)
    model = _make_model([fruit])
    br = _make_batch_result([freq], [H])

    result = compute_detachment_at_frequency(
        model, [br], freq, detachment_displacement_m=d_detach
    )
    assert result.n_detached == 0
    assert result.states[0].detached is False


def test_detachment_fraction_multiple_fruits():
    from orchard_fem.harvest.detachment import compute_detachment_at_frequency

    d_detach = 0.01
    freq = 5.0
    H = 0.003  # compliance: F_inertia = 0.05*(2π5)²*0.003 ≈ 0.148 N > 0.1 N

    # Fruit 1: easy to detach (low stiffness → F_detach=0.1 N < F_inertia)
    # Fruit 2: hard to detach (high stiffness → F_detach=1000 N)
    fruits = [
        _make_fruit("f1", mass=0.05, stiffness=10.0),
        _make_fruit("f2", mass=0.05, stiffness=100000.0),
    ]
    model = _make_model(fruits)
    br = _make_batch_result([freq], [H])

    result = compute_detachment_at_frequency(
        model, [br], freq, detachment_displacement_m=d_detach
    )
    assert result.n_detached == 1
    assert result.total_fruits == 2
    assert result.detachment_fraction == pytest.approx(0.5)


def test_empty_fruit_list():
    from orchard_fem.harvest.detachment import compute_detachment_at_frequency

    model = _make_model([])
    br = _make_batch_result([5.0], [0.001])
    result = compute_detachment_at_frequency(model, [br], 5.0)
    assert result.n_detached == 0
    assert result.detachment_fraction == 0.0


# ---------------------------------------------------------------------------
# compute_detachment_spectrum
# ---------------------------------------------------------------------------

def test_spectrum_length_matches_frequencies():
    from orchard_fem.harvest.detachment import compute_detachment_spectrum

    freqs = [1.0, 2.0, 3.0, 4.0, 5.0]
    fruits = [_make_fruit("f1", mass=0.1, stiffness=50.0)]
    model = _make_model(fruits)
    br = _make_batch_result(freqs, [0.001] * 5)

    spectrum = compute_detachment_spectrum(model, [br])
    assert len(spectrum.frequencies_hz) == len(freqs)
    assert len(spectrum.detachment_fractions) == len(freqs)
    assert len(spectrum.results) == len(freqs)


def test_spectrum_fraction_increases_with_frequency():
    """Higher frequency → larger inertia force → more detachments."""
    from orchard_fem.harvest.detachment import compute_detachment_spectrum

    d_detach = 0.01
    stiffnesses = [50.0, 200.0, 800.0, 3200.0, 12800.0]
    fruits = [_make_fruit(f"f{i}", mass=0.1, stiffness=k) for i, k in enumerate(stiffnesses)]
    model = _make_model(fruits)

    # Compliance is constant; inertia force ∝ ω²
    freqs = [1.0, 5.0, 10.0, 20.0, 40.0]
    H = 0.01   # m/N
    br = _make_batch_result(freqs, [H] * len(freqs))

    spectrum = compute_detachment_spectrum(model, [br], detachment_displacement_m=d_detach)
    # Detachment fraction should be monotonically non-decreasing
    fracs = spectrum.detachment_fractions
    for prev, curr in zip(fracs, fracs[1:]):
        assert curr >= prev - 1e-9


# ---------------------------------------------------------------------------
# optimize_harvest_excitation
# ---------------------------------------------------------------------------

def test_optimize_returns_sorted_descending():
    from orchard_fem.harvest.optimization import optimize_harvest_excitation

    d_detach = 0.01
    fruits = [_make_fruit("f1", mass=0.05, stiffness=20.0)]
    model = _make_model(fruits)

    freqs = [1.0, 3.0, 8.0, 15.0, 20.0]
    # Make H large at 15 Hz so that's the best frequency
    mags = [0.0001, 0.0002, 0.001, 0.1, 0.05]
    br = _make_batch_result(freqs, mags)

    top = optimize_harvest_excitation(
        model, [br], n_top=3, detachment_displacement_m=d_detach
    )
    assert len(top) <= 3
    # Results should be in descending order of detachment fraction
    for i in range(len(top) - 1):
        assert top[i][1] >= top[i + 1][1]


# ---------------------------------------------------------------------------
# Calibration dataclass validation
# ---------------------------------------------------------------------------

def test_calibration_parameter_invalid_bounds():
    from orchard_fem.calibration.modal_calibration import CalibrationParameter

    with pytest.raises(ValueError, match="initial_value"):
        CalibrationParameter(
            material_id="test",
            initial_value=5.0,   # outside [6, 10]
            lower_bound=6.0,
            upper_bound=10.0,
        )


def test_calibration_config_empty_parameters():
    from orchard_fem.calibration.modal_calibration import ModalCalibrationConfig

    with pytest.raises(ValueError, match="parameter"):
        ModalCalibrationConfig(
            parameters=[],
            target_frequencies_hz=[3.0, 5.0],
        )


def test_calibration_config_n_modes_too_small():
    from orchard_fem.calibration.modal_calibration import (
        CalibrationParameter,
        ModalCalibrationConfig,
    )

    p = CalibrationParameter("mat", 5e9, 1e9, 1e10)
    with pytest.raises(ValueError, match="n_modes"):
        ModalCalibrationConfig(
            parameters=[p],
            target_frequencies_hz=[1.0, 2.0, 3.0],
            n_modes=2,   # < 3 targets
        )


# ---------------------------------------------------------------------------
# ParameterRange validation
# ---------------------------------------------------------------------------

def test_parameter_range_low_equals_high_raises():
    from orchard_pinn.data.parameter_space import ParameterRange

    with pytest.raises(ValueError):
        ParameterRange("x", 5.0, 5.0)


def test_parameter_range_low_greater_raises():
    from orchard_pinn.data.parameter_space import ParameterRange

    with pytest.raises(ValueError):
        ParameterRange("x", 10.0, 5.0)
