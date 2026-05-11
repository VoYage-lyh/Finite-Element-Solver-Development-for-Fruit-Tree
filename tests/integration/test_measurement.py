"""Tests for FRF measurement CSV loading and comparison (no FEniCSx required)."""
from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# load_measured_frf_csv — complex format
# ---------------------------------------------------------------------------

def test_load_complex_format(tmp_path):
    from orchard_fem.io.measurement import load_measured_frf_csv

    csv_path = tmp_path / "frf_complex.csv"
    _write_csv(
        csv_path,
        "frequency_hz,real,imag\n"
        "1.0,3.0,4.0\n"
        "2.0,0.0,5.0\n",
    )
    frf = load_measured_frf_csv(csv_path, label="test")
    assert frf.label == "test"
    assert frf.frequencies_hz == [1.0, 2.0]
    assert math.isclose(frf.magnitudes[0], 5.0)   # hypot(3,4)
    assert math.isclose(frf.magnitudes[1], 5.0)   # hypot(0,5)
    assert frf.phases_deg is None


def test_load_polar_db_format(tmp_path):
    from orchard_fem.io.measurement import load_measured_frf_csv

    csv_path = tmp_path / "frf_db.csv"
    _write_csv(
        csv_path,
        "frequency_hz,magnitude_db,phase_deg\n"
        "5.0,20.0,45.0\n"
        "10.0,0.0,-90.0\n",
    )
    frf = load_measured_frf_csv(csv_path)
    assert frf.frequencies_hz == [5.0, 10.0]
    assert math.isclose(frf.magnitudes[0], 10.0)   # 10^(20/20)
    assert math.isclose(frf.magnitudes[1], 1.0)    # 10^(0/20)
    assert frf.phases_deg is not None
    assert frf.phases_deg[0] == 45.0
    assert frf.phases_deg[1] == -90.0


def test_load_unknown_format_raises(tmp_path):
    from orchard_fem.io.measurement import load_measured_frf_csv

    csv_path = tmp_path / "bad.csv"
    _write_csv(csv_path, "frequency_hz,amplitude\n1.0,2.0\n")
    with pytest.raises(ValueError, match="format"):
        load_measured_frf_csv(csv_path)


def test_load_skips_comment_lines(tmp_path):
    from orchard_fem.io.measurement import load_measured_frf_csv

    csv_path = tmp_path / "frf_comments.csv"
    _write_csv(
        csv_path,
        "frequency_hz,real,imag\n"
        "# this is a comment\n"
        "3.0,3.0,4.0\n",
    )
    frf = load_measured_frf_csv(csv_path)
    assert len(frf.frequencies_hz) == 1
    assert frf.frequencies_hz[0] == 3.0


# ---------------------------------------------------------------------------
# simulate_to_inertance
# ---------------------------------------------------------------------------

def test_simulate_to_inertance_unit():
    from orchard_fem.io.measurement import simulate_to_inertance

    freqs = [1.0]
    compliance = [1.0]
    inertance = simulate_to_inertance(freqs, compliance)
    expected = (2.0 * math.pi) ** 2   # ω² × H at 1 Hz
    assert math.isclose(inertance[0], expected, rel_tol=1e-9)


def test_simulate_to_inertance_zero_compliance():
    from orchard_fem.io.measurement import simulate_to_inertance

    result = simulate_to_inertance([5.0], [0.0])
    assert result[0] == 0.0


# ---------------------------------------------------------------------------
# compare_frf
# ---------------------------------------------------------------------------

def _make_dummy_result(freqs, mags):
    """Build a minimal FrequencyResponseResult for testing."""
    from orchard_fem.dynamics.frequency_response import (
        FrequencyResponsePoint,
        FrequencyResponseResult,
    )

    return FrequencyResponseResult(
        observation_names=["test"],
        points=[
            FrequencyResponsePoint(
                frequency_hz=f,
                excitation_response_magnitude=m,
                observation_magnitudes=[m],
            )
            for f, m in zip(freqs, mags)
        ],
    )


def test_compare_frf_perfect_match():
    from orchard_fem.io.measurement import (
        MeasuredFRF,
        compare_frf,
        simulate_to_inertance,
    )

    freqs = [1.0, 2.0, 3.0, 4.0, 5.0]
    compliance = [1e-4] * 5
    inertance = simulate_to_inertance(freqs, compliance)

    measured = MeasuredFRF(
        frequencies_hz=freqs,
        magnitudes=inertance,
        phases_deg=None,
    )
    result = _make_dummy_result(freqs, compliance)
    comparison = compare_frf(measured, result)

    assert comparison.mac_value == pytest.approx(1.0, abs=1e-9)
    assert len(comparison.frequencies_hz) == len(freqs)


def test_compare_frf_mac_between_zero_and_one():
    from orchard_fem.io.measurement import MeasuredFRF, compare_frf

    freqs = [1.0, 2.0, 3.0, 4.0]
    measured = MeasuredFRF(
        frequencies_hz=freqs,
        magnitudes=[1.0, 2.0, 1.0, 0.5],
        phases_deg=None,
    )
    # Different shape → MAC < 1
    result = _make_dummy_result(freqs, [0.5e-4, 0.5e-4, 2e-4, 3e-4])
    comparison = compare_frf(measured, result)
    assert 0.0 <= comparison.mac_value <= 1.0


def test_compare_frf_interpolates_on_finer_grid():
    from orchard_fem.io.measurement import MeasuredFRF, compare_frf

    sim_freqs = [1.0, 3.0, 5.0]
    measured_freqs = [1.0, 2.0, 3.0, 4.0, 5.0]
    measured = MeasuredFRF(
        frequencies_hz=measured_freqs,
        magnitudes=[1.0] * 5,
        phases_deg=None,
    )
    result = _make_dummy_result(sim_freqs, [1.0, 1.0, 1.0])
    comparison = compare_frf(measured, result)
    # Should not raise; simulated magnitude interpolated to 5 points
    assert len(comparison.simulated_magnitude) == 5
