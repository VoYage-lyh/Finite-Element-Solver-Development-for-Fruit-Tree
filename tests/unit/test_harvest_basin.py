"""Tests for basin-of-attraction / Integrity Factor (no FEniCSx required)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from orchard_fem.harvest.basin import (
    DuffingElement,
    compute_basin_ccm,
    integrity_factor,
    steady_amplitude,
)


# ---------------------------------------------------------------------------
# Integrity factor geometry (deterministic, synthetic basin maps)
# ---------------------------------------------------------------------------

def _grids(n=101, amax=1.0, vmax=1.0):
    return np.linspace(-amax, amax, n), np.linspace(-vmax, vmax, n)


def test_if_zero_when_no_high_basin():
    x, v = _grids()
    basin = np.zeros((x.size, x.size), dtype=int)
    assert integrity_factor(basin, x, v) == 0.0


def test_if_one_when_full_basin():
    x, v = _grids()
    basin = np.ones((x.size, x.size), dtype=int)
    # Whole normalized domain high → inscribed disk fills it → IF ≈ 1.
    assert integrity_factor(basin, x, v) == pytest.approx(1.0, abs=0.05)


def test_if_matches_centered_disk_radius():
    # High-energy region = a centered disk of normalized radius 0.5 → IF ≈ 0.5.
    n = 201
    x, v = _grids(n)
    xi = x / np.max(np.abs(x))
    eta = v / np.max(np.abs(v))
    XX, VV = np.meshgrid(xi, eta)
    basin = (np.hypot(XX, VV) <= 0.5).astype(int)
    assert integrity_factor(basin, x, v) == pytest.approx(0.5, abs=0.03)


def test_if_zero_for_offset_basin_not_covering_origin():
    # High region is an off-origin blob; origin-centred disk can't grow → small IF.
    n = 151
    x, v = _grids(n)
    xi = x / np.max(np.abs(x))
    eta = v / np.max(np.abs(v))
    XX, VV = np.meshgrid(xi, eta)
    basin = (np.hypot(XX - 0.6, VV - 0.6) <= 0.2).astype(int)
    assert integrity_factor(basin, x, v) < 0.1


# ---------------------------------------------------------------------------
# Steady amplitude physics: linear SDOF matches |H(ω)|·F0
# ---------------------------------------------------------------------------

def test_steady_amplitude_linear_resonance():
    m, k, c = 1.0, (2 * math.pi * 5.0) ** 2, 2.0   # ω_n = 5 Hz, light damping
    elem = DuffingElement(mass_eq=m, k_lin=k, c_lin=c, k3=0.0, c2=0.0)
    f_hz, F0 = 5.0, 1.0
    omega = 2 * math.pi * f_hz
    # Analytic steady amplitude |H| = F0 / sqrt((k - mω²)² + (cω)²)
    expected = F0 / math.sqrt((k - m * omega**2) ** 2 + (c * omega) ** 2)
    amp = steady_amplitude(elem, f_hz, F0, (0.0, 0.0), n_periods=120)
    assert amp == pytest.approx(expected, rel=0.05)


def test_steady_amplitude_off_resonance_smaller():
    m, k, c = 1.0, (2 * math.pi * 5.0) ** 2, 2.0
    elem = DuffingElement(mass_eq=m, k_lin=k, c_lin=c)
    at_res = steady_amplitude(elem, 5.0, 1.0, (0.0, 0.0), n_periods=120)
    off_res = steady_amplitude(elem, 9.0, 1.0, (0.0, 0.0), n_periods=120)
    assert off_res < at_res


# ---------------------------------------------------------------------------
# End-to-end CCM on a small grid (sane structure, IF in [0, 1])
# ---------------------------------------------------------------------------

def test_compute_basin_ccm_runs_and_is_sane():
    m, k, c = 1.0, (2 * math.pi * 5.0) ** 2, 3.0
    elem = DuffingElement(mass_eq=m, k_lin=k, c_lin=c, k3=0.0, c2=0.0)
    res = compute_basin_ccm(
        elem, frequency_hz=5.0, force=2.0,
        n_cells=11, amax_m=20e-3, vmax_m_s=0.5, n_periods=20,
    )
    assert res.basin_map.shape == (11, 11)
    assert 0.0 <= res.integrity_factor <= 1.0
    assert 0.0 <= res.high_energy_ratio <= 1.0
    # A driven linear element reaches the same (single) attractor from any IC,
    # so the domain is uniformly classified → no spurious bistability.
    assert res.is_bistable is False
