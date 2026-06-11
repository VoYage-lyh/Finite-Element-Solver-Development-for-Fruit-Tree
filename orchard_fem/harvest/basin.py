"""Basin-of-attraction and Integrity Factor (IF) for harvest-parameter design.

Python port of the cell-to-cell mapping (CCM) basin analysis from the manuscript
"A reduced-order equivalent nonlinear modeling method for tree-specific vibratory
harvesting parameter determination of Camellia oleifera" (and its MATLAB
reference, ``computeBasinCCM`` / ``computeIntegrityFactorCCM``).

Why this matters
----------------
A softening connection element near resonance is *bistable*: a high-amplitude
working state and a low-amplitude idle state coexist at the same excitation
``(f, F0)``.  Which one the tree settles into from rest depends on the start-up
transient.  The forward FRF/harmonic-balance solve gives *one* steady response;
this module answers the orthogonal question — *from which initial conditions is
the harvesting-effective state actually reached, and how robustly?*

A candidate element is isolated as a single-DOF Duffing oscillator driven by its
equivalent local force ``F0,e`` (manuscript Eq. 22):

    m·x'' + c_lin·x' + c2·|x'|·x' + k_lin·x + k3·x³ = F0·cos(ω t)

The (x, x') plane (a Poincaré-section proxy at the excitation point) is gridded;
each cell is integrated to steady state and classified high/low energy.  The
**Integrity Factor** is the normalized radius of the largest origin-centred disk
fully contained in the high-energy basin — i.e. the tolerance of the working
state to start-up disturbance (manuscript Eq. 36).

This is intentionally SDOF and cheap (no whole-tree integration); it composes
with the whole-tree forward solve, which supplies ``F0,e`` per element.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DuffingElement:
    """Reduced single-DOF Duffing parameters of a connection element.

    Parameters
    ----------
    mass_eq:
        Equivalent modal mass ``m_eq,e`` [kg].
    k_lin / c_lin:
        Linear stiffness [N/m] and damping [N·s/m].
    k3:
        Cubic stiffness [N/m³] (``> 0`` hardening, ``< 0`` softening).
    c2:
        Amplitude-dependent (quadratic-velocity) damping [N·s²/m²].
    """

    mass_eq: float
    k_lin: float
    c_lin: float
    k3: float = 0.0
    c2: float = 0.0


@dataclass(frozen=True)
class BasinResult:
    """Outcome of a CCM basin analysis at one ``(frequency, force)`` point.

    Parameters
    ----------
    integrity_factor:
        IF ∈ [0, 1] — normalized inscribed-disk radius of the high-energy basin.
    high_energy_ratio:
        Fraction of the analysis domain in the high-energy state ∈ [0, 1].
    basin_map:
        ``Nc × Nc`` array, 1 = high-energy (working) state, 0 = low-energy.
    x_grid / v_grid:
        Displacement [m] and velocity [m/s] grid axes.
    amplitude_threshold:
        Steady-amplitude threshold separating the two states [m].
    is_bistable:
        Whether two distinct attractors were detected.
    """

    integrity_factor: float
    high_energy_ratio: float
    basin_map: np.ndarray
    x_grid: np.ndarray
    v_grid: np.ndarray
    amplitude_threshold: float
    is_bistable: bool


def duffing_rhs(t: float, y, element: DuffingElement, omega: float, force: float):
    """Right-hand side of the driven Duffing oscillator (state ``y = [x, v]``)."""
    x, v = y
    accel = (
        force * np.cos(omega * t)
        - element.c_lin * v
        - element.c2 * abs(v) * v
        - element.k_lin * x
        - element.k3 * x**3
    ) / element.mass_eq
    return (v, accel)


def steady_amplitude(
    element: DuffingElement,
    frequency_hz: float,
    force: float,
    y0,
    *,
    n_periods: int = 30,
    steady_fraction: float = 1.0 / 3.0,
) -> float:
    """Steady-state half peak-to-peak amplitude from one initial condition [m].

    Integrates over ``n_periods`` excitation periods and measures
    ``(max - min) / 2`` of the last ``steady_fraction`` of the trajectory.
    Returns 0.0 if integration fails.
    """
    omega = 2.0 * np.pi * frequency_hz
    t_end = n_periods * (2.0 * np.pi / omega)
    try:
        sol = solve_ivp(
            duffing_rhs, (0.0, t_end), list(y0),
            args=(element, omega, force),
            rtol=1e-6, atol=1e-9, max_step=0.01 / frequency_hz, dense_output=False,
        )
        if not sol.success or sol.y.shape[1] < 4:
            return 0.0
        x = sol.y[0]
        start = int(round(len(x) * (1.0 - steady_fraction)))
        x_ss = x[start:]
        return 0.5 * float(np.max(x_ss) - np.min(x_ss))
    except Exception:
        return 0.0


def _split_threshold(amplitudes: np.ndarray) -> tuple[bool, float]:
    """Detect bistability and a separating amplitude via the largest 1-D gap.

    Dependency-free stand-in for the MATLAB ``kmeans(2)``: sort the sampled
    steady amplitudes, split at the largest gap, and call the system bistable
    when the two cluster means differ by ≥ 3×.
    """
    amps = np.sort(np.asarray(amplitudes, dtype=float))
    if amps.size <= 1 or np.ptp(amps) <= 1e-9:
        a_res = float(amps.max()) if amps.size else 0.0
        return False, a_res * 0.55  # threshold below the single attractor
    gaps = np.diff(amps)
    k = int(np.argmax(gaps))
    low, high = amps[: k + 1], amps[k + 1:]
    a_low, a_high = float(low.mean()), float(high.mean())
    is_bistable = a_high >= 3.0 * max(a_low, 1e-10)
    return is_bistable, 0.5 * (a_low + a_high)


def integrity_factor(
    basin_map: np.ndarray, x_grid: np.ndarray, v_grid: np.ndarray, *, n_theta: int = 72
) -> float:
    """Normalized inscribed-disk radius of the high-energy basin, IF ∈ [0, 1].

    The disk is centred on the **origin** (static equilibrium, where the tree
    starts from rest), per manuscript Eq. 36 — so IF measures tolerance to
    start-up disturbance, and ``IF = 0`` if the origin is not in the basin.
    (This is the physically intended definition; the MATLAB reference centred
    the disk on the basin centroid instead.)

    The search is in **normalized** coordinates ``ξ = x/Amax``, ``η = v/Vmax``
    (Eq. 37), so the circle is isotropic and the domain is ``[-1, 1]²``.
    """
    if basin_map.size == 0 or not np.any(basin_map == 1):
        return 0.0
    amax = float(np.max(np.abs(x_grid))) or 1.0
    vmax = float(np.max(np.abs(v_grid))) or 1.0
    xi = x_grid / amax
    eta = v_grid / vmax

    # Origin must itself be in the high-energy basin, else IF = 0.
    iz_x = int(np.abs(xi).argmin())
    iz_v = int(np.abs(eta).argmin())
    if basin_map[iz_v, iz_x] != 1:
        return 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tol = 0.5 * min(xi[1] - xi[0], eta[1] - eta[0]) if xi.size > 1 else 1e-3

    def disk_inside(radius: float) -> bool:
        px = radius * cos_t
        py = radius * sin_t
        if np.any(np.abs(px) > 1.0) or np.any(np.abs(py) > 1.0):
            return False
        ix = np.abs(xi[None, :] - px[:, None]).argmin(axis=1)
        iv = np.abs(eta[None, :] - py[:, None]).argmin(axis=1)
        return bool(np.all(basin_map[iv, ix] == 1))

    r_low, r_high = 0.0, float(np.hypot(1.0, 1.0))
    while (r_high - r_low) > tol:
        r_mid = 0.5 * (r_low + r_high)
        if disk_inside(r_mid):
            r_low = r_mid
        else:
            r_high = r_mid
    return min(r_low, 1.0)


def compute_basin_ccm(
    element: DuffingElement,
    frequency_hz: float,
    force: float,
    *,
    n_cells: int = 100,
    amax_m: float = 15e-3,
    vmax_m_s: float = 400e-3,
    n_periods: int = 30,
) -> BasinResult:
    """Cell-to-cell mapping basin analysis + Integrity Factor at one ``(f, F0)``.

    Parameters
    ----------
    element:
        Reduced Duffing parameters of the candidate connection element.
    frequency_hz / force:
        Excitation frequency [Hz] and equivalent local force amplitude ``F0,e`` [N].
    n_cells:
        Grid resolution per axis (manuscript suggests 100–200; use less for speed).
    amax_m / vmax_m_s:
        Half-widths of the displacement/velocity analysis domain.
    n_periods:
        Excitation periods integrated per cell before reading steady state.

    Returns
    -------
    BasinResult
    """
    x_grid = np.linspace(-amax_m, amax_m, n_cells)
    v_grid = np.linspace(-vmax_m_s, vmax_m_s, n_cells)

    # --- detect attractors / classification threshold ---
    n_samples = 12
    half = n_samples // 2
    test_x0 = np.concatenate([np.linspace(-amax_m * 0.8, amax_m * 0.8, half), np.zeros(half)])
    test_v0 = np.concatenate([np.zeros(half), np.linspace(-vmax_m_s * 0.8, vmax_m_s * 0.8, half)])
    probe = np.array([
        steady_amplitude(element, frequency_hz, force, (x0, v0), n_periods=n_periods)
        for x0, v0 in zip(test_x0, test_v0)
    ])
    is_bistable, threshold = _split_threshold(probe)

    # --- CCM main loop ---
    basin_map = np.zeros((n_cells, n_cells), dtype=int)
    for i, v0 in enumerate(v_grid):
        for j, x0 in enumerate(x_grid):
            a_ss = steady_amplitude(element, frequency_hz, force, (x0, v0), n_periods=n_periods)
            if a_ss >= threshold:
                basin_map[i, j] = 1

    high_ratio = float(np.count_nonzero(basin_map)) / basin_map.size
    if_value = integrity_factor(basin_map, x_grid, v_grid)

    return BasinResult(
        integrity_factor=if_value,
        high_energy_ratio=high_ratio,
        basin_map=basin_map,
        x_grid=x_grid,
        v_grid=v_grid,
        amplitude_threshold=threshold,
        is_bistable=is_bistable,
    )
