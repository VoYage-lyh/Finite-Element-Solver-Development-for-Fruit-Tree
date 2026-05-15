"""Hammer-impact FRF identification and modal-parameter extraction.

Implements the experimental post-processing described in Section 3.4 of the
parameter-uncertainty harvesting paper:

* H1 estimator ``H1(ω) = G_af(ω) / G_ff(ω)`` for a single force–response pair.
* Coherence-based record rejection (``γ² < 0.8`` discarded).
* Averaging of multiple impact records into one final FRF estimate.
* Half-power-bandwidth peak picking → ``(f_r, ζ_r)`` for the lowest ``n`` modes.

Designed to consume raw hammer-test time-series in the canonical ``HammerRecord``
container and emit a clean :class:`FRFEstimate` plus :class:`ModalParameters`
suitable for the Bayesian likelihood in :mod:`bayesian_calibration`.

The numerical core uses ``scipy.signal`` (Welch's method) and is fully testable
without any FEM dependency.

Example::

    from orchard_fem.processing import HammerRecord, estimate_h1_average, identify_modes

    records = [HammerRecord(force, response, fs=1024.0) for ... in raw_data]
    frf = estimate_h1_average(records, gamma_min=0.8)
    modes = identify_modes(frf, n_modes=3)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
#  Data containers
# ────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HammerRecord:
    """One impact-hammer time record.

    Parameters
    ----------
    force:
        Force time series from the impact hammer load cell [N].
    response:
        Response time series from the accelerometer [m/s²] (or any consistent
        physical units — the resulting FRF will have the same units / N).
    sample_rate_hz:
        Sampling frequency [Hz].
    label:
        Optional identifier (e.g. ``"T3_hit_2"``) used in logs / plots.
    """

    force: np.ndarray
    response: np.ndarray
    sample_rate_hz: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.force.shape != self.response.shape:
            raise ValueError(
                f"force and response must have the same shape, "
                f"got {self.force.shape} vs {self.response.shape}."
            )
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive.")


@dataclass(frozen=True)
class FRFEstimate:
    """Frequency-domain estimate of an FRF.

    Parameters
    ----------
    frequencies_hz:
        Frequency vector [Hz].
    H:
        Complex FRF values (response per unit force).
    coherence:
        Magnitude-squared coherence ``γ²(ω) ∈ [0, 1]``; ``None`` if the FRF was
        computed from a single record (no averaging → coherence undefined).
    n_averaged:
        Number of impact records averaged into this estimate.
    """

    frequencies_hz: np.ndarray
    H: np.ndarray
    coherence: np.ndarray | None
    n_averaged: int = 1

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.H)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.H, deg=True)


@dataclass(frozen=True)
class ModalParameters:
    """One half-power-bandwidth mode estimate."""

    frequency_hz: float
    damping_ratio: float          # half-power bandwidth ratio (dimensionless)
    peak_magnitude: float


@dataclass(frozen=True)
class ModalIdentification:
    """Result of multi-mode peak picking on an FRF."""

    modes: list[ModalParameters]
    method: str = "half_power_bandwidth"

    def frequencies_hz(self) -> list[float]:
        return [m.frequency_hz for m in self.modes]

    def damping_ratios(self) -> list[float]:
        return [m.damping_ratio for m in self.modes]


# ────────────────────────────────────────────────────────────────────────────
#  H1 estimator
# ────────────────────────────────────────────────────────────────────────────
def _welch_cross_spectra(
    force: np.ndarray,
    response: np.ndarray,
    fs: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (f, G_ff, G_aa, G_af) using scipy.signal.csd / welch."""
    try:
        from scipy.signal import csd, welch
    except ImportError as exc:
        raise ImportError(
            "scipy is required for FRF estimation. "
            "Install with: pip install scipy"
        ) from exc
    overlap = nperseg // 2
    freqs, g_ff = welch(force, fs=fs, nperseg=nperseg, noverlap=overlap)
    _, g_aa = welch(response, fs=fs, nperseg=nperseg, noverlap=overlap)
    _, g_af = csd(force, response, fs=fs, nperseg=nperseg, noverlap=overlap)
    return freqs, g_ff, g_aa, g_af


def estimate_h1(
    record: HammerRecord,
    *,
    nperseg: int | None = None,
) -> FRFEstimate:
    """Compute the H1 FRF estimate for a single impact record.

    H1 is unbiased when only the response is contaminated by noise — the usual
    assumption for impact testing where the force cell SNR is high.

    Parameters
    ----------
    record:
        One :class:`HammerRecord`.
    nperseg:
        FFT segment length for Welch's method. Defaults to ``min(2048, N)`` so
        short records still produce a non-trivial frequency grid.

    Returns
    -------
    FRFEstimate
        FRF with ``coherence=None`` (single record → no segment averaging
        available for a meaningful γ²).
    """
    n = record.force.shape[0]
    if nperseg is None:
        nperseg = min(2048, n)
    freqs, g_ff, _, g_af = _welch_cross_spectra(
        record.force, record.response, record.sample_rate_hz, nperseg
    )
    h1 = np.where(np.abs(g_ff) > 0.0, g_af / g_ff, 0.0)
    return FRFEstimate(
        frequencies_hz=freqs,
        H=h1.astype(np.complex128),
        coherence=None,
        n_averaged=1,
    )


def coherence_filter(
    records: Sequence[HammerRecord],
    *,
    gamma_min: float = 0.8,
    nperseg: int | None = None,
) -> list[HammerRecord]:
    """Keep records whose input-power-weighted mean coherence exceeds *gamma_min*.

    For impulse-test data the input PSD has near-zero bins outside the hammer's
    excitation band, where ``γ²`` is numerically undefined. We therefore weight
    the per-bin coherence by the input PSD and only count bins above a small
    power threshold, then check that the weighted mean meets *gamma_min*.

    This matches the paper's policy of "5 hits per point, reject any
    γ² < 0.8" interpreted physically — the reject is on the frequency bins
    where the hammer actually delivered energy.

    Parameters
    ----------
    records:
        Iterable of :class:`HammerRecord`.
    gamma_min:
        Minimum acceptable input-power-weighted coherence (default 0.8).
    nperseg:
        Welch segment length; defaults to ``min(2048, N)``.

    Returns
    -------
    list[HammerRecord]
        Subset of input records that pass the coherence gate.
    """
    try:
        from scipy.signal import coherence, welch
    except ImportError as exc:
        raise ImportError("scipy is required.") from exc

    kept: list[HammerRecord] = []
    for rec in records:
        n = rec.force.shape[0]
        nps = nperseg if nperseg is not None else min(2048, n)
        _, gamma_sq = coherence(
            rec.force, rec.response,
            fs=rec.sample_rate_hz, nperseg=nps, noverlap=nps // 2,
        )
        _, g_ff = welch(rec.force, fs=rec.sample_rate_hz, nperseg=nps,
                        noverlap=nps // 2)
        # Mask out bins where the input has effectively no energy
        peak_pow = float(np.max(g_ff)) if g_ff.size else 0.0
        if peak_pow <= 0.0:
            continue
        mask = (g_ff > 0.01 * peak_pow) & np.isfinite(gamma_sq)
        if not mask.any():
            continue
        weights = g_ff[mask]
        weighted_gamma = float(
            np.sum(gamma_sq[mask] * weights) / np.sum(weights)
        )
        if not np.isfinite(weighted_gamma):
            continue
        if weighted_gamma >= gamma_min:
            kept.append(rec)
    return kept


def estimate_h1_average(
    records: Sequence[HammerRecord],
    *,
    gamma_min: float = 0.8,
    nperseg: int | None = None,
) -> FRFEstimate:
    """Coherence-filter then average H1 across multiple impact records.

    Parameters
    ----------
    records:
        Multiple hits at the same input / response point.
    gamma_min:
        Coherence threshold (default 0.8). Set to ``0.0`` to disable.
    nperseg:
        Welch segment length passed through to :func:`estimate_h1`.

    Returns
    -------
    FRFEstimate
        Averaged FRF with multi-record magnitude-squared coherence.

    Raises
    ------
    ValueError
        If no records survive the coherence filter.
    """
    if not records:
        raise ValueError("estimate_h1_average requires at least one record.")
    kept = coherence_filter(records, gamma_min=gamma_min, nperseg=nperseg)
    if not kept:
        raise ValueError(
            f"No records met the coherence threshold (γ² ≥ {gamma_min:.2f}). "
            "Consider lowering gamma_min or inspecting raw signals."
        )

    fs = kept[0].sample_rate_hz
    n = kept[0].force.shape[0]
    nps = nperseg if nperseg is not None else min(2048, n)

    h_sum: np.ndarray | None = None
    g_ff_sum: np.ndarray | None = None
    g_aa_sum: np.ndarray | None = None
    g_af_sum: np.ndarray | None = None
    freqs: np.ndarray | None = None

    for rec in kept:
        if rec.sample_rate_hz != fs:
            raise ValueError("All records must share the same sample rate.")
        f_i, g_ff, g_aa, g_af = _welch_cross_spectra(
            rec.force, rec.response, rec.sample_rate_hz, nps
        )
        if freqs is None:
            freqs = f_i
            g_ff_sum = g_ff.copy()
            g_aa_sum = g_aa.copy()
            g_af_sum = g_af.copy()
        else:
            g_ff_sum = g_ff_sum + g_ff  # type: ignore[operator]
            g_aa_sum = g_aa_sum + g_aa  # type: ignore[operator]
            g_af_sum = g_af_sum + g_af  # type: ignore[operator]

    assert freqs is not None and g_ff_sum is not None
    assert g_aa_sum is not None and g_af_sum is not None
    # Sums are sufficient since H1 = G_af / G_ff and the same scaling applies
    # to numerator and denominator; coherence uses |G_af|² / (G_ff · G_aa).
    h1 = np.where(np.abs(g_ff_sum) > 0.0, g_af_sum / g_ff_sum, 0.0)
    denom = g_ff_sum * g_aa_sum
    gamma_sq = np.where(
        np.abs(denom) > 0.0,
        np.abs(g_af_sum) ** 2 / np.real(denom),
        0.0,
    )
    return FRFEstimate(
        frequencies_hz=freqs,
        H=h1.astype(np.complex128),
        coherence=np.clip(gamma_sq.real, 0.0, 1.0),
        n_averaged=len(kept),
    )


# ────────────────────────────────────────────────────────────────────────────
#  Modal identification — half-power bandwidth
# ────────────────────────────────────────────────────────────────────────────
def half_power_bandwidth(
    frequencies_hz: np.ndarray,
    magnitudes: np.ndarray,
    *,
    peak_index: int,
) -> ModalParameters:
    """Estimate ``(f_r, ζ_r)`` for a single peak via the half-power-bandwidth method.

    The damping ratio is approximated by ``ζ ≈ (f_2 − f_1) / (2 f_r)`` where
    ``f_1, f_2`` are the frequencies at which the magnitude drops to
    ``peak_magnitude / √2``.

    Parameters
    ----------
    frequencies_hz, magnitudes:
        FRF magnitude on a frequency grid.
    peak_index:
        Index of the peak in *magnitudes*.

    Returns
    -------
    ModalParameters
        Frequency, damping ratio, and peak amplitude. ``damping_ratio`` is set
        to ``nan`` if the −3 dB points cannot be bracketed within the array.
    """
    f_r = float(frequencies_hz[peak_index])
    a_peak = float(magnitudes[peak_index])
    half_power = a_peak / np.sqrt(2.0)

    # Scan left from peak for f1
    f1: float | None = None
    for i in range(peak_index - 1, -1, -1):
        if magnitudes[i] <= half_power:
            # linear interpolate between i and i+1
            x0, x1 = magnitudes[i], magnitudes[i + 1]
            denom = x1 - x0
            t = (half_power - x0) / denom if abs(denom) > 1e-15 else 0.0
            f1 = float(frequencies_hz[i] + t * (frequencies_hz[i + 1] - frequencies_hz[i]))
            break

    # Scan right from peak for f2
    f2: float | None = None
    for i in range(peak_index + 1, magnitudes.shape[0]):
        if magnitudes[i] <= half_power:
            x0, x1 = magnitudes[i - 1], magnitudes[i]
            denom = x1 - x0
            t = (half_power - x0) / denom if abs(denom) > 1e-15 else 0.0
            f2 = float(frequencies_hz[i - 1] + t * (frequencies_hz[i] - frequencies_hz[i - 1]))
            break

    if f1 is None or f2 is None or f_r <= 0.0:
        zeta = float("nan")
    else:
        zeta = (f2 - f1) / (2.0 * f_r)

    return ModalParameters(
        frequency_hz=f_r,
        damping_ratio=zeta,
        peak_magnitude=a_peak,
    )


def identify_modes(
    frf: FRFEstimate,
    *,
    n_modes: int = 3,
    frequency_band_hz: tuple[float, float] = (0.5, 30.0),
    min_peak_separation_hz: float = 0.5,
) -> ModalIdentification:
    """Pick the ``n_modes`` largest magnitude peaks within *frequency_band_hz*.

    Uses ``scipy.signal.find_peaks`` with a minimum-separation constraint to
    avoid duplicate detections on adjacent samples; falls back to a manual
    local-maximum scan if scipy is not installed.

    Parameters
    ----------
    frf:
        :class:`FRFEstimate` to peak-pick.
    n_modes:
        Number of modes to return (default 3, matching the paper).
    frequency_band_hz:
        Inclusive analysis band; peaks outside are ignored.
    min_peak_separation_hz:
        Minimum spacing between adjacent peaks (default 0.5 Hz).

    Returns
    -------
    ModalIdentification
        Sorted by frequency ascending.
    """
    freqs = frf.frequencies_hz
    mag = frf.magnitude
    f_min, f_max = frequency_band_hz
    band_mask = (freqs >= f_min) & (freqs <= f_max)
    if not band_mask.any():
        return ModalIdentification(modes=[])

    try:
        from scipy.signal import find_peaks
        df = float(np.median(np.diff(freqs)))
        distance = max(1, int(min_peak_separation_hz / df))
        peak_idx, _ = find_peaks(mag, distance=distance)
        # restrict to band
        peak_idx = peak_idx[band_mask[peak_idx]]
    except ImportError:
        # Fallback: simple local-max scan
        peak_idx = []
        for i in range(1, len(mag) - 1):
            if band_mask[i] and mag[i] > mag[i - 1] and mag[i] > mag[i + 1]:
                peak_idx.append(i)
        peak_idx = np.asarray(peak_idx, dtype=int)

    if peak_idx.size == 0:
        return ModalIdentification(modes=[])

    # Select the n_modes peaks with largest magnitude, then re-sort by frequency
    ordered = peak_idx[np.argsort(mag[peak_idx])[::-1]][:n_modes]
    ordered = ordered[np.argsort(freqs[ordered])]

    modes = [half_power_bandwidth(freqs, mag, peak_index=int(i)) for i in ordered]
    return ModalIdentification(modes=modes)
