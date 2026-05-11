"""FRF measurement data loading and comparison.

Loads frequency-response function (FRF) data recorded by accelerometer + impact
hammer instruments, stored as CSV.  Supports two common export formats:

* **Complex format**: columns ``frequency_hz, real, imag``
  → magnitude = √(real² + imag²)
* **Polar / dB format**: columns ``frequency_hz, magnitude_db, phase_deg``
  → magnitude (linear) = 10^(dB / 20)

The measured FRF is expressed in *inertance* units [m/s²/N], which is the natural
output of an accelerometer-based measurement chain.  The FEM simulation outputs
*compliance* [m/N]; use :func:`simulate_to_inertance` to convert before calling
:func:`compare_frf`.

Example::

    from orchard_fem.io.measurement import (
        load_measured_frf_csv,
        compare_frf,
        simulate_to_inertance,
    )

    measured = load_measured_frf_csv("data/apple_branch_frf.csv", label="branch_A")
    # ... run FEM and obtain FrequencyResponseResult ...
    comparison = compare_frf(measured, fem_result)
    print(f"MAC = {comparison.mac_value:.4f}")
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchard_fem.dynamics.frequency_response import FrequencyResponseResult


@dataclass(frozen=True)
class MeasuredFRF:
    """Experimentally measured FRF from accelerometer + impact hammer.

    Parameters
    ----------
    frequencies_hz:
        Frequency grid [Hz].
    magnitudes:
        FRF magnitude [m/s²/N] (inertance).
    phases_deg:
        Phase angle [°].  ``None`` if not available in the source CSV.
    label:
        Human-readable identifier (e.g. branch ID or measurement point).
    """

    frequencies_hz: list[float]
    magnitudes: list[float]
    phases_deg: list[float] | None
    label: str = ""


@dataclass(frozen=True)
class FRFComparison:
    """Side-by-side measured vs. simulated FRF at common frequency points.

    Parameters
    ----------
    frequencies_hz:
        Frequency grid used for comparison (from *measured*).
    measured_magnitude:
        Measured inertance [m/s²/N].
    simulated_magnitude:
        Simulated inertance [m/s²/N] after converting compliance → inertance
        and interpolating to the measured frequency grid.
    mac_value:
        Modal Assurance Criterion computed on the magnitude vectors:
        ``MAC = (aᵀb)² / (aᵀa × bᵀb)`` ∈ [0, 1].
        Values close to 1 indicate good shape correlation.
    """

    frequencies_hz: list[float]
    measured_magnitude: list[float]
    simulated_magnitude: list[float]
    mac_value: float


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _detect_format(headers: list[str]) -> str:
    """Return ``"complex"`` or ``"polar"`` based on column names."""
    lower = [h.lower() for h in headers]
    has_real = any("real" in h for h in lower)
    has_imag = any("imag" in h for h in lower)
    if has_real and has_imag:
        return "complex"
    has_db = any("db" in h for h in lower)
    has_mag = any("mag" in h for h in lower)
    if has_db or has_mag:
        return "polar"
    raise ValueError(
        f"Cannot determine FRF CSV format from headers {headers!r}. "
        "Expected columns containing 'real'/'imag' (complex) or 'db'/'mag' (polar)."
    )


def _find_col(headers: list[str], *candidates: str) -> int:
    """Return the index of the first header that matches any candidate (case-insensitive)."""
    lower = [h.lower() for h in headers]
    for candidate in candidates:
        for i, h in enumerate(lower):
            if candidate in h:
                return i
    raise ValueError(
        f"Could not find a column matching {candidates!r} in headers {headers!r}."
    )


def load_measured_frf_csv(path: str | Path, *, label: str = "") -> MeasuredFRF:
    """Load a measured FRF from a CSV file.

    Auto-detects whether the file uses complex (``real``/``imag``) or
    polar/dB (``magnitude_db``/``phase_deg``) format.

    Parameters
    ----------
    path:
        Path to the CSV file.
    label:
        Optional descriptive label stored in the returned :class:`MeasuredFRF`.

    Returns
    -------
    MeasuredFRF
        Frequencies [Hz], magnitudes [m/s²/N], optional phases [°].
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        headers = [h.strip() for h in headers]

        fmt = _detect_format(headers)
        freq_col = _find_col(headers, "freq")

        frequencies: list[float] = []
        magnitudes: list[float] = []
        phases: list[float] | None = None

        if fmt == "complex":
            real_col = _find_col(headers, "real")
            imag_col = _find_col(headers, "imag")
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                f = float(row[freq_col])
                re = float(row[real_col])
                im = float(row[imag_col])
                frequencies.append(f)
                magnitudes.append(math.hypot(re, im))
        else:
            db_col = _find_col(headers, "db", "mag")
            phase_col: int | None = None
            try:
                phase_col = _find_col(headers, "phase")
            except ValueError:
                pass

            if phase_col is not None:
                phases = []

            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                f = float(row[freq_col])
                db_val = float(row[db_col])
                mag_linear = 10.0 ** (db_val / 20.0)
                frequencies.append(f)
                magnitudes.append(mag_linear)
                if phases is not None and phase_col is not None:
                    phases.append(float(row[phase_col]))

    return MeasuredFRF(
        frequencies_hz=frequencies,
        magnitudes=magnitudes,
        phases_deg=phases,
        label=label,
    )


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def simulate_to_inertance(
    frequencies_hz: list[float],
    compliance_magnitudes: list[float],
) -> list[float]:
    """Convert FEM compliance [m/N] to inertance [m/s²/N].

    The relationship is ``H_inertance = ω² × H_compliance`` (absolute value;
    the sign is absorbed into the magnitude).

    Parameters
    ----------
    frequencies_hz:
        Frequency grid [Hz].
    compliance_magnitudes:
        FEM compliance magnitudes [m/N].

    Returns
    -------
    list[float]
        Inertance magnitudes [m/s²/N].
    """
    tau = 2.0 * math.pi
    return [
        (tau * f) ** 2 * mag
        for f, mag in zip(frequencies_hz, compliance_magnitudes)
    ]


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def _interp1d(x_query: float, x_data: list[float], y_data: list[float]) -> float:
    """Linear interpolation (clamps to boundary outside range)."""
    if x_query <= x_data[0]:
        return y_data[0]
    if x_query >= x_data[-1]:
        return y_data[-1]
    for i in range(len(x_data) - 1):
        if x_data[i] <= x_query <= x_data[i + 1]:
            t = (x_query - x_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + t * (y_data[i + 1] - y_data[i])
    return y_data[-1]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_frf(measured: MeasuredFRF, result: "FrequencyResponseResult") -> FRFComparison:
    """Compare a measured FRF against a simulated :class:`FrequencyResponseResult`.

    The simulation compliance is converted to inertance and interpolated onto the
    measured frequency grid.  The Modal Assurance Criterion is computed on the
    magnitude vectors.

    Parameters
    ----------
    measured:
        Measured inertance FRF.
    result:
        FEM frequency-response result (compliance [m/N]).

    Returns
    -------
    FRFComparison
    """
    sim_freqs = [p.frequency_hz for p in result.points]
    sim_compliance = [p.excitation_response_magnitude for p in result.points]
    sim_inertance = simulate_to_inertance(sim_freqs, sim_compliance)

    simulated_interp = [
        _interp1d(f, sim_freqs, sim_inertance) for f in measured.frequencies_hz
    ]

    a = measured.magnitudes
    b = simulated_interp
    dot_ab = sum(ai * bi for ai, bi in zip(a, b))
    dot_aa = sum(ai * ai for ai in a)
    dot_bb = sum(bi * bi for bi in b)
    denom = dot_aa * dot_bb
    mac = (dot_ab ** 2) / denom if denom > 0.0 else 0.0

    return FRFComparison(
        frequencies_hz=list(measured.frequencies_hz),
        measured_magnitude=list(measured.magnitudes),
        simulated_magnitude=simulated_interp,
        mac_value=mac,
    )
