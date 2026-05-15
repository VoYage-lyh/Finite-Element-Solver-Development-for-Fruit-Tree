"""Fixed-frequency excitation validation against the posterior predictive.

Implements Section 3.5 + Table 5 of the parameter-uncertainty harvesting paper:
the calibrated posterior is propagated through the FE forward operator at
fixed excitation frequencies (e.g. 6, 10, 14, 18, 22 Hz) and the posterior
predictive RMS distribution is compared to the measured steady-state RMS.
The reported coverage rate (fraction of measurements falling inside the
posterior 90 % CI) is the headline check that the uncertainty quantification
is calibrated rather than optimistic.

This module is intentionally agnostic of which forward operator is used —
the caller supplies either the per-frequency posterior predictive RMS samples
or the simulator function and a list of parameter dicts to evaluate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class FixedFrequencyRecord:
    """One measured steady-state RMS observation at a fixed excitation frequency.

    Parameters
    ----------
    frequency_hz:
        Drive frequency [Hz].
    measured_rms:
        Steady-state RMS amplitude on the observation channel
        (e.g. ``[m/s²]`` for accelerometer signals).
    sample_rate_hz, transient_skip_s, total_hold_s:
        Provenance metadata so ``compute_steady_state_rms`` can be recomputed
        from raw data if needed. Optional.
    label:
        Free-form identifier (e.g. ``"T3_target_branch"``).
    """

    frequency_hz: float
    measured_rms: float
    sample_rate_hz: float | None = None
    transient_skip_s: float | None = None
    total_hold_s: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive.")
        if self.measured_rms < 0.0:
            raise ValueError("measured_rms must be non-negative.")


@dataclass(frozen=True)
class CoverageReport:
    """Outcome of running posterior predictive coverage on a set of records.

    Parameters
    ----------
    nominal_coverage:
        Nominal probability of the posterior credible interval (e.g. 0.90).
    empirical_coverage:
        Fraction of measurements falling inside the predicted interval. Should
        equal *nominal_coverage* if the posterior is well calibrated.
    record_results:
        Per-record details (frequency, measured, median, CI bounds, hit/miss).
    """

    nominal_coverage: float
    empirical_coverage: float
    record_results: list[dict]

    def n_hits(self) -> int:
        return sum(1 for r in self.record_results if r["inside_ci"])

    def n_records(self) -> int:
        return len(self.record_results)

    def summary_table(self) -> str:
        """Return a human-readable plain-text table (Table-5 style)."""
        lines = [
            "freq[Hz]  measured     median       90%CI",
            "-" * 56,
        ]
        for r in self.record_results:
            mark = " " if r["inside_ci"] else "*"
            lines.append(
                f"{r['frequency_hz']:7.2f}  {r['measured_rms']:9.3f}  "
                f"{r['posterior_median']:9.3f}  "
                f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]{mark}"
            )
        lines.append("-" * 56)
        lines.append(
            f"Coverage: {self.n_hits()}/{self.n_records()} "
            f"(empirical {self.empirical_coverage:.2f} vs nominal "
            f"{self.nominal_coverage:.2f})"
        )
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
#  Steady-state RMS extraction (from raw time-series)
# ────────────────────────────────────────────────────────────────────────────
def compute_steady_state_rms(
    time_series: np.ndarray,
    *,
    sample_rate_hz: float,
    transient_skip_s: float = 0.5,
) -> float:
    """Discard the leading transient and return the RMS of the remainder.

    Parameters
    ----------
    time_series:
        1-D array of observation samples.
    sample_rate_hz:
        Acquisition sample rate.
    transient_skip_s:
        Seconds to discard at the start of the record. Default 0.5 s, which is
        usually well past the build-up transient for tree-canopy harmonic
        excitation in the 5–25 Hz band.

    Returns
    -------
    float
        Root-mean-square amplitude of the steady-state segment.
    """
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    skip = int(transient_skip_s * sample_rate_hz)
    if skip >= time_series.size:
        raise ValueError("transient_skip_s exceeds record length.")
    segment = np.asarray(time_series[skip:], dtype=float)
    return float(np.sqrt(np.mean(segment * segment)))


# ────────────────────────────────────────────────────────────────────────────
#  Posterior predictive coverage
# ────────────────────────────────────────────────────────────────────────────
def check_posterior_coverage(
    records: Sequence[FixedFrequencyRecord],
    posterior_predictive_rms: dict[float, np.ndarray],
    *,
    nominal_coverage: float = 0.90,
) -> CoverageReport:
    """Compare measured RMS to posterior predictive RMS samples.

    Parameters
    ----------
    records:
        Measured fixed-frequency RMS observations.
    posterior_predictive_rms:
        Map ``frequency_hz -> array_of_RMS_samples`` (typically one entry per
        record, with K samples drawn by re-running the forward operator on
        each posterior θ sample). Frequencies are matched to records by a
        small tolerance (1 mHz).
    nominal_coverage:
        Probability of the credible interval (default 0.90 → 5%/95% quantiles).

    Returns
    -------
    CoverageReport
    """
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must lie in (0, 1).")
    lo_q = (1.0 - nominal_coverage) / 2.0
    hi_q = 1.0 - lo_q
    results = []
    n_hit = 0
    for rec in records:
        # Find matching frequency in the predictive samples (tolerance 1 mHz)
        match = None
        for f_key in posterior_predictive_rms.keys():
            if abs(f_key - rec.frequency_hz) <= 1.0e-3:
                match = f_key
                break
        if match is None:
            raise KeyError(
                f"No posterior predictive samples for f={rec.frequency_hz} Hz; "
                f"available: {sorted(posterior_predictive_rms.keys())}"
            )
        samples = np.asarray(posterior_predictive_rms[match], dtype=float)
        if samples.size == 0:
            raise ValueError(f"Empty posterior predictive sample at {match} Hz.")
        median = float(np.median(samples))
        ci_low = float(np.quantile(samples, lo_q))
        ci_high = float(np.quantile(samples, hi_q))
        inside = ci_low <= rec.measured_rms <= ci_high
        n_hit += int(inside)
        results.append({
            "frequency_hz": rec.frequency_hz,
            "label": rec.label,
            "measured_rms": rec.measured_rms,
            "posterior_median": median,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "inside_ci": inside,
        })
    return CoverageReport(
        nominal_coverage=nominal_coverage,
        empirical_coverage=n_hit / max(1, len(records)),
        record_results=results,
    )


def propagate_posterior_to_fixed_freq(
    posterior_samples: list[dict[str, float]],
    forward_rms: Callable[[dict[str, float], float], float],
    frequencies_hz: Sequence[float],
) -> dict[float, np.ndarray]:
    """Run the forward operator on every (posterior sample, frequency) pair.

    Use this to build the ``posterior_predictive_rms`` dictionary required by
    :func:`check_posterior_coverage` when the caller has only the raw
    posterior θ samples rather than pre-computed predictive RMS samples.

    Parameters
    ----------
    posterior_samples:
        List of K parameter dicts (typically thinned from a
        :class:`PosteriorResult`).
    forward_rms:
        Callable ``(params, freq_hz) -> RMS``. Should return the predicted
        steady-state RMS on the observation channel of interest.
    frequencies_hz:
        Fixed excitation frequencies to evaluate.

    Returns
    -------
    dict[float, np.ndarray]
        ``{f_hz: array(K,)}`` ready for :func:`check_posterior_coverage`.
    """
    out: dict[float, np.ndarray] = {}
    for f in frequencies_hz:
        samples = np.empty(len(posterior_samples), dtype=float)
        for k, params in enumerate(posterior_samples):
            try:
                samples[k] = float(forward_rms(params, float(f)))
            except Exception:
                samples[k] = np.nan
        out[float(f)] = samples
    return out


__all__ = [
    "CoverageReport",
    "FixedFrequencyRecord",
    "check_posterior_coverage",
    "compute_steady_state_rms",
    "propagate_posterior_to_fixed_freq",
]
