"""Process hammer-impact test data and plot curves comparable to FEM output.

Inputs (4 CSV files per test, sharing a ``<prefix>`` such as ``tree1_p1``):

  - ``<prefix>_force.csv``   Hammer force log. ASCII, header ``Time(s),CH 1,``.
                              Time is relative seconds from recording start.
                              Sampling is 1 ms uniform (1 kHz).
  - ``<prefix>_root.csv``    3-axis accel at the root station.
  - ``<prefix>_mid.csv``     3-axis accel at the mid station.
  - ``<prefix>_tip.csv``     3-axis accel at the tip station.

  The three accel files are GBK-encoded with header
  ``时间,X加速度,Y加速度,Z加速度,`` and time as wall-clock ``HH_MM_SS_.mmm``.
  Samples are non-uniform (multiple rows often share one millisecond, then
  a gap to the next burst); we resample to a uniform grid.

  Impact order along the branch is root -> mid -> tip.

Outputs (under ``--out-dir``):

  - ``time_history_force.png``       Full force trace, with detected impacts.
  - ``time_history_<station>.png``   3-axis accel time history.
  - ``frf_<station>.png``            |Accel/Force| FRF magnitude, X/Y/Z.
  - ``resampled.csv``                Force + accel on a shared uniform time axis,
                                     columns named to match the simulation's
                                     ``obs_<branch>_<station>_<component>_accel_ms2``
                                     convention so overlays are easy.

Run::

    python scripts/process_hammer_test.py --prefix trees/tree1_p1 \
        --out-dir results/hammer_test/tree1_p1
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


STATIONS = ("root", "mid", "tip")
COMPONENTS = ("X", "Y", "Z")

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "cache" / "hammer_test"


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_force_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time_s, force)`` from a hammer force CSV."""
    times: list[float] = []
    forces: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # Tolerate the trailing empty column produced by the logger.
        if "Time(s)" not in header[0]:
            raise ValueError(f"Unexpected force header in {path}: {header!r}")
        for row in reader:
            if len(row) < 2 or row[0] == "" or row[1] == "":
                continue
            times.append(float(row[0]))
            forces.append(float(row[1]))
    return np.asarray(times, dtype=float), np.asarray(forces, dtype=float)


def _parse_clock_seconds(stamp: str) -> float:
    """Parse ``HH_MM_SS_.mmm`` into seconds-of-day (float)."""
    # Example: "11_46_21_.395"
    h_str, m_str, s_str, frac_str = stamp.split("_")
    # frac_str looks like ".395"
    frac = float(frac_str) if frac_str else 0.0
    return int(h_str) * 3600.0 + int(m_str) * 60.0 + int(s_str) + frac


def load_accel_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time_s_of_day, accel_xyz)`` from a 3-axis accel CSV.

    The file is GBK-encoded; the first column is wall-clock time. We return
    seconds-of-day so absolute alignment with other accel files is trivial.
    """
    times: list[float] = []
    axyz: list[tuple[float, float, float]] = []
    with path.open("r", encoding="gbk", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if "时间" not in header[0]:
            raise ValueError(f"Unexpected accel header in {path}: {header!r}")
        for row in reader:
            if len(row) < 4 or row[0] == "":
                continue
            times.append(_parse_clock_seconds(row[0]))
            axyz.append((float(row[1]), float(row[2]), float(row[3])))
    return np.asarray(times, dtype=float), np.asarray(axyz, dtype=float)


# ---------------------------------------------------------------------------
# Resampling and alignment
# ---------------------------------------------------------------------------

@dataclass
class ResampledRecord:
    time_s: np.ndarray            # uniform grid starting at 0
    force: np.ndarray             # aligned with time_s
    accel: dict[str, np.ndarray]  # station -> (N, 3) array, columns X/Y/Z


def _resample_irregular(
    src_t: np.ndarray, src_y: np.ndarray, dst_t: np.ndarray
) -> np.ndarray:
    """Linear interpolation onto ``dst_t``. Out-of-range samples become NaN.

    ``src_t`` need not be strictly monotonic (the logger sometimes emits
    duplicate timestamps); we sort and average duplicates first.
    """
    order = np.argsort(src_t, kind="stable")
    t = src_t[order]
    y = src_y[order]
    # Average values that share a timestamp so np.interp is well-defined.
    uniq_t, inv = np.unique(t, return_inverse=True)
    if uniq_t.size != t.size:
        if y.ndim == 1:
            sums = np.bincount(inv, weights=y)
            counts = np.bincount(inv)
            y_uniq = sums / counts
        else:
            y_uniq = np.empty((uniq_t.size, y.shape[1]), dtype=float)
            counts = np.bincount(inv)
            for k in range(y.shape[1]):
                sums = np.bincount(inv, weights=y[:, k])
                y_uniq[:, k] = sums / counts
        t, y = uniq_t, y_uniq

    if y.ndim == 1:
        out = np.interp(dst_t, t, y, left=np.nan, right=np.nan)
        out[(dst_t < t[0]) | (dst_t > t[-1])] = np.nan
        return out

    out = np.empty((dst_t.size, y.shape[1]), dtype=float)
    for k in range(y.shape[1]):
        out[:, k] = np.interp(dst_t, t, y[:, k], left=np.nan, right=np.nan)
        out[(dst_t < t[0]) | (dst_t > t[-1]), k] = np.nan
    return out


def resample_record(
    force_t: np.ndarray,
    force_y: np.ndarray,
    accel_streams: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    fs: float = 1000.0,
    force_clock_start: float | None,
) -> ResampledRecord:
    """Merge force + accel onto a uniform grid at ``fs`` Hz.

    The force trace starts at relative t=0 (no clock). The accel traces are
    in seconds-of-day. ``force_clock_start`` is the wall-clock seconds-of-day
    that corresponds to force ``t=0``. If ``None``, we align force ``t=0``
    with the earliest accel sample seen across all stations -- the user can
    then visually correct the offset if needed.
    """
    if force_clock_start is None:
        all_starts = [stream[0][0] for stream in accel_streams.values()]
        force_clock_start = float(min(all_starts))

    accel_ends = [stream[0][-1] for stream in accel_streams.values()]
    t_end_clock = min(float(force_t[-1] + force_clock_start), float(max(accel_ends)))
    t_start_clock = max(force_clock_start, float(min(s[0][0] for s in accel_streams.values())))
    duration = t_end_clock - t_start_clock
    if duration <= 0:
        raise ValueError(
            "No overlap between force and accel records; check --force-start."
        )

    n = int(np.floor(duration * fs)) + 1
    grid = np.arange(n) / fs  # seconds, starts at 0 = t_start_clock

    force_rel = force_t - (t_start_clock - force_clock_start)
    force_grid = np.interp(grid, force_rel, force_y, left=np.nan, right=np.nan)
    force_grid[(grid < force_rel[0]) | (grid > force_rel[-1])] = np.nan

    accel_grid: dict[str, np.ndarray] = {}
    for name, (t_clock, y_xyz) in accel_streams.items():
        accel_grid[name] = _resample_irregular(t_clock - t_start_clock, y_xyz, grid)

    return ResampledRecord(time_s=grid, force=force_grid, accel=accel_grid)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def apply_lowpass(
    record: "ResampledRecord",
    fs: float,
    *,
    cutoff_hz: float = 50.0,
    order: int = 4,
) -> "ResampledRecord":
    """Zero-phase Butterworth lowpass on the ACCELERATION channels only.

    Inspired by the upstream MATLAB pipeline in
    Excitation-Machinery-Parameter-Prediction-Method/analyse_chibi_data.m.
    The wireless accelerometers are dominated by HF sensor noise + bursty-
    packetisation artefacts; filtering them suppresses that band-wide noise
    and lets the per-impact FFT recover the true low-frequency structural
    response.

    The hammer-force trace is **deliberately left unfiltered**: a sharp
    impulse has flat spectral content well above any practical cutoff and
    LPF would smear it into a broad bump, destroying both the peak-detector
    threshold and the FRF input spectrum.
    """
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    if cutoff_hz >= nyq or cutoff_hz <= 0.0:
        return record
    b, a = butter(order, cutoff_hz / nyq, btype="low")

    def _filt(signal: np.ndarray, axis: int = -1) -> np.ndarray:
        finite_mask = np.isfinite(signal)
        if not finite_mask.any():
            return signal
        clean = np.where(finite_mask, signal, 0.0)
        if signal.ndim == 1:
            clean = clean - np.nanmean(signal)
        else:
            clean = clean - np.nanmean(signal, axis=0, keepdims=True)
        out = filtfilt(b, a, clean, axis=axis)
        out[~finite_mask] = np.nan
        return out

    new_accel = {s: _filt(arr, axis=0) for s, arr in record.accel.items()}
    return ResampledRecord(record.time_s, record.force, new_accel)


def auto_align_force(
    record: ResampledRecord,
    fs: float,
    *,
    reference_station: str = "root",
    smooth_s: float = 0.05,
    max_search_s: float = 60.0,
) -> tuple[ResampledRecord, float]:
    """Estimate force-vs-accel time lag via envelope cross-correlation.

    Builds smoothed envelopes (|force| and |accel|-magnitude at the
    reference station), cross-correlates them, and shifts the force trace
    so the two impact trains line up. Returns ``(record, lag_s)`` where
    ``lag_s > 0`` means the force was moved to *later* times by that amount.
    """
    if reference_station not in record.accel:
        return record, 0.0
    accel_ref = record.accel[reference_station]
    a_env = np.linalg.norm(np.where(np.isfinite(accel_ref), accel_ref, 0.0), axis=1)
    f_env = np.abs(np.where(np.isfinite(record.force), record.force, 0.0))

    box = max(1, int(smooth_s * fs))
    kernel = np.ones(box) / box
    a_env = np.convolve(a_env, kernel, mode="same")
    f_env = np.convolve(f_env, kernel, mode="same")
    a_env = a_env - a_env.mean()
    f_env = f_env - f_env.mean()

    n = f_env.size
    corr = np.correlate(a_env, f_env, mode="full")
    lags = np.arange(-(n - 1), n)
    max_lag = int(max_search_s * fs)
    mask = np.abs(lags) <= max_lag
    best_idx = int(np.argmax(corr[mask]))
    lag_samples = int(lags[mask][best_idx])

    new_force = np.full_like(record.force, np.nan)
    if lag_samples >= 0:
        new_force[lag_samples:] = record.force[: n - lag_samples]
    elif lag_samples < 0:
        new_force[: n + lag_samples] = record.force[-lag_samples:]
    return (
        ResampledRecord(record.time_s, new_force, record.accel),
        lag_samples / fs,
    )


def _detect_impacts_raw(
    time_s: np.ndarray,
    force: np.ndarray,
    *,
    threshold_ratio: float,
    min_separation_s: float,
) -> np.ndarray:
    if force.size == 0:
        return np.empty(0, dtype=int)
    mag = np.abs(force - np.nanmedian(force))
    peak_mag = np.nanmax(mag)
    if not np.isfinite(peak_mag) or peak_mag == 0.0:
        return np.empty(0, dtype=int)
    above = mag > threshold_ratio * peak_mag
    dt = time_s[1] - time_s[0]
    min_gap = max(1, int(min_separation_s / dt))
    picks: list[int] = []
    i = 0
    while i < mag.size:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < mag.size and above[j + 1]:
            j += 1
        local = i + int(np.nanargmax(mag[i : j + 1]))
        if not picks or local - picks[-1] >= min_gap:
            picks.append(local)
        i = j + 1
    return np.asarray(picks, dtype=int)


def detect_impacts(
    time_s: np.ndarray,
    force: np.ndarray,
    *,
    threshold_ratio: float = 0.2,
    min_separation_s: float = 0.05,
    reject_outliers_sigma: float = 5.0,
    outlier_max_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Hammer impact peaks with MAD-based drift-spike rejection.

    A peak is the most-extreme sample (positive or negative magnitude) above
    ``threshold_ratio * max(|force|)`` separated from any prior peak by at
    least ``min_separation_s`` seconds.

    Drift / glitch peaks (e.g., the wireless logger occasionally registering
    a single-sample saturation spike a few × larger than any real hammer
    strike) would otherwise inflate the detection threshold and mask real
    strikes. With ``reject_outliers_sigma > 0`` we iteratively:

      1. detect peaks on the working trace,
      2. compute the median and MAD of the detected peak magnitudes,
      3. flag peaks whose magnitude exceeds
         ``median + reject_outliers_sigma * 1.4826·MAD`` as drift artifacts,
      4. zero-mask a ±20 ms window around each flagged sample,
      5. re-detect, until no further outliers appear (or
         ``outlier_max_passes`` is reached).

    Returns ``(peak_indices, rejected_indices)`` where ``rejected_indices``
    is empty when no drift was found. Pass
    ``reject_outliers_sigma=0`` to disable rejection.
    """
    if force.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    work = np.where(np.isfinite(force), force, 0.0).astype(float, copy=True)
    rejected: list[int] = []
    dt = time_s[1] - time_s[0]
    half_window = max(1, int(0.02 / dt))  # ±20 ms zero-mask around a glitch

    for _ in range(max(1, outlier_max_passes)):
        peaks = _detect_impacts_raw(
            time_s, work,
            threshold_ratio=threshold_ratio,
            min_separation_s=min_separation_s,
        )
        if reject_outliers_sigma <= 0.0 or peaks.size < 4:
            return peaks, np.asarray(rejected, dtype=int)
        amps = np.abs(work[peaks] - np.nanmedian(work))
        med = float(np.median(amps))
        mad = float(np.median(np.abs(amps - med))) + 1e-30
        threshold = med + reject_outliers_sigma * 1.4826 * mad
        outlier_mask = amps > threshold
        if not outlier_mask.any():
            return peaks, np.asarray(rejected, dtype=int)
        for p in peaks[outlier_mask]:
            lo = max(0, int(p) - half_window)
            hi = min(work.size, int(p) + half_window + 1)
            work[lo:hi] = 0.0
            rejected.append(int(p))

    # Final pass after the last masking iteration.
    peaks = _detect_impacts_raw(
        time_s, work,
        threshold_ratio=threshold_ratio,
        min_separation_s=min_separation_s,
    )
    return peaks, np.asarray(rejected, dtype=int)


def compute_frf_ensemble(
    force: np.ndarray,
    accel: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    *,
    window_s: float = 1.0,
    pre_trigger_s: float = 0.05,
    force_pulse_s: float = 0.01,
    exp_window_end: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """H1 FRF + coherence from ensemble-averaged impact windows.

    For each impact peak, extract a ``window_s``-long segment starting
    ``pre_trigger_s`` before the peak. Apply:

      - Force window: rectangular over the pulse only (``pre_trigger_s +
        force_pulse_s``), zero elsewhere. Suppresses noise between impacts
        from leaking into the input spectrum.
      - Exponential window on the accel response, decaying to
        ``exp_window_end`` at the window end. Damps truncation leakage.

    Then accumulate Sxx, Sxy, Syy across windows, take the ensemble mean,
    and form H1 = Sxy/Sxx and coherence γ² = |Sxy|²/(Sxx·Syy).

    Windows where the next impact falls inside the window, or where any
    sample is non-finite, are skipped.

    Returns ``(freqs_hz, H_complex, gamma_squared, n_used)``.
    """
    nwin = int(round(window_s * fs))
    npre = int(round(pre_trigger_s * fs))
    npulse = int(round(force_pulse_s * fs))

    force_win = np.zeros(nwin)
    force_win[: npre + npulse] = 1.0
    tau = window_s / max(-np.log(exp_window_end), 1e-3)
    accel_win = np.exp(-np.arange(nwin) / fs / tau)

    nfreq = nwin // 2 + 1
    sxx = np.zeros(nfreq)
    sxy = np.zeros(nfreq, dtype=complex)
    syy = np.zeros(nfreq)
    used = 0

    for idx, p in enumerate(peaks):
        i0 = int(p) - npre
        i1 = i0 + nwin
        if i0 < 0 or i1 > force.size:
            continue
        if idx + 1 < peaks.size and int(peaks[idx + 1]) < i1:
            continue  # next impact intrudes; skip to keep input clean
        x = force[i0:i1]
        y = accel[i0:i1]
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            continue
        x = (x - x.mean()) * force_win
        y = (y - y.mean()) * accel_win
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)
        sxx += (X.conj() * X).real
        sxy += X.conj() * Y
        syy += (Y.conj() * Y).real
        used += 1

    if used == 0:
        raise ValueError("No valid impact windows for ensemble averaging.")
    sxx /= used
    sxy /= used
    syy /= used
    eps = float(np.max(sxx)) * 1e-12 + 1e-30
    H = sxy / (sxx + eps)
    gamma2 = (sxy.conj() * sxy).real / ((sxx + eps) * (syy + eps))
    freqs = np.fft.rfftfreq(nwin, d=1.0 / fs)
    return freqs, H, gamma2.astype(float), used


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cache for the heavy preprocessing pipeline (load + resample + auto-align).
# Stored under cache/hammer_test/<test>/processed.pkl with a small signature
# of input file mtimes + preprocessing args. Invalidated automatically when
# any of those change; --force on the CLI bypasses the cache.
# ---------------------------------------------------------------------------

def _file_signature(path: Path) -> tuple[str, float, int]:
    st = path.stat()
    return (path.name, st.st_mtime, st.st_size)


def _preprocessing_signature(
    force_path: Path,
    accel_paths: dict[str, Path],
    *,
    fs: float,
    force_clock_start: float | None,
    auto_align: bool,
    align_ref: str,
    lowpass_hz: float | None,
    lowpass_order: int,
    reject_outliers_sigma: float,
) -> dict[str, Any]:
    return {
        "version": 3,
        "force": _file_signature(force_path),
        "accel": {s: _file_signature(p) for s, p in accel_paths.items()},
        "fs": fs,
        "force_clock_start": force_clock_start,
        "auto_align": auto_align,
        "align_ref": align_ref,
        "lowpass_hz": lowpass_hz,
        "lowpass_order": lowpass_order,
        "reject_outliers_sigma": reject_outliers_sigma,
    }


@dataclass
class _ProcessedBlob:
    record: "ResampledRecord"
    peaks: np.ndarray
    lag_s: float
    signature: dict[str, Any]
    rejected_peaks: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rejected_peaks is None:
            self.rejected_peaks = np.empty(0, dtype=int)


def _try_load_cache(cache_path: Path, signature: dict[str, Any]) -> _ProcessedBlob | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as fh:
            blob: _ProcessedBlob = pickle.load(fh)
    except Exception as exc:
        print(f"[cache] ignoring unreadable cache ({exc}).")
        return None
    if getattr(blob, "signature", None) != signature:
        return None
    return blob


def _write_cache(cache_path: Path, blob: _ProcessedBlob) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)


# Styling reused across hammer-test figures so they match results/frf/frf_tree_*.png.
PRIMARY_COLOR = "#2166AC"
ACCENT_COLOR = "#B2182B"
GRID_MAJOR = "#d0d0d0"
GRID_MINOR = "#ececec"

# Unit strings: dot-multiplication form with negative exponents (paper style).
UNIT_ACCEL = r"m$\cdot$s$^{-2}$"
UNIT_FRF = r"m$\cdot$s$^{-2}\cdot$N$^{-1}$"


def _configure_matplotlib_style() -> None:
    """Use Times New Roman (or metrically-compatible fallback) for all text."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",       # native on Windows / mac
            "Liberation Serif",      # open-source metric-equivalent (Linux)
            "Nimbus Roman",          # urw alternative
            "DejaVu Serif",          # matplotlib's bundled fallback
        ],
        "mathtext.fontset": "stix",  # Times-like glyphs for math symbols
        "pdf.fonttype": 42,          # embed TrueType outlines (paper-safe)
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def _apply_house_grid(ax) -> None:
    ax.grid(True, which="major", linewidth=0.6, color=GRID_MAJOR)
    ax.grid(True, which="minor", linewidth=0.4, color=GRID_MINOR)


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    """Save a figure as both PNG (150 dpi) and PDF."""
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")


def plot_time_history_overview(
    record: ResampledRecord,
    peaks: np.ndarray,
    out_dir: Path,
    *,
    station: str = "tip",
    rejected: np.ndarray | None = None,
    excluded: np.ndarray | None = None,
    excluded_span: tuple[float, float] | None = None,
) -> None:
    """Two-panel overview: hammer force on top, station accel-Z on bottom."""
    _configure_matplotlib_style()
    import matplotlib.pyplot as plt

    fig, (ax_f, ax_a) = plt.subplots(
        2, 1, figsize=(6.6, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    t_lo = record.time_s[0]
    t_hi = record.time_s[-1]
    if excluded_span is not None:
        kept_lo, kept_hi = excluded_span
        for ax in (ax_f, ax_a):
            if kept_lo > t_lo:
                ax.axvspan(t_lo, kept_lo, color="#e9e9e9", alpha=0.55,
                           zorder=0, linewidth=0)
            if kept_hi < t_hi:
                ax.axvspan(kept_hi, t_hi, color="#e9e9e9", alpha=0.55,
                           zorder=0, linewidth=0,
                           label="off-axis (excluded)")

    ax_f.plot(record.time_s, record.force, color=PRIMARY_COLOR, linewidth=0.9)
    if peaks.size:
        ax_f.plot(
            record.time_s[peaks], record.force[peaks],
            linestyle="none", marker="v", markersize=4.0,
            markerfacecolor=ACCENT_COLOR, markeredgecolor="white",
            markeredgewidth=0.5, zorder=3,
            label=f"Z-axis impacts (N={peaks.size})",
        )
    if excluded is not None and excluded.size:
        ax_f.plot(
            record.time_s[excluded], record.force[excluded],
            linestyle="none", marker="v", markersize=4.0,
            markerfacecolor="#bdbdbd", markeredgecolor="white",
            markeredgewidth=0.5, zorder=2,
            label=f"off-axis impacts (N={excluded.size})",
        )
    if rejected is not None and rejected.size:
        ax_f.plot(
            record.time_s[rejected], record.force[rejected],
            linestyle="none", marker="x", markersize=7.0,
            markerfacecolor="none", markeredgecolor="#555555",
            markeredgewidth=1.4, zorder=4,
            label=f"rejected drift (N={rejected.size})",
        )
    ax_f.set_ylabel("Force [N]")
    ax_f.set_title("Hammer impact-test record (50 strikes, root→tip)")
    _apply_house_grid(ax_f)
    if peaks.size:
        ax_f.legend(loc="upper right", fontsize=9)

    if station in record.accel:
        z = record.accel[station][:, 2]
        ax_a.plot(record.time_s, z, color=PRIMARY_COLOR, linewidth=0.7)
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel(f"Accel-Z @ {station} [{UNIT_ACCEL}]")
    _apply_house_grid(ax_a)

    fig.tight_layout(pad=0.4)
    _save_fig(fig, out_dir, "time_history_overview")
    plt.close(fig)


def plot_station_frf(
    record: ResampledRecord,
    peaks: np.ndarray,
    out_dir: Path,
    fs: float,
    *,
    station: str,
    fmax: float = 30.0,
    window_s: float = 1.0,
) -> None:
    """FRF + coherence for a single station, paper-ready style."""
    _configure_matplotlib_style()
    import matplotlib.pyplot as plt

    if station not in record.accel:
        return
    axyz = record.accel[station]
    csv_cols: dict[str, np.ndarray] = {}
    freqs_full: np.ndarray | None = None
    H_by_comp: dict[str, np.ndarray] = {}
    coh_by_comp: dict[str, np.ndarray] = {}
    n_used = 0
    for k, comp in enumerate(COMPONENTS):
        freqs, H, gamma2, n_used = compute_frf_ensemble(
            record.force, axyz[:, k], fs, peaks, window_s=window_s
        )
        if freqs_full is None:
            freqs_full = freqs
        H_by_comp[comp] = np.abs(H)
        coh_by_comp[comp] = gamma2
        csv_cols[f"H_{comp}_mag_ms2_per_N"] = np.abs(H)
        csv_cols[f"coherence_{comp}"] = gamma2

    assert freqs_full is not None
    _write_frf_csv(out_dir / f"frf_{station}.csv", freqs_full, csv_cols)

    fig, (ax_h, ax_c) = plt.subplots(
        2, 1, figsize=(6.6, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    mask = (freqs_full > 0.5) & (freqs_full <= fmax)
    ax_h.semilogy(
        freqs_full[mask], H_by_comp["Z"][mask],
        color=PRIMARY_COLOR, linewidth=1.4,
        marker="o", markersize=3.2,
        markerfacecolor=PRIMARY_COLOR, markeredgecolor="white",
        markeredgewidth=0.5, zorder=2,
        label=fr"measured $|H|$, {station}-Z ($H_1$ avg of {n_used} impacts)",
    )
    ax_h.set_ylabel(rf"$|H|$ [{UNIT_FRF}]")
    ax_h.set_title(f"Measured FRF — {station} station, vertical (Z)")
    ax_h.set_xlim(0.0, fmax)
    _apply_house_grid(ax_h)
    ax_h.legend(loc="upper right", fontsize=10)

    ax_c.plot(freqs_full[mask], coh_by_comp["Z"][mask],
              color=ACCENT_COLOR, linewidth=1.0)
    ax_c.set_ylim(0.0, 1.05)
    ax_c.set_xlabel("Frequency [Hz]")
    ax_c.set_ylabel(r"Coherence $\gamma^2$")
    _apply_house_grid(ax_c)

    fig.tight_layout(pad=0.4)
    _save_fig(fig, out_dir, f"frf_{station}")
    plt.close(fig)


def compute_and_save_frf_csv_only(
    record: ResampledRecord,
    peaks: np.ndarray,
    out_dir: Path,
    fs: float,
    *,
    window_s: float = 1.0,
) -> None:
    """Compute FRF + coherence for every station; write CSV but no figure."""
    for station, axyz in record.accel.items():
        csv_cols: dict[str, np.ndarray] = {}
        freqs_full: np.ndarray | None = None
        for k, comp in enumerate(COMPONENTS):
            freqs, H, gamma2, _ = compute_frf_ensemble(
                record.force, axyz[:, k], fs, peaks, window_s=window_s
            )
            if freqs_full is None:
                freqs_full = freqs
            csv_cols[f"H_{comp}_mag_ms2_per_N"] = np.abs(H)
            csv_cols[f"coherence_{comp}"] = gamma2
        assert freqs_full is not None
        _write_frf_csv(out_dir / f"frf_{station}.csv", freqs_full, csv_cols)


def _write_frf_csv(
    path: Path, freqs: np.ndarray, columns: dict[str, np.ndarray]
) -> None:
    cols = ["frequency_hz", *columns.keys()]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for i, f in enumerate(freqs):
            row = [f"{f:.6g}"]
            for name in columns:
                row.append(_fmt(float(columns[name][i])))
            writer.writerow(row)


def write_resampled_csv(record: ResampledRecord, out_dir: Path, branch_id: str) -> None:
    cols = ["time_s", "excitation_force_N"]
    for station in STATIONS:
        if station not in record.accel:
            continue
        for comp in COMPONENTS:
            cols.append(f"obs_{branch_id}_{station}_{comp}_accel_ms2")

    with (out_dir / "resampled.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for i, t in enumerate(record.time_s):
            row: list[str] = [f"{t:.6f}", _fmt(record.force[i])]
            for station in STATIONS:
                if station not in record.accel:
                    continue
                row.extend(_fmt(record.accel[station][i, k]) for k in range(3))
            writer.writerow(row)


def _fmt(value: float) -> str:
    return "" if not np.isfinite(value) else f"{value:.6g}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_force_start(text: str | None) -> float | None:
    if text is None:
        return None
    # Accept "HH:MM:SS.mmm" or "HH_MM_SS_.mmm".
    raw = text.replace(":", "_").replace(".", "_.", 1) if ":" in text else text
    return _parse_clock_seconds(raw)


def _parse_time_range(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    sep = "," if "," in text else ":"
    lo_str, hi_str = text.split(sep, 1)
    return float(lo_str), float(hi_str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prefix",
        required=True,
        help="Common file prefix, e.g. 'trees/tree1_p1'. Files <prefix>_force.csv "
        "and <prefix>_{root,mid,tip}.csv must exist.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: results/hammer_test/<prefix-stem>).",
    )
    parser.add_argument(
        "--branch-id",
        default="branch1",
        help="Branch id used in output CSV column names "
        "(default: branch1).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="Resampling rate in Hz (default 1000, matches force logger).",
    )
    parser.add_argument(
        "--force-start",
        default=None,
        help="Wall-clock time (HH:MM:SS.mmm) of force t=0. If omitted, "
        "force is aligned with the earliest accel sample.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached preprocessing under cache/hammer_test/<name>/ and recompute.",
    )
    parser.add_argument(
        "--impacts-time-range",
        default=None,
        help="Restrict the FRF ensemble to impacts within [t_min, t_max] seconds "
        "of the common (post-align) time axis. Use this to exclude off-axis "
        "strikes (e.g., during tree1_p1 the X-direction strikes ran after "
        "t≈73 s, so pass '0,73' to keep only the Z-direction strikes). "
        "Format: 'min,max'. Excluded impacts still appear (greyed) in the "
        "time-history figure; the cache is *not* invalidated, so you can "
        "iterate this quickly.",
    )
    parser.add_argument(
        "--reject-outliers-sigma",
        type=float,
        default=5.0,
        help="MAD-based drift-spike rejection threshold on detected peak "
        "magnitudes (default 5.0; pass 0 to disable). "
        "Removes single-sample sensor saturation glitches before the FRF.",
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=50.0,
        help="Zero-phase Butterworth lowpass cutoff applied to force + accel "
        "before peak detection / FRF (default 50 Hz; ref: chibi pipeline uses 65 Hz). "
        "Pass 0 to disable.",
    )
    parser.add_argument(
        "--lowpass-order",
        type=int,
        default=4,
        help="Butterworth lowpass order (default 4).",
    )
    parser.add_argument(
        "--no-auto-align",
        action="store_true",
        help="Disable cross-correlation alignment of force vs accel envelopes. "
        "Use this only if --force-start is supplied and known to be exact.",
    )
    parser.add_argument(
        "--align-ref",
        choices=STATIONS,
        default="root",
        help="Reference accel station for envelope cross-correlation alignment.",
    )
    parser.add_argument(
        "--station",
        choices=STATIONS,
        default="tip",
        help="Which station the FRF / response figures focus on (default tip, "
        "best coherence in tree1_p1 dataset).",
    )
    parser.add_argument(
        "--frf-fmax",
        type=float,
        default=30.0,
        help="Upper frequency limit (Hz) for FRF plots (default 30).",
    )
    parser.add_argument(
        "--frf-window-s",
        type=float,
        default=1.0,
        help="Per-impact window length (s) for ensemble FRF (default 1.0).",
    )
    args = parser.parse_args()

    prefix = Path(args.prefix)
    force_path = prefix.with_name(prefix.name + "_force.csv")
    if not force_path.exists():
        print(f"Force CSV not found: {force_path}")
        return 2

    accel_paths = {s: prefix.with_name(prefix.name + f"_{s}.csv") for s in STATIONS}
    missing = [str(p) for p in accel_paths.values() if not p.exists()]
    if missing:
        print("Missing accel CSV(s):\n  " + "\n  ".join(missing))
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path("results/hammer_test") / prefix.name
    out_dir.mkdir(parents=True, exist_ok=True)

    force_clock_start = _parse_force_start(args.force_start)
    lowpass_hz: float | None = args.lowpass_hz if args.lowpass_hz > 0 else None
    signature = _preprocessing_signature(
        force_path, accel_paths,
        fs=args.fs, force_clock_start=force_clock_start,
        auto_align=not args.no_auto_align, align_ref=args.align_ref,
        lowpass_hz=lowpass_hz, lowpass_order=args.lowpass_order,
        reject_outliers_sigma=args.reject_outliers_sigma,
    )
    cache_path = CACHE_ROOT / prefix.name / "processed.pkl"

    blob: _ProcessedBlob | None = None
    if not args.force:
        blob = _try_load_cache(cache_path, signature)
        if blob is not None:
            print(f"[cache] hit {cache_path.relative_to(REPO_ROOT)} "
                  f"(lag={blob.lag_s:+.3f} s, {blob.peaks.size} impacts)")

    if blob is None:
        print(f"Loading {force_path}")
        force_t, force_y = load_force_csv(force_path)
        accel_streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for station, p in accel_paths.items():
            print(f"Loading {p}")
            accel_streams[station] = load_accel_csv(p)

        print(f"Resampling to {args.fs} Hz")
        record = resample_record(
            force_t, force_y, accel_streams, fs=args.fs,
            force_clock_start=force_clock_start,
        )
        lag_s = 0.0
        if not args.no_auto_align:
            record, lag_s = auto_align_force(
                record, fs=args.fs, reference_station=args.align_ref
            )
            print(f"Auto-aligned force by {lag_s:+.3f} s "
                  f"(cross-corr vs {args.align_ref} envelope)")
        if lowpass_hz is not None:
            print(f"Lowpass: Butterworth order {args.lowpass_order} @ "
                  f"{lowpass_hz:.1f} Hz (zero-phase filtfilt)")
            record = apply_lowpass(
                record, fs=args.fs,
                cutoff_hz=lowpass_hz, order=args.lowpass_order,
            )
        peaks, rejected = detect_impacts(
            record.time_s, record.force,
            reject_outliers_sigma=args.reject_outliers_sigma,
        )
        if rejected.size:
            print(
                f"Rejected {rejected.size} drift outlier(s) at t = "
                + ", ".join(f"{record.time_s[i]:.3f} s" for i in rejected)
            )
        blob = _ProcessedBlob(
            record=record, peaks=peaks, lag_s=lag_s,
            signature=signature, rejected_peaks=rejected,
        )
        _write_cache(cache_path, blob)
        print(f"[cache] wrote {cache_path.relative_to(REPO_ROOT)}")

    record = blob.record
    all_peaks = blob.peaks
    rejected = getattr(blob, "rejected_peaks", np.empty(0, dtype=int))

    # Optionally restrict to a contiguous time window so off-axis strikes
    # (e.g. X-direction hammering segments) don't contaminate the Z FRF.
    time_range = _parse_time_range(args.impacts_time_range)
    if time_range is not None:
        t_lo, t_hi = time_range
        peak_times = record.time_s[all_peaks]
        in_range = (peak_times >= t_lo) & (peak_times <= t_hi)
        peaks = all_peaks[in_range]
        excluded_peaks = all_peaks[~in_range]
        print(f"Time filter [{t_lo:.2f}, {t_hi:.2f}] s: "
              f"keeping {peaks.size} / {all_peaks.size} impacts "
              f"({excluded_peaks.size} excluded as off-axis)")
    else:
        peaks = all_peaks
        excluded_peaks = np.empty(0, dtype=int)
    if peaks.size:
        print("Detected impacts at t [s]:", ", ".join(f"{record.time_s[p]:.3f}" for p in peaks))
    else:
        print("Warning: no impacts detected above threshold.")

    print("Plotting time-history overview")
    plot_time_history_overview(
        record, peaks, out_dir,
        station=args.station, rejected=rejected,
        excluded=excluded_peaks,
        excluded_span=time_range,
    )
    print(f"Plotting FRF for {args.station} station")
    plot_station_frf(
        record,
        peaks,
        out_dir,
        fs=args.fs,
        station=args.station,
        fmax=args.frf_fmax,
        window_s=args.frf_window_s,
    )
    # Also dump CSVs for the non-plotted stations so downstream analysis has
    # everything without needing to rerun with --all-stations.
    other_stations = [s for s in STATIONS if s != args.station and s in record.accel]
    if other_stations:
        print(f"Writing FRF CSVs (no figure) for: {', '.join(other_stations)}")
    for station in other_stations:
        axyz = record.accel[station]
        csv_cols: dict[str, np.ndarray] = {}
        freqs_full: np.ndarray | None = None
        for k, comp in enumerate(COMPONENTS):
            freqs, H, gamma2, _ = compute_frf_ensemble(
                record.force, axyz[:, k], fs=args.fs, peaks=peaks,
                window_s=args.frf_window_s,
            )
            if freqs_full is None:
                freqs_full = freqs
            csv_cols[f"H_{comp}_mag_ms2_per_N"] = np.abs(H)
            csv_cols[f"coherence_{comp}"] = gamma2
        assert freqs_full is not None
        _write_frf_csv(out_dir / f"frf_{station}.csv", freqs_full, csv_cols)
    print("Writing resampled CSV")
    write_resampled_csv(record, out_dir, branch_id=args.branch_id)

    print(f"Done. Outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
