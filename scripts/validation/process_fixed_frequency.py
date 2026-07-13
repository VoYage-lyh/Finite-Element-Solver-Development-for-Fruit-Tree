"""Process variable-frequency excitation test data for Section 4.6 validation.

Reads a GBK-encoded chibi-format CSV (a single 3-axis accelerometer record
during a hold-staircase of discrete excitation frequencies), identifies each
hold segment, extracts the dominant drive frequency and the steady-state
RMS, and emits the per-segment metrics used as the posterior-predictive
validation observations.

Pipeline
--------
  1. Load (encoding=gbk), parse ``HH_MM_SS_.mmm`` timestamps,
     resample to a uniform 1 kHz grid (averaging duplicate timestamps).
  2. Mask hard-saturation samples (|a| ≥ ``--sat-thresh``, default 195 m·s⁻²)
     — the wireless accel clips at ±200 m·s⁻² and clipped samples distort
     both FFT and RMS. We replace saturated values with NaN.
  3. Detect active hold segments by thresholding a moving RMS of the
     unsaturated signal. Short blips and inter-segment gaps are merged.
  4. Per segment: Welch PSD on the unsaturated portion → dominant peak in
     [3, 25] Hz → drive frequency. RMS computed on the steady-state portion
     only (skip ``--transient-s`` seconds of leading transient).
  5. Group segments by closest nominal frequency (default {6, 10, 14, 18,
     22} Hz) and report mean ± std RMS per nominal frequency.

Outputs
-------
  - ``workspace/outputs/calibration/fixed_freq_segments.csv``  one row per segment
  - ``workspace/outputs/calibration/fixed_freq_summary.csv``   one row per nominal freq
  - ``workspace/outputs/calibration/fixed_freq_overview.{png,pdf}`` time-history +
     spectrogram + segment markers

This script is intentionally agnostic about which sensor the recording came
from (target branch / non-target branch). The mapping from this CSV to a
specific accelerometer location is supplied separately via ``--label``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from orchard_fem.workspace import display_path, workspace_paths

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = workspace_paths()


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────
def _parse_clock(stamp: str) -> float:
    h, m, s, f = stamp.split("_")
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + float(f)


def load_gbk_triaxial(path: Path):
    times: list[float] = []
    a: list[tuple[float, float, float]] = []
    with path.open("r", encoding="gbk", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if len(row) < 4 or not row[0]:
                continue
            times.append(_parse_clock(row[0]))
            a.append((float(row[1]), float(row[2]), float(row[3])))
    t = np.asarray(times)
    t = t - t[0]
    a = np.asarray(a)
    return t, a[:, 0], a[:, 1], a[:, 2]


def resample_uniform(t_raw, y_raw, fs):
    """Average duplicate timestamps then linear-interp onto a uniform grid."""
    uniq_t, inv = np.unique(t_raw, return_inverse=True)
    n_dup = np.bincount(inv)
    y_avg = np.bincount(inv, weights=y_raw) / n_dup
    t_grid = np.arange(0.0, uniq_t[-1], 1.0 / fs)
    return t_grid, np.interp(t_grid, uniq_t, y_avg)


# ─────────────────────────────────────────────────────────────────────────────
#  Segment detection
# ─────────────────────────────────────────────────────────────────────────────
def mask_saturated(a, sat_thresh):
    bad = np.abs(a) >= sat_thresh
    return bad


def moving_rms(y, win):
    """Causal moving RMS with window length ``win`` samples, NaN-safe."""
    y2 = np.where(np.isfinite(y), y * y, 0.0)
    mask = np.isfinite(y).astype(float)
    s = np.convolve(y2, np.ones(win), mode="same")
    n = np.convolve(mask, np.ones(win), mode="same")
    out = np.full_like(y, np.nan)
    ok = n > 0
    out[ok] = np.sqrt(s[ok] / n[ok])
    return out


def detect_segments(env, fs, *, threshold, min_seg_s=2.0, merge_gap_s=0.5):
    """Return ``[(start, end), ...]`` sample-index pairs where ``env >
    threshold`` for at least ``min_seg_s`` and gaps shorter than
    ``merge_gap_s`` are merged in."""
    active = env > threshold
    merge_gap = int(merge_gap_s * fs)
    # Close short gaps
    i = 0
    while i < active.size:
        if active[i]:
            j = i + 1
            while j < active.size and active[j]:
                j += 1
            # look ahead: if a gap then re-active inside merge_gap, fill it
            k = j
            while k < min(j + merge_gap, active.size) and not active[k]:
                k += 1
            if k < min(j + merge_gap, active.size):
                active[j:k] = True
                i = k
                continue
            i = j
        else:
            i += 1
    # Extract runs
    segments: list[tuple[int, int]] = []
    i = 0
    min_seg = int(min_seg_s * fs)
    while i < active.size:
        if active[i]:
            j = i + 1
            while j < active.size and active[j]:
                j += 1
            if (j - i) >= min_seg:
                segments.append((i, j))
            i = j
        else:
            i += 1
    return segments


# ─────────────────────────────────────────────────────────────────────────────
#  Per-segment metrics
# ─────────────────────────────────────────────────────────────────────────────
def segment_drive_freq_and_rms(t, ay, az, seg, fs, *, transient_s,
                                f_search=(3.0, 25.0)):
    """For one segment, return (drive_freq_Hz, rms_steady_ms2) using the
    sat-free portion. Drive frequency picked as the highest Welch PSD peak
    inside ``f_search``."""
    from scipy.signal import welch

    s, e = seg
    n_trans = int(transient_s * fs)
    s_steady = s + n_trans
    if s_steady >= e:
        return float("nan"), float("nan")

    # Pick the channel with more energy in this segment (target axis)
    sig_y = ay[s_steady:e]
    sig_z = az[s_steady:e]
    var_y = float(np.nanvar(sig_y))
    var_z = float(np.nanvar(sig_z))
    sig, axis = (sig_z, "z") if var_z >= var_y else (sig_y, "y")

    # Sat-free portion for spectral analysis
    finite = np.isfinite(sig)
    clean = sig[finite]
    if clean.size < int(0.5 * fs):
        return float("nan"), float("nan")
    # zero-mean
    clean = clean - clean.mean()
    nperseg = min(clean.size, int(2.0 * fs))
    f, Pxx = welch(clean, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band = (f >= f_search[0]) & (f <= f_search[1])
    if not band.any():
        return float("nan"), float("nan")
    peak_idx = int(np.argmax(Pxx[band]))
    drive_hz = float(f[band][peak_idx])

    rms = float(np.sqrt(np.mean(clean ** 2)))
    return drive_hz, rms, axis


def group_by_nominal(rows, nominal_freqs, tol_hz=1.0):
    """Cluster segments by their closest nominal frequency."""
    groups: dict[float, list[dict]] = {f: [] for f in nominal_freqs}
    for r in rows:
        d = [(abs(r["drive_hz"] - nf), nf) for nf in nominal_freqs]
        d.sort()
        nearest, nf = d[0]
        if nearest <= tol_hz:
            groups[nf].append(r)
    summary = []
    for nf, rs in groups.items():
        if not rs:
            summary.append({"nominal_hz": nf, "n": 0,
                            "rms_mean": float("nan"), "rms_std": float("nan"),
                            "drive_mean": float("nan")})
            continue
        rms = np.array([r["rms_steady"] for r in rs])
        drv = np.array([r["drive_hz"] for r in rs])
        summary.append({
            "nominal_hz": nf, "n": int(len(rs)),
            "rms_mean": float(rms.mean()), "rms_std": float(rms.std()),
            "drive_mean": float(drv.mean()),
        })
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────
def _configure_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def plot_overview(t, ay, az, segments, segment_rows, out_stem, fs):
    _configure_paper_style()
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    fig, axes = plt.subplots(
        3, 1, figsize=(10.0, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 2]},
    )

    for ax_panel, sig, name in zip(axes[:2], [ay, az], ["aY", "aZ"]):
        ax_panel.plot(t, sig, color="#2166AC", linewidth=0.35)
        for r, (s, e) in zip(segment_rows, segments):
            ax_panel.axvspan(t[s], t[e - 1], color="#ffe082", alpha=0.35,
                             zorder=0, linewidth=0)
        ax_panel.set_ylabel(rf"{name} [m$\cdot$s$^{{-2}}$]")
        ax_panel.set_ylim(-220, 220)
        ax_panel.grid(True, which="major", linewidth=0.6, color="#d0d0d0")

    # spectrogram with saturated samples masked
    az_clean = np.where(np.isfinite(az), az, 0.0)
    f_sp, t_sp, Sxx = spectrogram(az_clean, fs=fs, nperseg=int(fs * 1.0),
                                   noverlap=int(fs * 0.75))
    band = f_sp <= 30
    axes[2].pcolormesh(t_sp, f_sp[band], 10 * np.log10(Sxx[band] + 1e-12),
                       shading="auto", cmap="viridis")
    for r, (s, e) in zip(segment_rows, segments):
        axes[2].axvline(t[s], color="white", lw=0.5, alpha=0.5)
        axes[2].axvline(t[e - 1], color="white", lw=0.5, alpha=0.5)
        axes[2].text(0.5 * (t[s] + t[e - 1]), 28, f"{r['drive_hz']:.1f} Hz",
                      ha="center", va="top", color="white", fontsize=8,
                      bbox=dict(facecolor="#000", alpha=0.4, pad=1))
    axes[2].set_ylabel("Frequency [Hz]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylim(0, 30)
    axes[2].set_title("aZ spectrogram with detected hold segments")

    fig.tight_layout(pad=0.4)
    fig.savefig(out_stem.with_suffix(".png"), dpi=150)
    fig.savefig(out_stem.with_suffix(".pdf"))
    import matplotlib.pyplot as _plt
    _plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", default=str(WORKSPACE.tree_models / "变频激振数据.csv"),
        help="GBK-encoded triaxial accel CSV (default: 变频激振数据.csv).",
    )
    parser.add_argument("--fs", type=float, default=1000.0)
    parser.add_argument(
        "--sat-thresh", type=float, default=195.0,
        help="Saturation clip threshold in m·s⁻² (default 195).",
    )
    parser.add_argument(
        "--env-thresh", type=float, default=10.0,
        help="Moving-RMS threshold in m·s⁻² to flag an active hold segment.",
    )
    parser.add_argument(
        "--transient-s", type=float, default=0.5,
        help="Discard this many leading seconds of each segment as transient.",
    )
    parser.add_argument(
        "--nominal-freqs", default="6,10,14,18,22",
        help="Comma-separated nominal hold frequencies in Hz.",
    )
    parser.add_argument(
        "--out-dir", default=str(WORKSPACE.outputs / "calibration"),
        help="Output directory (default: workspace outputs/calibration).",
    )
    parser.add_argument(
        "--name", required=True,
        help="Sensor name. Used both as the row label in the segments CSV "
        "and as the filename suffix (e.g. --name sensor1 → "
        "fixed_freq_segments_sensor1.csv).",
    )
    args = parser.parse_args()

    nominal_freqs = [float(s) for s in args.nominal_freqs.split(",")]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} …")
    t_raw, ax, ay, az = load_gbk_triaxial(REPO / args.input)
    print(f"  {t_raw.size} samples, duration {t_raw[-1]:.1f} s")

    fs = args.fs
    t, ay_g = resample_uniform(t_raw, ay, fs)
    _, az_g = resample_uniform(t_raw, az, fs)
    # DC removal per channel
    ay_g -= np.nanmean(ay_g)
    az_g -= np.nanmean(az_g)

    # Mask saturated samples
    bad_y = mask_saturated(ay_g, args.sat_thresh)
    bad_z = mask_saturated(az_g, args.sat_thresh)
    ay_g[bad_y] = np.nan
    az_g[bad_z] = np.nan
    n_sat = int(bad_y.sum() + bad_z.sum())
    print(f"  saturated samples: {n_sat} ({n_sat / (2 * az_g.size) * 100:.1f}% of y+z)")

    # Active-segment detection via moving RMS on Z (typically the lateral
    # bending direction excited by the shaker)
    env_win = int(0.5 * fs)
    env = moving_rms(az_g, env_win)
    segments = detect_segments(env, fs, threshold=args.env_thresh,
                                min_seg_s=2.0, merge_gap_s=0.5)
    print(f"  detected {len(segments)} active hold segments")

    rows = []
    for k, (s, e) in enumerate(segments, 1):
        drive_hz, rms_steady, axis = segment_drive_freq_and_rms(
            t, ay_g, az_g, (s, e), fs, transient_s=args.transient_s,
        )
        rows.append({
            "segment": k,
            "t_start_s": float(t[s]),
            "t_end_s": float(t[e - 1]),
            "duration_s": float(t[e - 1] - t[s]),
            "drive_hz": drive_hz,
            "rms_steady": rms_steady,
            "axis": axis,
        })
        print(f"  seg {k:2d}: t=[{t[s]:6.1f}, {t[e-1]:6.1f}] s, "
              f"drive={drive_hz:5.2f} Hz, RMS={rms_steady:6.2f} m·s⁻² ({axis})")

    summary = group_by_nominal(rows, nominal_freqs, tol_hz=1.5)
    print("\nSummary (RMS averaged per nominal hold frequency):")
    print(f"{'nominal Hz':>11s}  {'n':>2s}  {'drive Hz':>9s}  "
          f"{'RMS mean':>10s}  {'RMS std':>10s}")
    for s in summary:
        if s["n"] == 0:
            print(f"  {s['nominal_hz']:>9.1f}   0     --        --        --")
        else:
            print(f"  {s['nominal_hz']:>9.1f}  {s['n']:2d}  "
                  f"{s['drive_mean']:>7.2f}    "
                  f"{s['rms_mean']:>8.2f}  {s['rms_std']:>8.2f}")

    # Write CSVs (filename suffix = --name so concurrent sensors stay separate)
    suffix = args.name
    seg_csv = out_dir / f"fixed_freq_segments_{suffix}.csv"
    with seg_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "segment", "t_start_s", "t_end_s",
                    "duration_s", "drive_hz", "rms_steady", "axis"])
        for r in rows:
            w.writerow([args.name, r["segment"], f"{r['t_start_s']:.3f}",
                        f"{r['t_end_s']:.3f}", f"{r['duration_s']:.3f}",
                        f"{r['drive_hz']:.3f}", f"{r['rms_steady']:.4f}",
                        r["axis"]])

    summary_csv = out_dir / f"fixed_freq_summary_{suffix}.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nominal_hz", "n", "drive_mean_hz",
                    "rms_mean_ms2", "rms_std_ms2"])
        for s in summary:
            w.writerow([s["nominal_hz"], s["n"],
                        f"{s['drive_mean']:.3f}",
                        f"{s['rms_mean']:.4f}",
                        f"{s['rms_std']:.4f}"])

    print(f"\nWrote: {display_path(seg_csv)}, {display_path(summary_csv)}")

    plot_overview(t, ay_g, az_g, segments, rows,
                  out_dir / f"fixed_freq_overview_{suffix}", fs)
    print(f"Wrote: {display_path(out_dir / ('fixed_freq_overview_' + suffix))}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
