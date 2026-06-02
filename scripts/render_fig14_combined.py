"""Combine the two single-sensor Fig 14 panels into a single side-by-side figure.

Renders ``fig14_combined.{png,pdf}`` from the per-sensor Table 5 CSVs so the
2-panel layout matches the paper format. Re-runs forward predictions for the
posterior-predictive band? No — for compactness it just re-uses the values
already in Table 5 (drive_hz, measured_rms, posterior_median, q05_ms2, q95_ms2,
rel_err_pct), so it's offline / cheap.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_table5(path: Path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    out = {
        "drive_hz": np.array([float(r["drive_hz"]) for r in rows]),
        "rms_meas": np.array([float(r["measured_rms"]) for r in rows]),
        "rms_pred": np.array([float(r["posterior_median"]) for r in rows]),
        "q05": np.array([float(r["q05_ms2"]) for r in rows]),
        "q95": np.array([float(r["q95_ms2"]) for r in rows]),
        "rel_err": np.array([float(r["rel_err_pct"]) for r in rows]),
    }
    out["covered"] = (out["rms_meas"] >= out["q05"]) & (out["rms_meas"] <= out["q95"])
    return out


def _configure_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def draw_panel(ax_top, ax_err, data, *, title, accent="#B2182B"):
    primary = "#2166AC"

    ax_top.fill_between(
        data["drive_hz"], data["q05"], data["q95"],
        color=accent, alpha=0.18, linewidth=0,
        label=r"90% posterior predictive band",
    )
    ax_top.plot(
        data["drive_hz"], data["rms_pred"],
        color=accent, linewidth=1.6, marker="s", markersize=4.4,
        markerfacecolor="white", markeredgecolor=accent, markeredgewidth=1.2,
        label="posterior median",
    )
    ax_top.errorbar(
        data["drive_hz"], data["rms_meas"],
        yerr=0.05 * data["rms_meas"],
        fmt="o", color="black", markersize=5.0, capsize=2.5,
        markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.6,
        label="measured RMS",
    )
    ax_top.set_ylabel(r"steady-state RMS [m$\cdot$s$^{-2}$]")
    ax_top.set_title(title, fontsize=11)
    ax_top.grid(True, which="major", linewidth=0.6, color="#d0d0d0")
    ax_top.legend(loc="upper right", fontsize=8.5)

    bar_colors = [primary if abs(e) <= 15 else accent for e in data["rel_err"]]
    ax_err.bar(data["drive_hz"], data["rel_err"], color=bar_colors,
               edgecolor="#333", width=0.35)
    ax_err.axhline(0, color="black", lw=0.6)
    ax_err.axhline(15, color="#999", lw=0.8, linestyle="--")
    ax_err.axhline(-15, color="#999", lw=0.8, linestyle="--")
    ax_err.set_ylabel("rel. error [%]")
    ax_err.set_xlabel(r"drive frequency $f$ [Hz]")
    ax_err.grid(True, which="major", linewidth=0.6, color="#d0d0d0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--left", default="results/calibration/table5_fixed_freq_sensor1.csv",
    )
    parser.add_argument(
        "--right", default="results/calibration/table5_fixed_freq_sensor2.csv",
    )
    parser.add_argument(
        "--left-title", default=r"(a) sensor 1 — left\_leader / tip / $u_x$",
    )
    parser.add_argument(
        "--right-title", default=r"(b) sensor 2 — right\_terminal\_spray / tip / $u_x$",
    )
    parser.add_argument(
        "--out-stem", default="results/calibration/fig14_combined",
    )
    args = parser.parse_args()

    d_left = load_table5(REPO / args.left)
    d_right = load_table5(REPO / args.right)

    cov_l = float(d_left["covered"].mean()) * 100.0
    cov_r = float(d_right["covered"].mean()) * 100.0
    err_l = float(np.mean(np.abs(d_left["rel_err"])))
    err_r = float(np.mean(np.abs(d_right["rel_err"])))

    title_l = (rf"{args.left_title};  "
               rf"coverage = {cov_l:.0f}%,  $|\bar{{e}}| = {err_l:.0f}\%$")
    title_r = (rf"{args.right_title};  "
               rf"coverage = {cov_r:.0f}%,  $|\bar{{e}}| = {err_r:.0f}\%$")

    _configure_paper_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 2, figsize=(11.4, 4.6), sharex="col",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    draw_panel(axes[0, 0], axes[1, 0], d_left, title=title_l)
    draw_panel(axes[0, 1], axes[1, 1], d_right, title=title_r)
    fig.tight_layout(pad=0.5)

    stem = REPO / args.out_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote: {stem.relative_to(REPO)}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
