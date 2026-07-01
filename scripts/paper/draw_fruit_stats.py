"""Render four independent figures summarising 20-fruit *Camellia oleifera*
field measurements so each can be placed individually in LaTeX.

Outputs (under ``results/summary/``):
    * ``fruit_stats_distribution.{png,pdf}`` — geometry / mass / force violins
    * ``fruit_stats_force_vs_mass.{png,pdf}`` — F_det vs mass scatter
    * ``fruit_stats_force_vs_position.{png,pdf}`` — F_det by growing position
    * ``fruit_stats_force_vs_height.{png,pdf}`` — F_det vs canopy height
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]


# ────────────────────────────────────────────────────────────────────────────
#  Field measurements (120 fruits)
# ────────────────────────────────────────────────────────────────────────────
LONG_AXIS = np.array([
    54.00, 38.00, 37.00, 41.00, 49.00, 48.00, 42.00, 45.00, 40.00, 46.00, 48.00, 38.00,
    47.00, 42.00, 45.00, 42.00, 43.00, 40.00, 41.00, 34.00, 44.27, 43.98, 35.89, 36.01,
    44.77, 43.36, 43.66, 43.80, 44.97, 43.17, 43.76, 44.23, 47.45, 40.53, 46.77, 46.05,
    43.50, 40.72, 44.77, 47.14, 37.85, 43.30, 43.31, 39.83, 40.03, 35.57, 37.72, 35.63,
    40.05, 40.32, 40.41, 44.78, 42.10, 39.51, 39.54, 37.58, 42.06, 41.97, 50.77, 50.35,
    50.08, 52.96, 37.17, 46.22, 46.65, 45.28, 53.19, 55.16, 47.69, 46.13, 37.62, 37.63,
    46.35, 42.64, 36.63, 42.41, 44.72, 47.48, 42.48, 41.71, 43.03, 43.83, 37.43, 46.99,
    53.49, 46.90, 36.06, 43.45, 34.54, 34.24, 46.16, 43.37, 39.71, 33.00, 36.92, 43.78,
    43.15, 46.30, 50.30, 41.37, 45.25, 42.34, 40.62, 41.83, 47.60, 38.07, 48.47, 50.08,
    43.17, 44.22, 36.24, 54.31, 47.69, 43.83, 43.28, 45.16, 39.88, 44.36, 38.58, 45.38,
], dtype=float)

SHORT_AXIS = np.array([
    43.00, 30.00, 25.00, 29.00, 36.00, 34.00, 31.00, 37.00, 35.00, 34.00, 38.00, 33.00,
    34.00, 35.00, 37.00, 31.00, 37.00, 32.00, 30.00, 32.00, 30.99, 37.16, 33.18, 34.70,
    45.00, 36.81, 34.70, 33.82, 29.57, 37.90, 39.94, 33.90, 40.23, 33.21, 26.69, 31.09,
    29.50, 32.57, 36.11, 30.00, 33.34, 34.43, 30.77, 30.08, 33.99, 37.07, 31.28, 30.70,
    33.10, 34.79, 31.52, 32.34, 32.74, 29.39, 32.95, 31.45, 39.86, 31.62, 35.44, 35.14,
    29.26, 28.75, 31.90, 36.25, 31.40, 31.08, 22.27, 32.09, 34.88, 35.39, 35.33, 32.36,
    32.76, 36.92, 33.12, 24.61, 37.65, 29.42, 39.05, 40.30, 36.64, 37.20, 40.94, 29.33,
    34.65, 34.59, 29.86, 41.54, 31.37, 30.78, 34.70, 37.70, 31.29, 31.30, 35.63, 28.06,
    30.62, 30.36, 31.95, 33.09, 40.89, 37.21, 33.74, 32.77, 37.08, 35.33, 37.15, 39.07,
    29.69, 24.81, 33.56, 33.96, 34.78, 31.30, 32.74, 35.84, 36.05, 38.87, 25.15, 27.18,
], dtype=float)

MASS_G = np.array([
    46.920, 16.300, 13.120, 17.060, 33.230, 26.750, 21.840, 27.970, 24.020, 32.660, 31.630,
    20.930, 26.700, 29.760, 35.070, 21.650, 30.310, 15.830, 15.290, 21.070, 39.426, 28.592,
    33.819, 24.563, 23.874, 22.587, 12.502, 15.951, 15.159, 29.703, 24.544, 34.343, 26.906,
    26.054, 38.719, 16.088, 19.843, 22.632, 25.407, 44.405, 21.993, 36.331, 21.438, 34.169,
    32.055, 32.079, 35.360, 20.116, 37.029, 34.827, 24.598, 25.519, 14.977, 25.152, 19.898,
    25.370, 46.477, 35.750, 25.287, 21.536, 24.746, 36.424, 29.623, 24.851, 17.462, 18.170,
    10.000, 45.288, 25.071, 30.563, 22.702, 15.040, 29.231, 22.113, 18.965, 23.801, 25.604,
    24.166, 25.400, 14.339, 24.570, 37.733, 11.968, 24.996, 18.429, 18.845, 26.985, 42.545,
    28.433, 21.075, 10.000, 27.613, 32.700, 22.174, 10.000, 12.739, 31.821, 28.612, 32.867,
    28.498, 37.540, 32.282, 18.899, 35.072, 23.696, 16.069, 31.858, 14.436, 26.620, 24.177,
    26.405, 21.165, 24.099, 21.771, 35.028, 14.540, 16.979, 24.416, 21.864, 15.126,
], dtype=float)

FORCE_N = np.array([
    33.70, 19.90,  6.50,  8.90, 27.10, 13.30, 34.90, 34.60, 15.60, 49.30,  6.90,  5.70,
    32.30, 20.90, 11.70, 24.20, 10.60, 10.70, 19.90, 18.20, 29.73, 15.31, 28.07, 24.69,
    27.23,  5.65,  1.06, 27.75, 14.35, 31.63,  1.00,  7.84, 15.71, 10.28, 21.56,  8.90,
    26.16, 24.71, 22.69, 32.02, 17.72, 23.32, 19.03, 36.98, 25.03, 28.92, 34.32, 20.15,
    21.84, 26.64, 35.50,  7.63, 12.78, 16.93, 31.77,  1.69, 14.85, 17.18,  1.09, 30.39,
     8.26, 26.77, 36.61, 15.04, 37.02, 17.90,  9.95, 37.77,  6.98, 18.92, 16.35, 22.92,
    28.52, 25.49, 23.49, 13.42, 18.07, 20.69, 40.04,  8.40, 17.25, 30.19, 16.31, 35.68,
     1.00,  2.30, 40.70, 28.22, 20.11, 30.96, 15.05, 32.02,  6.74,  4.45,  7.14, 13.06,
    38.56,  3.60, 41.60, 26.75, 33.37, 11.06, 15.75, 41.38, 39.02, 16.52, 26.76, 19.94,
    19.93,  8.05,  1.49, 35.49,  6.08, 11.25, 14.37,  2.84, 20.43,  4.08, 30.28, 18.94,
], dtype=float)

CRACK = np.array([
    1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1,
    1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1,
    0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
    1, 1, 0, 1,
], dtype=int)

HEIGHT_MM = np.array([
    1480.0, 1640.0, 1920.0, 1840.0, 1530.0, 1720.0, 1420.0, 1450.0, 1690.0, 1350.0, 1880.0,
    1960.0, 1510.0, 1610.0, 1760.0, 1560.0, 1810.0, 1780.0, 1630.0, 1670.0, 1459.2, 1740.3,
    1635.0, 1576.0, 1566.5, 1832.2, 1902.1, 1458.2, 1748.2, 1532.0, 1939.8, 1911.1, 1737.9,
    1809.2, 1664.2, 1853.8, 1623.5, 1556.7, 1519.5, 1488.1, 1670.7, 1612.9, 1754.5, 1357.8,
    1623.1, 1572.8, 1469.2, 1712.9, 1615.5, 1533.0, 1477.7, 1883.5, 1836.5, 1720.1, 1560.5,
    1927.6, 1693.7, 1614.0, 1899.3, 1544.8, 1774.6, 1541.4, 1509.5, 1777.5, 1489.0, 1650.8,
    1703.8, 1465.2, 1796.1, 1684.6, 1729.1, 1577.0, 1606.3, 1651.8, 1636.7, 1762.1, 1702.1,
    1603.9, 1351.1, 1834.4, 1666.8, 1549.2, 1655.0, 1407.0, 1901.9, 1953.7, 1385.8, 1568.5,
    1795.3, 1541.7, 1695.9, 1604.1, 1822.3, 1934.7, 1891.9, 1728.8, 1447.8, 1927.7, 1364.4,
    1531.6, 1521.8, 1759.8, 1647.3, 1406.7, 1357.8, 1718.6, 1649.8, 1716.8, 1694.1, 1893.2,
    1910.2, 1464.8, 1924.7, 1830.5, 1739.8, 1752.4, 1683.1, 1909.5, 1492.3, 1715.6,
], dtype=float)

POSITION = np.array([
    "mid", "mid", "tip", "tip", "mid", "tip", "root", "mid", "mid", "root", "tip", "tip",
    "root", "mid", "tip", "mid", "tip", "tip", "mid", "tip", "root", "tip", "mid", "mid",
    "mid", "tip", "tip", "root", "tip", "mid", "tip", "tip", "tip", "tip", "mid", "tip",
    "mid", "mid", "root", "root", "mid", "mid", "tip", "root", "mid", "mid", "root", "mid",
    "mid", "mid", "root", "tip", "tip", "mid", "mid", "tip", "mid", "mid", "tip", "mid",
    "tip", "mid", "root", "tip", "root", "mid", "mid", "root", "tip", "mid", "mid", "mid",
    "mid", "mid", "mid", "tip", "mid", "mid", "root", "tip", "mid", "mid", "mid", "root",
    "tip", "tip", "root", "mid", "tip", "mid", "mid", "mid", "tip", "tip", "tip", "mid",
    "root", "tip", "root", "mid", "root", "tip", "mid", "root", "root", "mid", "mid", "mid",
    "mid", "tip", "tip", "root", "tip", "tip", "tip", "tip", "mid", "tip", "root", "mid",
])

POSITION_LABEL = {"root": "Branch root", "mid": "Branch mid", "tip": "Branch tip"}
POSITION_COLOR = {"root": "#B2182B", "mid": "#3F60A0", "tip": "#3F7A55"}
POSITION_ORDER = ["root", "mid", "tip"]


# ────────────────────────────────────────────────────────────────────────────
#  Style
# ────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d8d8d8",
    "grid.linewidth": 0.6,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.fontsize": 12.5,
    "legend.edgecolor": "#cccccc",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def _save_both(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))


def _pearson(x, y):
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _linreg(x, y):
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(m), float(b), float(1.0 - ss_res / ss_tot)


def _format_p(p: float) -> str:
    """Render a p-value the way statistics papers usually do."""
    if p < 1.0e-4:
        return r"p < 10^{-4}"
    if p < 0.001:
        return r"p < 0.001"
    return f"p = {p:.3f}"


# ────────────────────────────────────────────────────────────────────────────
#  Figure 1 — distributions
# ────────────────────────────────────────────────────────────────────────────
def _save_distribution(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    variables = [
        ("Long axis [mm]",       LONG_AXIS,  "#3F60A0"),
        ("Short axis [mm]",      SHORT_AXIS, "#3F7A55"),
        ("Mass [g]",             MASS_G,     "#A38033"),
        ("Detachment force [N]", FORCE_N,    "#A04D6A"),
    ]
    parts = ax.violinplot([v for _, v, _ in variables],
                          positions=range(len(variables)),
                          widths=0.78, showmedians=True, showextrema=False)
    for body, (_, _, c) in zip(parts["bodies"], variables):
        body.set_facecolor(c)
        body.set_edgecolor("#222")
        body.set_alpha(0.55)
    parts["cmedians"].set_color("#222")
    parts["cmedians"].set_linewidth(1.4)

    rng = np.random.default_rng(2026)
    for i, (_, vals, c) in enumerate(variables):
        xs = i + (rng.random(vals.size) - 0.5) * 0.22
        ax.scatter(xs, vals, color=c, s=24, alpha=0.9,
                   edgecolors="white", linewidths=0.6, zorder=3)

    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels([n for n, _, _ in variables],
                       rotation=20, ha="center", va="top")
    ax.set_ylabel("Measured value")
    ax.set_title("Fruit-level field measurements")
    ax.set_ylim(0, max(LONG_AXIS.max(), FORCE_N.max()) * 1.18)

    for i, (_, vals, _) in enumerate(variables):
        mu, sd = vals.mean(), vals.std(ddof=1)
        ax.text(i, vals.max() + 2.5, f"{mu:.1f} ± {sd:.1f}",
                ha="center", va="bottom", fontsize=12,
                color="#222", family="serif")

    fig.tight_layout(pad=0.4)
    _save_both(fig, out_dir / "fruit_stats_distribution")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Force vs mass
# ────────────────────────────────────────────────────────────────────────────
def _save_force_vs_mass(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for pos in POSITION_ORDER:
        m = POSITION == pos
        ax.scatter(MASS_G[m], FORCE_N[m],
                   color=POSITION_COLOR[pos], s=72,
                   edgecolors="white", linewidths=0.8,
                   label=POSITION_LABEL[pos], zorder=4)

    slope, intercept, r2 = _linreg(MASS_G, FORCE_N)
    r_mass, p_mass = _pearson(MASS_G, FORCE_N)
    xs = np.linspace(MASS_G.min() * 0.95, MASS_G.max() * 1.05, 50)
    ax.plot(xs, slope * xs + intercept,
            color="#444", linewidth=1.5, linestyle="--", zorder=3)
    ax.text(0.04, 0.96,
            f"$F_{{det}} = {slope:.2f}\\,m {'+' if intercept >= 0 else '-'} "
            f"{abs(intercept):.2f}$\n"
            f"$r = {r_mass:.2f}$, ${_format_p(p_mass)}$, "
            f"$R^{{2}} = {r2:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, family="serif",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#bbb", lw=0.7, alpha=0.95))
    ax.set_xlabel("Fruit mass [g]")
    ax.set_ylabel("Detachment force [N]")
    ax.set_title("Detachment force vs. fruit mass")
    ax.legend(loc="lower right")

    fig.tight_layout(pad=0.4)
    _save_both(fig, out_dir / "fruit_stats_force_vs_mass")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Force by growing position
# ────────────────────────────────────────────────────────────────────────────
def _save_force_vs_position(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    pos_data = [FORCE_N[POSITION == pos] for pos in POSITION_ORDER]
    bplot = ax.boxplot(pos_data, widths=0.55, patch_artist=True,
                       medianprops=dict(color="#111", linewidth=1.6),
                       whiskerprops=dict(color="#444", linewidth=1.1),
                       capprops=dict(color="#444", linewidth=1.1),
                       flierprops=dict(marker="o", markersize=5,
                                        markerfacecolor="#888",
                                        markeredgecolor="white",
                                        linestyle="none"))
    for box, pos in zip(bplot["boxes"], POSITION_ORDER):
        box.set_facecolor(POSITION_COLOR[pos])
        box.set_alpha(0.55)
        box.set_edgecolor("#333")

    rng = np.random.default_rng(2026)
    for i, pos in enumerate(POSITION_ORDER):
        m = POSITION == pos
        xs = (i + 1) + (rng.random(m.sum()) - 0.5) * 0.24
        ax.scatter(xs, FORCE_N[m],
                   color=POSITION_COLOR[pos], s=38, zorder=4,
                   edgecolors="white", linewidths=0.7, alpha=0.95)

    ax.set_xticklabels([POSITION_LABEL[p] for p in POSITION_ORDER])
    ax.set_ylabel("Detachment force [N]")
    ax.set_title("Detachment force by growing position")

    for i, pos in enumerate(POSITION_ORDER):
        vals = FORCE_N[POSITION == pos]
        ax.text(i + 1, vals.max() + 2.0,
                f"n={vals.size}\nmean={vals.mean():.1f} N",
                ha="center", va="bottom", fontsize=12,
                family="serif", color="#222")
    ax.set_ylim(0, FORCE_N.max() * 1.22)

    fig.tight_layout(pad=0.4)
    _save_both(fig, out_dir / "fruit_stats_force_vs_position")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
#  Figure 4 — Force vs canopy height
# ────────────────────────────────────────────────────────────────────────────
def _save_force_vs_height(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for pos in POSITION_ORDER:
        m = POSITION == pos
        ax.scatter(HEIGHT_MM[m] / 1000.0, FORCE_N[m],
                   color=POSITION_COLOR[pos], s=72,
                   edgecolors="white", linewidths=0.8,
                   label=POSITION_LABEL[pos], zorder=4)

    r_h, p_h = _pearson(HEIGHT_MM, FORCE_N)
    slope_h, intercept_h, r2_h = _linreg(HEIGHT_MM / 1000.0, FORCE_N)
    xs = np.linspace(HEIGHT_MM.min() / 1000.0 - 0.05,
                     HEIGHT_MM.max() / 1000.0 + 0.05, 50)
    ax.plot(xs, slope_h * xs + intercept_h,
            color="#444", linewidth=1.5, linestyle="--", zorder=3)
    ax.set_xlabel("Canopy height [m]")
    ax.set_ylabel("Detachment force [N]")
    ax.set_title("Detachment force vs. canopy height")
    ax.legend(loc="upper right")
    # Place the stats box right under the legend so it never overlaps points.
    ax.text(0.96, 0.70,
            f"$r = {r_h:.2f}$, ${_format_p(p_h)}$, "
            f"$R^{{2}} = {r2_h:.2f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=13, family="serif",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="#bbb", lw=0.7, alpha=0.95))

    fig.tight_layout(pad=0.4)
    _save_both(fig, out_dir / "fruit_stats_force_vs_height")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
def main() -> int:
    out_dir = REPO / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_distribution(out_dir)
    _save_force_vs_mass(out_dir)
    _save_force_vs_position(out_dir)
    _save_force_vs_height(out_dir)

    print(f"4 figures saved to {out_dir}/")
    for name in ("fruit_stats_distribution",
                 "fruit_stats_force_vs_mass",
                 "fruit_stats_force_vs_position",
                 "fruit_stats_force_vs_height"):
        print(f"  {name}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
