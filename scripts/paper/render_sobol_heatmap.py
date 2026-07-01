"""Render Fig 15(b) — cross-tree heat map of total-effect Sobol indices.

Reads ``results/calibration/sobol_t{1..5}_indices.csv`` and draws a
parameters × trees heat map of $S_T$, with each cell annotated by its
numerical value. Missing trees are simply skipped (so the heat map
adapts to whatever has finished so far).

Output:
  ``results/calibration/fig15b_sobol_heatmap.{png,pdf}``
  ``results/calibration/sobol_st_table.csv``  (Table 6 data)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


PARAM_DESCRIPTIONS = {
    "E_factor": r"$E$",
    "rho_factor": r"$\rho$",
    "zeta_1": r"$\zeta_1$",
    "zeta_2": r"$\zeta_2$",
    "log10_kc": r"$k_c$",
    "log10_cc": r"$c_c$",
    "log10_kf": r"$k_f$",
    "log10_cf": r"$c_f$",
    "d_tr_root_factor": r"$d_{\rm tr,root}$",
    "L_tr_factor": r"$L_{\rm tr}$",
    "d_br_factor": r"$\bar{d}_{\rm br}$",
    "L_br_factor": r"$\bar{L}_{\rm br}$",
}


def _configure_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def load_indices(path: Path):
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return {r["parameter"]: float(r["ST"]) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trees", default="1,2,3,4,5",
        help="Comma-separated tree ids to include (default 1,2,3,4,5).",
    )
    parser.add_argument(
        "--in-dir", default="results/calibration",
    )
    args = parser.parse_args()

    in_dir = REPO / args.in_dir
    trees = [int(s) for s in args.trees.split(",")]

    # Load each tree's Sobol indices
    data: dict[int, dict[str, float]] = {}
    for t in trees:
        path = in_dir / f"sobol_t{t}_indices.csv"
        d = load_indices(path)
        if d is None:
            print(f"  [t{t}] missing {path.name}, skipped")
            continue
        data[t] = d
        print(f"  [t{t}] loaded {len(d)} parameters")
    if not data:
        raise SystemExit("No Sobol index files found.")

    # Param list — use order from first tree
    param_order = list(next(iter(data.values())).keys())
    available_trees = sorted(data.keys())

    # Build matrix
    n_t = len(available_trees)
    n_p = len(param_order)
    M = np.full((n_t, n_p), np.nan)
    for i, t in enumerate(available_trees):
        for j, name in enumerate(param_order):
            M[i, j] = data[t].get(name, np.nan)

    # Sort params by mean ST across trees (descending)
    means = np.nanmean(M, axis=0)
    order_cols = np.argsort(-means)
    M = M[:, order_cols]
    param_order_sorted = [param_order[j] for j in order_cols]

    # ----------------- Fig 15(b)
    _configure_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 0.7 + 0.45 * n_t))
    im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=max(0.5, np.nanmax(M)),
                   aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$S_T$ (total-effect Sobol index for $f_r$)", fontsize=10)

    ax.set_xticks(np.arange(n_p))
    ax.set_xticklabels(
        [PARAM_DESCRIPTIONS.get(p, p) for p in param_order_sorted],
        rotation=0, fontsize=11,
    )
    ax.set_yticks(np.arange(n_t))
    ax.set_yticklabels([f"T{t}" for t in available_trees], fontsize=11)
    ax.set_xlabel("Input parameter", fontsize=11)
    ax.set_ylabel("Tree", fontsize=11)
    ax.set_title(
        rf"Total-effect Sobol indices $S_T$ across "
        rf"{n_t} sample tree{'s' if n_t > 1 else ''}",
        fontsize=12,
    )

    # Annotate cells
    for i in range(n_t):
        for j in range(n_p):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            text_color = "white" if v < 0.20 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8.5)

    fig.tight_layout(pad=0.4)
    stem = in_dir / "fig15b_sobol_heatmap"
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nFig 15(b): {stem.relative_to(REPO)}.{{png,pdf}}")

    # ----------------- Table 6 (CSV)
    tbl_path = in_dir / "table6_sobol_st_cross_tree.csv"
    with tbl_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        # Header
        header = ["Tree"] + [PARAM_DESCRIPTIONS.get(p, p).replace("$", "").replace("\\rm ", "")
                              for p in param_order_sorted]
        w.writerow(header)
        for i, t in enumerate(available_trees):
            row = [f"T{t}"] + [f"{M[i, j]:.3f}" for j in range(n_p)]
            w.writerow(row)
        # Mean row
        w.writerow(["mean"] + [f"{means[order_cols[j]]:.3f}" for j in range(n_p)])
    print(f"Table 6:  {tbl_path.relative_to(REPO)}")

    # Print top-3 per tree
    print("\nTop-3 parameters per tree:")
    for i, t in enumerate(available_trees):
        idx = np.argsort(-M[i])[:3]
        bits = []
        for j in idx:
            bits.append(f"{param_order_sorted[j]}={M[i, j]:.3f}")
        print(f"  T{t}: " + ", ".join(bits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
