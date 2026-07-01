"""Render a horizontal 4-column workflow diagram of the orchard-FEM
calibration → recommendation pipeline.

Output: ``results/summary/summary_workflow.{png,pdf}``

Layout invariants
-----------------
* Four side-by-side lanes (Modeling, Calibration, Validation, Recommendation).
* Each lane has its main vertical chain ending at a single **exit node**;
  all four exit nodes share the same Y coordinate so the inter-lane
  arrows are purely horizontal and never cross any text box.
* The cross-lane "posterior → CI propagation" link is routed along the
  top edge of the lanes (above all boxes) so it stays out of the way.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]


# ────────────────────────────────────────────────────────────────────────────
#  Style — Times New Roman everywhere
# ────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.default": "it",
    "font.size": 10,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ────────────────────────────────────────────────────────────────────────────
def _node(ax, x, y, w, h, text, *, fc="white", ec="#444",
          textcolor="#1a1a1a", fontsize=10, weight="normal"):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.4",
        ec=ec, fc=fc, lw=1.3, zorder=4,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, zorder=5,
            fontweight=weight, family="serif")
    return (x, y, w, h)


def _arrow_v(ax, src, dst, *, color="#444", lw=1.4):
    """Vertical arrow between two same-column boxes."""
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    p_from = (sx, sy - sh / 2)
    p_to = (dx, dy + dh / 2)
    arr = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>",
        color=color, lw=lw, mutation_scale=14, zorder=6,
    )
    ax.add_patch(arr)


def _arrow_h(ax, src, dst, *, color="#444", lw=2.0):
    """Horizontal arrow between two boxes (left → right)."""
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    p_from = (sx + sw / 2, sy)
    p_to = (dx - dw / 2, dy)
    arr = FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>",
        color=color, lw=lw, mutation_scale=16, zorder=6,
    )
    ax.add_patch(arr)


def _arrow_routed_overhead(ax, src, dst, *,
                            exit_gutter_x, entry_gutter_x, rail_y,
                            color="#666", lw=1.1, linestyle=(0, (5, 3))):
    """Manhattan-routed dashed arrow that stays *outside* every node:

    src_right → exit_gutter → up to rail → across → down to dst_y → into dst_left
    """
    sx, sy, sw, sh = src
    dx, dy, dw, dh = dst
    src_right = (sx + sw / 2, sy)
    dst_left = (dx - dw / 2, dy)
    p1 = (exit_gutter_x, sy)
    p2 = (exit_gutter_x, rail_y)
    p3 = (entry_gutter_x, rail_y)
    p4 = (entry_gutter_x, dy)
    for a, b in ((src_right, p1), (p1, p2), (p2, p3), (p3, p4)):
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=color, lw=lw, linestyle=linestyle,
                solid_capstyle="round", zorder=5)
    arr = FancyArrowPatch(
        p4, dst_left, arrowstyle="-|>",
        color=color, lw=lw, mutation_scale=12, zorder=6,
        linestyle=linestyle,
    )
    ax.add_patch(arr)


def _lane(ax, x_left, x_right, y_top, y_bot, label, *, fc, ec):
    rect = FancyBboxPatch(
        (x_left, y_bot), x_right - x_left, y_top - y_bot,
        boxstyle="round,pad=0.5,rounding_size=2.0",
        ec=ec, fc=fc, lw=1.0, alpha=0.50, zorder=1,
    )
    ax.add_patch(rect)
    # Two-line lane title so the longest names ("Recommendation") fit inside.
    if "·" in label:
        head, body = [s.strip() for s in label.split("·", 1)]
    else:
        head, body = label, ""
    ax.text((x_left + x_right) / 2, y_top - 1.5, head,
            ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=ec, zorder=2,
            family="serif")
    if body:
        ax.text((x_left + x_right) / 2, y_top - 4.5, body,
                ha="center", va="top",
                fontsize=12.5, fontweight="bold", color=ec, zorder=2,
                family="serif")


# ────────────────────────────────────────────────────────────────────────────
#  Diagram
# ────────────────────────────────────────────────────────────────────────────
def main() -> int:
    fig, ax = plt.subplots(figsize=(15.2, 8.0))
    ax.set_xlim(0, 200)
    ax.set_ylim(5, 95)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Lane geometry ───────────────────────────────────────────────────────
    lane_w = 46
    lane_gap = 4
    lane_top = 92
    lane_bot = 8
    x_starts = [3 + i * (lane_w + lane_gap) for i in range(4)]
    lane_centers = [x0 + lane_w / 2 for x0 in x_starts]

    lanes = [
        ("Stage 1 · Modeling",       "#E8F0FE", "#3F60A0"),
        ("Stage 2 · Calibration",    "#FCE4EC", "#A04D6A"),
        ("Stage 3 · Validation",     "#FFF8E1", "#A38033"),
        ("Stage 4 · Recommendation", "#E8F5E9", "#3F7A55"),
    ]
    for x0, (lbl, fc, ec) in zip(x_starts, lanes):
        _lane(ax, x0, x0 + lane_w, lane_top, lane_bot, lbl, fc=fc, ec=ec)

    # Common Y for inter-lane arrows: every lane's *exit* node sits here.
    exit_y = 20

    node_w = lane_w - 4  # full-width inside lane (a bit more room)
    body_fs = 11.0       # body-text font size in nodes

    # ── Stage 1: Modeling ──────────────────────────────────────────────────
    cx = lane_centers[0]
    s1_a = _node(ax, cx, 77, node_w, 7.5,
                 "Tree morphology measurement\n"
                 r"$\mathcal{G} = (\mathcal{B}, \mathcal{E})$",
                 ec="#3F60A0", fontsize=body_fs)
    s1_b = _node(ax, cx, 66, node_w, 7.5,
                 "Branch material tests\n"
                 r"$\rightarrow$ priors $p(\theta)$",
                 ec="#3F60A0", fontsize=body_fs)
    s1_c = _node(ax, cx, 55, node_w, 7.5,
                 "Fruit distribution model\n"
                 "(linear arc-length density)",
                 ec="#3F60A0", fontsize=body_fs)
    s1_d = _node(ax, cx, 44, node_w, 7.5,
                 "Assemble FE operators\n"
                 r"$\mathbf{M},\,\mathbf{K}(\theta),\,\mathbf{C}(\alpha,\beta)$",
                 ec="#3F60A0", fontsize=body_fs)
    s1_e = _node(ax, cx, 33, node_w, 7.5,
                 "Local nonlinear correction\n"
                 r"$k_3 \Delta u^{3} + c_2 |\Delta v|\Delta v$",
                 ec="#3F60A0", fontsize=body_fs)
    s1_exit = _node(ax, cx, exit_y, node_w, 8,
                    "Prior-constrained\nFE model",
                    fc="white", ec="#3F60A0", weight="bold", fontsize=13)
    _arrow_v(ax, s1_a, s1_b, color="#3F60A0")
    _arrow_v(ax, s1_b, s1_c, color="#3F60A0")
    _arrow_v(ax, s1_c, s1_d, color="#3F60A0")
    _arrow_v(ax, s1_d, s1_e, color="#3F60A0")
    _arrow_v(ax, s1_e, s1_exit, color="#3F60A0")

    # ── Stage 2: Calibration ───────────────────────────────────────────────
    cx = lane_centers[1]
    s2_a = _node(ax, cx, 77, node_w, 7.5,
                 "Hammer test\n"
                 "(force + accel. signals)",
                 ec="#A04D6A", fontsize=body_fs)
    s2_b = _node(ax, cx, 66, node_w, 7.5,
                 "Butterworth LP filter\n"
                 r"$f_c=50$ Hz, zero-phase",
                 ec="#A04D6A", fontsize=body_fs)
    s2_c = _node(ax, cx, 55, node_w, 7.5,
                 r"$H_{1}$ FRF estimate" + "\n" +
                 r"$H_{1} = G_{\mathrm{af}}/G_{\mathrm{ff}}$",
                 ec="#A04D6A", fontsize=body_fs)
    s2_d = _node(ax, cx, 44, node_w, 7.5,
                 "Peak picking\n"
                 r"$\hat{f}_{r}$, $|\hat{H}(\omega_{i})|$",
                 ec="#A04D6A", fontsize=body_fs)
    s2_e = _node(ax, cx, 33, node_w, 7.5,
                 "MCMC calibration (emcee)\n"
                 r"$\theta \in \mathbb{R}^{8}$",
                 ec="#A04D6A", fontsize=body_fs)
    s2_exit = _node(ax, cx, exit_y, node_w, 8,
                    "Posterior\n"
                    r"$p(\theta \mid \hat{f}, \hat{H})$",
                    fc="white", ec="#A04D6A", weight="bold", fontsize=13)
    _arrow_v(ax, s2_a, s2_b, color="#A04D6A")
    _arrow_v(ax, s2_b, s2_c, color="#A04D6A")
    _arrow_v(ax, s2_c, s2_d, color="#A04D6A")
    _arrow_v(ax, s2_d, s2_e, color="#A04D6A")
    _arrow_v(ax, s2_e, s2_exit, color="#A04D6A")

    # ── Stage 3: Validation ────────────────────────────────────────────────
    cx = lane_centers[2]
    s3_a = _node(ax, cx, 77, node_w, 7.5,
                 "Fixed-frequency\nexcitation test",
                 ec="#A38033", fontsize=body_fs)
    s3_b = _node(ax, cx, 66, node_w, 7.5,
                 r"Forward solve at $\hat{\theta}$" + "\n"
                 "(posterior median)",
                 ec="#A38033", fontsize=body_fs)
    s3_c = _node(ax, cx, 55, node_w, 7.5,
                 "Posterior predictive FRF\n"
                 "(NRMSE, peak error)",
                 ec="#A38033", fontsize=body_fs)
    s3_d = _node(ax, cx, 44, node_w, 7.5,
                 "90% CI coverage check\n"
                 "vs. independent data",
                 ec="#A38033", fontsize=body_fs)
    s3_e = _node(ax, cx, 33, node_w, 7.5,
                 "Linear vs nonlinear FRF\n"
                 r"$\Delta f / f_{\rm lin}$, peak shape",
                 ec="#A38033", fontsize=body_fs)
    s3_exit = _node(ax, cx, exit_y, node_w, 8,
                    "Validated\ncalibrated model",
                    fc="white", ec="#A38033", weight="bold", fontsize=13)
    _arrow_v(ax, s3_a, s3_b, color="#A38033")
    _arrow_v(ax, s3_b, s3_c, color="#A38033")
    _arrow_v(ax, s3_c, s3_d, color="#A38033")
    _arrow_v(ax, s3_d, s3_e, color="#A38033")
    _arrow_v(ax, s3_e, s3_exit, color="#A38033")

    # ── Stage 4: Recommendation ────────────────────────────────────────────
    cx = lane_centers[3]
    s4_a = _node(ax, cx, 77, node_w, 7.5,
                 r"$(p, f, A)$ sweep" + "\n"
                 r"$\mathcal{P}_{0} \times \mathcal{F} \times \mathcal{A}$",
                 ec="#3F7A55", fontsize=body_fs)
    s4_b = _node(ax, cx, 66, node_w, 7.5,
                 "Multi-clamp Pareto\n"
                 r"$C_{\mathrm{cov}}$ vs $\sigma_{\mathrm{tr}}$",
                 ec="#3F7A55", fontsize=body_fs)
    s4_c = _node(ax, cx, 55, node_w, 7.5,
                 "Knee selection\n"
                 r"$\min \|(\tilde{y}_{1}, \tilde{y}_{2})\|_{2}$",
                 ec="#3F7A55", fontsize=body_fs)
    s4_d = _node(ax, cx, 44, node_w, 7.5,
                 "Posterior + Sobol\n"
                 r"90% CI, $S_{T,i}$",
                 ec="#3F7A55", fontsize=body_fs)
    s4_e = _node(ax, cx, 33, node_w, 7.5,
                 "Operating-parameter sequence\n"
                 r"$\{(f_k, A_k)\}_{k=1}^{N},\; C_{\mathrm{cov}}^{\rm cum}\!\uparrow$",
                 ec="#3F7A55", fontsize=body_fs)
    s4_exit = _node(
        ax, cx, exit_y, node_w, 8,
        "Staged recommendation\n"
        r"$(p^{\dagger},\,\{(f_k, A_k)\}_{k=1}^{N}) \pm$ CI",
        fc="white", ec="#3F7A55", weight="bold", fontsize=12,
    )
    _arrow_v(ax, s4_a, s4_b, color="#3F7A55")
    _arrow_v(ax, s4_b, s4_c, color="#3F7A55")
    _arrow_v(ax, s4_c, s4_d, color="#3F7A55")
    _arrow_v(ax, s4_d, s4_e, color="#3F7A55")
    _arrow_v(ax, s4_e, s4_exit, color="#3F7A55")

    # ── Inter-lane main arrows: all at y = exit_y, perfectly horizontal ────
    _arrow_h(ax, s1_exit, s2_exit, color="#222", lw=2.0)
    _arrow_h(ax, s2_exit, s3_exit, color="#222", lw=2.0)
    _arrow_h(ax, s3_exit, s4_exit, color="#222", lw=2.0)

    # ── Cross-lane uncertainty rail: Stage 2 posterior → Stage 4 propagation.
    # Manhattan-routed through the gutters between lanes (never crossing
    # any node interior).
    gutter_2_3 = (x_starts[1] + lane_w + x_starts[2]) / 2  # centre of gap
    gutter_3_4 = (x_starts[2] + lane_w + x_starts[3]) / 2
    rail_y = 95   # above lane tops (92), below title (99)
    _arrow_routed_overhead(
        ax, s2_exit, s4_d,
        exit_gutter_x=gutter_2_3,
        entry_gutter_x=gutter_3_4,
        rail_y=rail_y,
        color="#666", lw=1.1, linestyle=(0, (5, 3)),
    )

    out_dir = REPO / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "summary_workflow"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved: {stem}.png and {stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
