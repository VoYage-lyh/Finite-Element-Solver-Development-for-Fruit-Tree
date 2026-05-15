"""Publication-ready figures for the parameter-uncertainty harvesting paper.

All functions take pre-computed result objects and return a ``matplotlib.Figure``
so the caller can ``fig.savefig(...)``. Matplotlib is imported lazily so the
module is safe to load in headless / minimal environments.

Figures provided:

* :func:`plot_posterior_corner`            — Fig 5 (parameter joint distribution)
* :func:`plot_posterior_predictive_frf`    — Fig 6 (measured FRF + posterior CI band)
* :func:`plot_pareto_scatter`              — Fig 7 (3 subplots, one per clamp point)
* :func:`plot_cross_tree_recommendations`  — Fig 8 ((f*, A*) per tree with error bars)
* :func:`plot_sobol_bars`                  — Fig 9 (S_i and S_T,i horizontal bars)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _apply_publication_style(plt) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for paper figures. "
            "Install with: pip install matplotlib"
        ) from exc
    return plt


# ────────────────────────────────────────────────────────────────────────────
#  Fig 5 — Corner plot of posterior joint distribution
# ────────────────────────────────────────────────────────────────────────────
def plot_posterior_corner(
    posterior,  # PosteriorResult
    *,
    parameter_labels: dict[str, str] | None = None,
    n_bins: int = 32,
    log_axes: Sequence[str] = (),
    figsize: tuple[float, float] | None = None,
):
    """Render a corner plot of the post-burn posterior samples.

    Parameters
    ----------
    posterior:
        :class:`PosteriorResult` from :func:`run_emcee_calibration`.
    parameter_labels:
        Optional override for axis labels, e.g. ``{"E": "$E$ [GPa]"}``.
        Defaults to the parameter name.
    n_bins:
        Histogram bins for diagonal panels.
    log_axes:
        Parameter names whose axes should be drawn on log scale.
    """
    plt = _require_pyplot()
    _apply_publication_style(plt)

    samples = posterior.flatten()
    names = posterior.parameter_names
    n = len(names)
    labels = {name: name for name in names}
    if parameter_labels:
        labels.update(parameter_labels)

    if figsize is None:
        figsize = (2.0 * n, 2.0 * n)
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = np.array([[axes]])

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            x = samples[:, j]
            if i == j:
                ax.hist(x, bins=n_bins, color="#2166AC", alpha=0.7,
                        density=True, edgecolor="white", linewidth=0.5)
                for q in (0.05, 0.5, 0.95):
                    ax.axvline(np.quantile(x, q), color="#B2182B",
                               linestyle=("-" if q == 0.5 else "--"),
                               linewidth=1.0, alpha=0.8)
                if names[i] in log_axes:
                    ax.set_xscale("log")
            else:
                y = samples[:, i]
                ax.hexbin(x, y, gridsize=24, cmap="Blues", mincnt=1)
                if names[j] in log_axes:
                    ax.set_xscale("log")
                if names[i] in log_axes:
                    ax.set_yscale("log")
            if i == n - 1:
                ax.set_xlabel(labels.get(names[j], names[j]))
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels.get(names[i], names[i]))
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)

    fig.suptitle("Posterior joint distribution", y=0.995, fontsize=14)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  Fig 6 — Posterior predictive FRF
# ────────────────────────────────────────────────────────────────────────────
def plot_posterior_predictive_frf(
    frequencies_hz: np.ndarray,
    measured_magnitude: np.ndarray,
    predictive_samples: np.ndarray,
    *,
    credible_alpha: float = 0.90,
    measured_phase_deg: np.ndarray | None = None,
    predictive_phase_samples: np.ndarray | None = None,
    modal_frequencies_hz: Sequence[float] | None = None,
    figsize: tuple[float, float] = (8.5, 5.5),
):
    """Plot measured FRF magnitude + posterior median + credible band.

    Parameters
    ----------
    frequencies_hz:
        Frequency grid shared by the measurement and the predictive samples.
    measured_magnitude:
        Measured |H(ω)|, same length as *frequencies_hz*.
    predictive_samples:
        Shape ``(K, N_freq)`` array of posterior-predictive magnitudes
        (``K`` posterior samples). Median + ``credible_alpha`` CI are derived
        on the fly.
    credible_alpha:
        Probability of the credible band (default 0.90 = 5%/95% quantiles).
    measured_phase_deg, predictive_phase_samples:
        Optional phase panel. If both are provided a second subplot is drawn.
    modal_frequencies_hz:
        Vertical dashed lines for natural-frequency markers.
    """
    plt = _require_pyplot()
    _apply_publication_style(plt)

    lo_q = (1.0 - credible_alpha) / 2.0
    hi_q = 1.0 - lo_q
    median = np.median(predictive_samples, axis=0)
    ci_low = np.quantile(predictive_samples, lo_q, axis=0)
    ci_high = np.quantile(predictive_samples, hi_q, axis=0)

    show_phase = (
        measured_phase_deg is not None
        and predictive_phase_samples is not None
    )
    if show_phase:
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw=dict(height_ratios=[2.3, 1.0]),
        )
    else:
        fig, ax_mag = plt.subplots(figsize=figsize)
        ax_phase = None

    ax_mag.fill_between(frequencies_hz, ci_low, ci_high,
                        color="#D94F3D", alpha=0.25,
                        label=f"Posterior {int(credible_alpha*100)}% CI")
    ax_mag.plot(frequencies_hz, median, color="#B2182B",
                linestyle="--", linewidth=1.6, label="Posterior median")
    ax_mag.plot(frequencies_hz, measured_magnitude,
                color="black", linewidth=1.5, label="Measured")
    ax_mag.set_ylabel(r"$|H(\omega)|$")
    ax_mag.set_yscale("log")
    ax_mag.grid(True, which="both", alpha=0.25)
    ax_mag.legend(loc="best")

    if modal_frequencies_hz:
        for f in modal_frequencies_hz:
            ax_mag.axvline(f, color="#666", linestyle=":", linewidth=0.9, alpha=0.7)

    if ax_phase is not None:
        med_p = np.median(predictive_phase_samples, axis=0)
        lo_p = np.quantile(predictive_phase_samples, lo_q, axis=0)
        hi_p = np.quantile(predictive_phase_samples, hi_q, axis=0)
        ax_phase.fill_between(frequencies_hz, lo_p, hi_p,
                              color="#D94F3D", alpha=0.25)
        ax_phase.plot(frequencies_hz, med_p,
                      color="#B2182B", linestyle="--", linewidth=1.4)
        ax_phase.plot(frequencies_hz, measured_phase_deg,
                      color="black", linewidth=1.3)
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.grid(True, alpha=0.25)
        ax_phase.set_xlabel("Frequency [Hz]")
    else:
        ax_mag.set_xlabel("Frequency [Hz]")

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  Fig 7 — Pareto scatter (one subplot per clamp point)
# ────────────────────────────────────────────────────────────────────────────
def plot_pareto_scatter(
    fronts,  # Sequence[ParetoFront]
    *,
    figsize: tuple[float, float] | None = None,
):
    """Plot Pareto front(s) in the (coverage, stress) plane.

    Each front gets its own subplot. The knee is marked with a red circle
    and the dominated cloud is shown faintly in the background. Stress is
    displayed on a MPa scale for readability.
    """
    plt = _require_pyplot()
    _apply_publication_style(plt)

    n = len(fronts)
    if figsize is None:
        figsize = (4.2 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, front in zip(axes, fronts):
        # objectives stored as (−coverage, stress) — flip back for plotting
        coverage = -front.objectives[:, 0]
        stress_MPa = front.objectives[:, 1] / 1.0e6

        # dominated cloud
        ax.scatter(coverage, stress_MPa, c="#cccccc", s=12, alpha=0.5,
                   edgecolors="none", label="Dominated")

        # non-dominated
        nd = front.non_dominated_index
        ax.scatter(coverage[nd], stress_MPa[nd], c="#2166AC",
                   s=42, edgecolors="black", linewidths=0.5,
                   label="Pareto front")

        # knee
        k = nd[front.knee_index]
        ax.scatter([coverage[k]], [stress_MPa[k]], s=200, marker="o",
                   facecolor="none", edgecolor="#B2182B", linewidth=2.0,
                   label="Knee", zorder=5)

        ax.set_xlabel("Detachment coverage")
        ax.set_ylabel(r"Trunk stress $\sigma_{\mathrm{max}}$ [MPa]")
        ax.set_title(front.clamp_node)
        ax.grid(True, alpha=0.25)
        if ax is axes[0]:
            ax.legend(loc="best")

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  Fig 8 — Cross-tree recommendation scatter with error bars
# ────────────────────────────────────────────────────────────────────────────
def plot_cross_tree_recommendations(
    recommendations: dict[str, "ParetoRecommendation"],  # noqa: F821
    *,
    figsize: tuple[float, float] = (6.4, 5.0),
    annotate: bool = True,
):
    """Plot recommended (f*, A*) per tree with 90% CI error bars.

    Marker size encodes the median ``a_tar``; tree labels are placed next to
    each point when ``annotate`` is True.
    """
    plt = _require_pyplot()
    _apply_publication_style(plt)
    fig, ax = plt.subplots(figsize=figsize)

    cov_max = max(rec.detachment_coverage_median for rec in recommendations.values())

    for label, rec in recommendations.items():
        size = 80.0 + 280.0 * (rec.detachment_coverage_median / max(cov_max, 1.0e-9))
        ax.errorbar(
            rec.frequency_hz_median, rec.amplitude_median,
            xerr=[[rec.frequency_hz_median - rec.frequency_hz_ci[0]],
                  [rec.frequency_hz_ci[1] - rec.frequency_hz_median]],
            yerr=[[rec.amplitude_median - rec.amplitude_ci[0]],
                  [rec.amplitude_ci[1] - rec.amplitude_median]],
            fmt="none", ecolor="#666", elinewidth=1.0, capsize=4, alpha=0.9,
            zorder=2,
        )
        ax.scatter([rec.frequency_hz_median], [rec.amplitude_median],
                   s=size, c="#2166AC", edgecolors="black", linewidth=0.8,
                   zorder=3)
        if annotate:
            ax.annotate(label,
                        (rec.frequency_hz_median, rec.amplitude_median),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=10, fontweight="bold")

    ax.set_xlabel("Recommended frequency $f^*$ [Hz]")
    ax.set_ylabel("Recommended amplitude $A^*$ [mm]")
    ax.set_title("Cross-tree Pareto-knee recommendations (90% CI)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  Fig 9 — Sobol horizontal bar chart
# ────────────────────────────────────────────────────────────────────────────
def plot_sobol_bars(
    sobol,  # SobolResult
    *,
    parameter_labels: dict[str, str] | None = None,
    top_k: int | None = None,
    figsize: tuple[float, float] = (7.5, 5.0),
):
    """Horizontal bar chart of S_i and S_T,i with bootstrap CI as error bars.

    Parameters
    ----------
    sobol:
        :class:`SobolResult`.
    parameter_labels:
        Optional override of axis labels.
    top_k:
        Show only the *top_k* parameters by total-effect (default: all).
    """
    plt = _require_pyplot()
    _apply_publication_style(plt)
    ranked = sobol.ranked_by_total_effect()
    if top_k is not None:
        ranked = ranked[:top_k]

    names = [n for n, _ in ranked]
    labels = [
        parameter_labels.get(n, n) if parameter_labels else n for n in names
    ]
    s1 = [sobol.first_order[n].value for n in names]
    s1_err = [
        sobol.first_order[n].value - sobol.first_order[n].confidence_low for n in names
    ]
    st = [sobol.total_effect[n].value for n in names]
    st_err = [
        sobol.total_effect[n].value - sobol.total_effect[n].confidence_low for n in names
    ]

    y = np.arange(len(names))
    height = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y + height / 2, s1, height=height, color="#7B3F9E",
            xerr=s1_err, ecolor="#444", capsize=3, label="$S_i$")
    ax.barh(y - height / 2, st, height=height, color="#2166AC",
            xerr=st_err, ecolor="#444", capsize=3, label="$S_{T,i}$")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index")
    ax.set_title("Variance contribution to recommended frequency $f^*$")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
#  Convenience: save fig to disk
# ────────────────────────────────────────────────────────────────────────────
def savefig(fig, output_path: str | Path, *, dpi: int = 180,
            close_after: bool = True) -> None:
    """Save *fig* to *output_path* with tight bbox; optionally close."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    if close_after:
        import matplotlib.pyplot as plt
        plt.close(fig)


__all__ = [
    "plot_posterior_corner",
    "plot_posterior_predictive_frf",
    "plot_pareto_scatter",
    "plot_cross_tree_recommendations",
    "plot_sobol_bars",
    "savefig",
]
