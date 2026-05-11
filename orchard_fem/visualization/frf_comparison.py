"""Publication-ready FRF comparison and detachment spectrum plots.

All matplotlib imports are deferred to function bodies so this module loads
cleanly in environments without matplotlib installed.

Example::

    from orchard_fem.visualization.frf_comparison import (
        plot_frf_comparison,
        plot_detachment_spectrum,
    )

    fig = plot_frf_comparison(comparison, title="Apple branch A", output_path="frf.pdf")
    fig2 = plot_detachment_spectrum(spectrum, output_path="detach.pdf")
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchard_fem.harvest.detachment import DetachmentSpectrum
    from orchard_fem.io.measurement import FRFComparison


def plot_frf_comparison(
    comparison: "FRFComparison",
    *,
    title: str = "",
    output_path: str | Path | None = None,
) -> Any:
    """Plot measured vs. simulated FRF as a two-subplot figure.

    Upper subplot: magnitude [dB re 1 m/s²/N] vs frequency [Hz].
    Lower subplot: phase [°] vs frequency [Hz] (only if phase data is available in
    the underlying ``FRFComparison``; otherwise shows residual magnitude).

    Parameters
    ----------
    comparison:
        :class:`~orchard_fem.io.measurement.FRFComparison` with measured and
        simulated magnitude arrays.
    title:
        Optional figure title.
    output_path:
        If given, saves the figure to this path and also returns the Figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib and numpy are required for plot_frf_comparison. "
            "Install with: pip install matplotlib numpy"
        ) from exc

    freqs = np.asarray(comparison.frequencies_hz)
    meas = np.asarray(comparison.measured_magnitude)
    sim = np.asarray(comparison.simulated_magnitude)

    # Convert to dB (avoid log of zero)
    _eps = 1.0e-30
    meas_db = 20.0 * np.log10(np.maximum(meas, _eps))
    sim_db = 20.0 * np.log10(np.maximum(sim, _eps))
    residual_db = meas_db - sim_db

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_mag, ax_phase = axes

    ax_mag.plot(freqs, meas_db, color="#1f77b4", linewidth=1.5, label="Measured")
    ax_mag.plot(
        freqs, sim_db, color="#ff7f0e", linewidth=1.5, linestyle="--", label="Simulated"
    )
    ax_mag.set_ylabel("FRF magnitude [dB re 1 m/s²/N]")
    ax_mag.legend(loc="upper right", fontsize=9)
    ax_mag.grid(True, linestyle=":", alpha=0.6)
    ax_mag.annotate(
        f"MAC = {comparison.mac_value:.3f}",
        xy=(0.02, 0.05),
        xycoords="axes fraction",
        fontsize=9,
        color="#333333",
    )

    ax_phase.plot(
        freqs, residual_db, color="#2ca02c", linewidth=1.0, label="Residual (meas − sim)"
    )
    ax_phase.axhline(0.0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax_phase.set_ylabel("Residual [dB]")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.legend(loc="upper right", fontsize=9)
    ax_phase.grid(True, linestyle=":", alpha=0.6)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig


def plot_detachment_spectrum(
    spectrum: "DetachmentSpectrum",
    *,
    highlight_optimum: bool = True,
    n_top: int = 3,
    title: str = "",
    output_path: str | Path | None = None,
) -> Any:
    """Plot detachment fraction vs. excitation frequency.

    Parameters
    ----------
    spectrum:
        :class:`~orchard_fem.harvest.detachment.DetachmentSpectrum`.
    highlight_optimum:
        If True, draw a vertical dashed line at the frequency with maximum
        detachment and shade the top-*n_top* candidate frequencies.
    n_top:
        Number of top frequencies to shade.
    title:
        Optional figure title.
    output_path:
        If given, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib and numpy are required for plot_detachment_spectrum. "
            "Install with: pip install matplotlib numpy"
        ) from exc

    freqs = np.asarray(spectrum.frequencies_hz)
    fracs = np.asarray(spectrum.detachment_fractions)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(freqs, fracs * 100.0, color="#1f77b4", linewidth=1.5)
    ax.fill_between(freqs, fracs * 100.0, alpha=0.15, color="#1f77b4")

    if highlight_optimum and len(freqs) > 0:
        sorted_idx = np.argsort(fracs)[::-1]
        top_idx = sorted_idx[:n_top]

        # Shade top-N
        for idx in top_idx:
            half_bw = (freqs[-1] - freqs[0]) / max(len(freqs) - 1, 1) * 2
            ax.axvspan(
                freqs[idx] - half_bw,
                freqs[idx] + half_bw,
                alpha=0.15,
                color="#ff7f0e",
            )

        # Best frequency vertical line
        best_idx = sorted_idx[0]
        ax.axvline(
            freqs[best_idx],
            color="#d62728",
            linewidth=1.5,
            linestyle="--",
            label=f"Best: {freqs[best_idx]:.1f} Hz  ({fracs[best_idx]*100:.0f}%)",
        )
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Detachment fraction [%]")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", alpha=0.6)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig
