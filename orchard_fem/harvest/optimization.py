"""Harvest excitation frequency optimization.

Finds the top-N frequencies that maximise the detachment fraction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchard_fem.domain import OrchardModel
    from orchard_fem.harvest.detachment import DetachmentSpectrum
    from orchard_fem.workflows.batch_excitation import BatchExcitationResult


def optimize_harvest_excitation(
    model: "OrchardModel",
    batch_results: "list[BatchExcitationResult]",
    *,
    n_top: int = 3,
    detachment_displacement_m: float | None = None,
) -> list[tuple[float, float]]:
    """Find the frequencies that maximise fruit detachment fraction.

    Parameters
    ----------
    model:
        Orchard model with fruit attachments.
    batch_results:
        Batch FRF results.
    n_top:
        Number of top candidate frequencies to return.
    detachment_displacement_m:
        Override detachment displacement [m].

    Returns
    -------
    list[tuple[float, float]]
        Up to *n_top* ``(frequency_hz, detachment_fraction)`` pairs,
        sorted by decreasing detachment fraction.
    """
    from orchard_fem.harvest.detachment import compute_detachment_spectrum

    spectrum = compute_detachment_spectrum(
        model,
        batch_results,
        detachment_displacement_m=detachment_displacement_m,
    )
    return _top_n_from_spectrum(spectrum, n_top)


def _top_n_from_spectrum(
    spectrum: "DetachmentSpectrum",
    n_top: int,
) -> list[tuple[float, float]]:
    """Extract the top-N (freq, fraction) pairs from a spectrum."""
    pairs = sorted(
        zip(spectrum.frequencies_hz, spectrum.detachment_fractions),
        key=lambda x: x[1],
        reverse=True,
    )
    return list(pairs[:n_top])
