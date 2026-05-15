"""Raw hammer-test CSV → averaged FRF + modal-frequency identification.

Bridges :class:`HammerRecord` to the persistent ``T<i>_frf.csv`` format the
``recommend`` workflow consumes. Two input layouts are supported:

* Three-column CSVs (one file per hit) — ``time_s, force_N, response``
* Multi-channel single CSV — pass ``--force-column`` and ``--response-column``
  and the function splits each file by hit using a manually supplied
  ``--hit-segments`` list.

Outputs a 2-column CSV (``frequency_hz, magnitude``) plus a sidecar JSON
describing the identified first ``n_modes`` frequencies.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from orchard_fem.processing.frf_processing import (
    HammerRecord,
    estimate_h1_average,
    identify_modes,
)


def load_hammer_csv_pair(
    path: Path | str,
    *,
    time_column: int = 0,
    force_column: int = 1,
    response_column: int = 2,
    delimiter: str = ",",
    skip_header: int = 1,
) -> HammerRecord:
    """Load one hit file with explicit column indices.

    Returns
    -------
    HammerRecord
        Sample rate is inferred from the median time-step of *time_column*.
    """
    data = np.loadtxt(str(path), delimiter=delimiter, skiprows=skip_header)
    if data.ndim != 2 or data.shape[1] <= max(time_column, force_column, response_column):
        raise ValueError(f"Bad CSV layout in {path}; got shape {data.shape}.")
    t = data[:, time_column]
    dt = float(np.median(np.diff(t)))
    if dt <= 0.0:
        raise ValueError(f"Non-monotonic time column in {path}.")
    return HammerRecord(
        force=data[:, force_column].astype(float),
        response=data[:, response_column].astype(float),
        sample_rate_hz=1.0 / dt,
        label=Path(path).stem,
    )


def load_hammer_records(
    paths: Iterable[Path | str],
    **csv_kwargs,
) -> list[HammerRecord]:
    """Load a sequence of hammer CSVs into :class:`HammerRecord` objects."""
    return [load_hammer_csv_pair(p, **csv_kwargs) for p in paths]


def write_frf_csv(
    output_path: Path | str,
    frequencies_hz: np.ndarray,
    magnitudes: np.ndarray,
    *,
    coherence: np.ndarray | None = None,
) -> None:
    """Write a 2-column (or 3-column with coherence) FRF CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [frequencies_hz, magnitudes]
    header = "frequency_hz,magnitude"
    if coherence is not None:
        cols.append(coherence)
        header += ",coherence"
    data = np.column_stack(cols)
    np.savetxt(str(output_path), data, delimiter=",", header=header,
               comments="")


def process_hammer_to_frf(
    paths: Iterable[Path | str],
    *,
    output_csv: Path | str,
    modal_sidecar_json: Path | str | None = None,
    gamma_min: float = 0.8,
    nperseg: int | None = None,
    n_modes: int = 3,
    frequency_band_hz: tuple[float, float] = (1.0, 30.0),
    **csv_kwargs,
) -> dict:
    """End-to-end: load N hit CSVs → averaged FRF + modal identification.

    Parameters
    ----------
    paths:
        Iterable of CSV paths (one per impact hit).
    output_csv:
        Destination for the merged ``frequency_hz,magnitude[,coherence]`` CSV.
    modal_sidecar_json:
        If set, write a JSON file with identified modal frequencies and
        damping ratios alongside the FRF — useful as direct input to
        ``orchard-fem recommend --measured-modal ...``.
    gamma_min, nperseg, n_modes, frequency_band_hz:
        Forwarded to :func:`estimate_h1_average` and :func:`identify_modes`.

    Returns
    -------
    dict
        Summary with frequencies, magnitudes, identified modes, and the
        rejection count from the coherence filter.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("process_hammer_to_frf requires at least one CSV path.")
    records = load_hammer_records(paths, **csv_kwargs)
    frf = estimate_h1_average(records, gamma_min=gamma_min, nperseg=nperseg)
    write_frf_csv(output_csv, frf.frequencies_hz, frf.magnitude,
                  coherence=frf.coherence)
    modes = identify_modes(frf, n_modes=n_modes,
                            frequency_band_hz=frequency_band_hz)
    summary = {
        "input_paths": [str(p) for p in paths],
        "n_records_used": frf.n_averaged,
        "n_records_total": len(records),
        "output_csv": str(output_csv),
        "modes": [asdict(m) for m in modes.modes],
    }
    if modal_sidecar_json is not None:
        sidecar = Path(modal_sidecar_json)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


__all__ = [
    "load_hammer_csv_pair",
    "load_hammer_records",
    "process_hammer_to_frf",
    "write_frf_csv",
]
