"""Experimental time-series post-processing (FRF identification, modal ID)."""
from orchard_fem.processing.frf_processing import (
    FRFEstimate,
    HammerRecord,
    ModalParameters,
    coherence_filter,
    estimate_h1,
    estimate_h1_average,
    half_power_bandwidth,
    identify_modes,
)
from orchard_fem.processing.hammer_io import (
    load_hammer_csv_pair,
    load_hammer_records,
    process_hammer_to_frf,
    write_frf_csv,
)

__all__ = [
    "FRFEstimate",
    "HammerRecord",
    "ModalParameters",
    "coherence_filter",
    "estimate_h1",
    "estimate_h1_average",
    "half_power_bandwidth",
    "identify_modes",
    "load_hammer_csv_pair",
    "load_hammer_records",
    "process_hammer_to_frf",
    "write_frf_csv",
]
