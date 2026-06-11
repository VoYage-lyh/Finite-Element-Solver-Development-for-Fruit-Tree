"""Fruit harvest detachment analysis."""
from orchard_fem.harvest.basin import (
    BasinResult,
    DuffingElement,
    compute_basin_ccm,
    integrity_factor,
    steady_amplitude,
)
from orchard_fem.harvest.detachment import (
    DetachmentResult,
    DetachmentSpectrum,
    FruitDetachmentState,
    compute_detachment_at_frequency,
    compute_detachment_spectrum,
)
from orchard_fem.harvest.objective import (
    DetachmentFatigueLaw,
    HarvestObjectiveConfig,
    HarvestObjectiveResult,
    HarvestParameters,
    StressFatigueLaw,
    evaluate_harvest_objective,
    load_ratios_from_detachment,
    scale_stress_with_amplitude,
)
from orchard_fem.harvest.optimization import optimize_harvest_excitation
from orchard_fem.harvest.stress_recovery import (
    ElementEndForces,
    NetworkStressResult,
    clamp_stress_from_solution,
    element_peak_stress,
    element_stress_from_solution,
    extreme_fibre_distance,
    network_peak_stress,
    recover_element_end_forces,
)

__all__ = [
    "DetachmentResult",
    "DetachmentSpectrum",
    "FruitDetachmentState",
    "compute_detachment_at_frequency",
    "compute_detachment_spectrum",
    "optimize_harvest_excitation",
    # objective (time-dependent, two-tier damage)
    "DetachmentFatigueLaw",
    "StressFatigueLaw",
    "HarvestParameters",
    "HarvestObjectiveConfig",
    "HarvestObjectiveResult",
    "evaluate_harvest_objective",
    "load_ratios_from_detachment",
    "scale_stress_with_amplitude",
    # stress recovery (element bending stress for fracture/clamp tiers)
    "ElementEndForces",
    "recover_element_end_forces",
    "element_peak_stress",
    "extreme_fibre_distance",
    # network-level stress aggregation
    "NetworkStressResult",
    "element_stress_from_solution",
    "network_peak_stress",
    "clamp_stress_from_solution",
    # basin of attraction + integrity factor
    "DuffingElement",
    "BasinResult",
    "compute_basin_ccm",
    "integrity_factor",
    "steady_amplitude",
]
