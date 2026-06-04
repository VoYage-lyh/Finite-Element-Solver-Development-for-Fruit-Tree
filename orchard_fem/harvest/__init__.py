"""Fruit harvest detachment analysis."""
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
    element_peak_stress,
    extreme_fibre_distance,
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
]
