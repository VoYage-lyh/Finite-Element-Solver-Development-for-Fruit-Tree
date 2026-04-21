"""FEniCSx/PETSc/SLEPc-facing coordination layer for OrchardVibrationSolver.

The orchard domain model stays project-specific, while numerical solver
responsibilities migrate away from handwritten dense routines toward the
PETSc/SLEPc backend used in the FEniCSx stack.
"""

from orchard_fem.io.legacy_loader import load_orchard_model
from orchard_fem.model import OrchardModel
from orchard_fem.topology.tree import BranchPath, ObservationPoint, TreeTopology, Vec3

__all__ = [
    "BranchPath",
    "ObservationPoint",
    "OrchardModel",
    "TreeTopology",
    "Vec3",
    "load_orchard_model",
]
