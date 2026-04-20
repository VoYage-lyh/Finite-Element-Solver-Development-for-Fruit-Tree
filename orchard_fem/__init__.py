"""Python coordination layer for OrchardVibrationSolver.

This package is introduced to host the FEniCSx/PETSc-facing workflow without
displacing the existing C++ project core. The current C++ solver remains the
reference backend while Python modules are added incrementally.
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
