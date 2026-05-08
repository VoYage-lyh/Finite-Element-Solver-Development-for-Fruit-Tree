from __future__ import annotations

from dataclasses import dataclass


Matrix = list[list[float]]


@dataclass(frozen=True)
class BeamElementProperties:
    youngs_modulus: float
    shear_modulus: float
    area: float
    iy: float
    iz: float
    torsion_constant: float
    density: float
    length: float
    axial_rigidity: float | None = None
    torsional_rigidity: float | None = None
    bending_rigidity_y: float | None = None
    bending_rigidity_z: float | None = None
