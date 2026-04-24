from __future__ import annotations

from enum import Enum


class MaterialModelKind(str, Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    ORTHOTROPIC_PLACEHOLDER = "orthotropic_placeholder"


class JointLawKind(str, Enum):
    NONE = "none"
    POLYNOMIAL = "polynomial"
    GAP_FRICTION = "gap_friction"


class ExcitationKind(str, Enum):
    HARMONIC_FORCE = "harmonic_force"
    HARMONIC_DISPLACEMENT = "harmonic_displacement"
    HARMONIC_ACCELERATION = "harmonic_acceleration"


class AnalysisMode(str, Enum):
    FREQUENCY_RESPONSE = "frequency_response"
    TIME_HISTORY = "time_history"
