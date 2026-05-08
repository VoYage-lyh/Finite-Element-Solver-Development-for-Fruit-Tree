from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi, sqrt

from orchard_fem.discretization.types import NonlinearLinkDefinition, NonlinearLinkKind


@dataclass(frozen=True)
class FirstHarmonicLinkResponse:
    force_real: float
    force_imag: float
    tangent_rr: float
    tangent_ri: float
    tangent_ir: float
    tangent_ii: float


def first_harmonic_stiffness_and_derivative(
    link: NonlinearLinkDefinition,
    amplitude: float,
) -> tuple[float, float]:
    if link.kind == NonlinearLinkKind.CUBIC_SPRING:
        stiffness = 0.75 * link.cubic_stiffness * amplitude * amplitude
        derivative = 1.5 * link.cubic_stiffness * amplitude
        return stiffness, derivative

    if link.kind == NonlinearLinkKind.GAP_SPRING:
        threshold = max(float(link.gap_threshold), 0.0)
        if amplitude <= threshold or amplitude <= 1.0e-14:
            return 0.0, 0.0

        stiffness_delta = link.open_stiffness - link.linear_stiffness
        if abs(stiffness_delta) <= 1.0e-14:
            return 0.0, 0.0

        ratio = max(0.0, min(threshold / amplitude, 1.0))
        root = sqrt(max(1.0 - (ratio * ratio), 0.0))
        stiffness = (2.0 * stiffness_delta / pi) * (
            acos(ratio) - (ratio * root)
        )
        derivative = (4.0 * stiffness_delta * ratio * root) / (pi * amplitude)
        return stiffness, derivative

    raise ValueError(f"Unsupported nonlinear link kind: {link.kind}")


def first_harmonic_link_response(
    link: NonlinearLinkDefinition,
    relative_real: float,
    relative_imag: float,
) -> FirstHarmonicLinkResponse:
    amplitude = sqrt((relative_real * relative_real) + (relative_imag * relative_imag))
    stiffness, derivative = first_harmonic_stiffness_and_derivative(link, amplitude)
    if amplitude <= 1.0e-14:
        radial_tangent = 0.0
    else:
        radial_tangent = derivative / amplitude

    return FirstHarmonicLinkResponse(
        force_real=stiffness * relative_real,
        force_imag=stiffness * relative_imag,
        tangent_rr=stiffness + (radial_tangent * relative_real * relative_real),
        tangent_ri=radial_tangent * relative_real * relative_imag,
        tangent_ir=radial_tangent * relative_imag * relative_real,
        tangent_ii=stiffness + (radial_tangent * relative_imag * relative_imag),
    )


def relative_complex_components(
    real: list[float],
    imag: list[float],
    link: NonlinearLinkDefinition,
) -> tuple[float, float]:
    second_real = real[link.second_dof] if link.second_dof >= 0 else 0.0
    second_imag = imag[link.second_dof] if link.second_dof >= 0 else 0.0
    return real[link.first_dof] - second_real, imag[link.first_dof] - second_imag
