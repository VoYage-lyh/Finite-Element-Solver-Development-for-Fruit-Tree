"""Rayleigh damping coefficient helpers.

For proportional damping ``C = α M + β K``, the modal damping ratio of the
``r``-th mode (natural angular frequency ``ω_r``) is::

    ζ_r = (1 / 2) (α / ω_r + β ω_r)

Given two target ratios ``(ζ_1, ζ_2)`` at frequencies ``(ω_1, ω_2)`` we solve
the resulting 2×2 linear system for ``(α, β)``.  This is the standard two-mode
fit used in vibration-harvesting FEM models so the Bayesian calibration can
parameterise damping with physically interpretable modal ratios.

Example::

    from orchard_fem.dynamics.rayleigh import rayleigh_from_modal_damping
    alpha, beta = rayleigh_from_modal_damping(
        zeta1=0.05, zeta2=0.04,
        omega1=2 * math.pi * 5.5,
        omega2=2 * math.pi * 10.3,
    )
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RayleighCoefficients:
    """Pair of Rayleigh damping coefficients ``(α, β)``.

    Parameters
    ----------
    alpha:
        Mass-proportional coefficient [1/s].
    beta:
        Stiffness-proportional coefficient [s].
    """

    alpha: float
    beta: float

    def modal_damping_ratio(self, omega: float) -> float:
        """Return the modal damping ratio at angular frequency *omega*."""
        if omega <= 0.0:
            raise ValueError("omega must be positive")
        return 0.5 * (self.alpha / omega + self.beta * omega)


def rayleigh_from_modal_damping(
    *,
    zeta1: float,
    zeta2: float,
    omega1: float,
    omega2: float,
) -> RayleighCoefficients:
    """Solve for ``(α, β)`` from two modal damping ratios at two frequencies.

    The system to invert is::

        [ 1/(2 ω_1)   ω_1/2 ] [ α ]   [ ζ_1 ]
        [ 1/(2 ω_2)   ω_2/2 ] [ β ] = [ ζ_2 ]

    Parameters
    ----------
    zeta1, zeta2:
        Target modal damping ratios (dimensionless, typically in [0, 0.2]).
    omega1, omega2:
        Natural angular frequencies [rad/s] for the two anchor modes.

    Returns
    -------
    RayleighCoefficients
        Coefficients reproducing the two ratios exactly.

    Raises
    ------
    ValueError
        If either angular frequency is non-positive or the two anchors
        coincide (singular system).
    """
    if omega1 <= 0.0 or omega2 <= 0.0:
        raise ValueError("Angular frequencies must be positive.")
    if abs(omega1 - omega2) < 1.0e-12:
        raise ValueError("omega1 and omega2 must differ to invert the system.")

    # Closed-form solution of the 2×2 system ζ_r = (α/ω_r + β ω_r) / 2.
    denom = omega2 * omega2 - omega1 * omega1
    alpha = 2.0 * omega1 * omega2 * (zeta1 * omega2 - zeta2 * omega1) / denom
    beta = 2.0 * (zeta2 * omega2 - zeta1 * omega1) / denom
    return RayleighCoefficients(alpha=float(alpha), beta=float(beta))


def rayleigh_from_modal_damping_hz(
    *,
    zeta1: float,
    zeta2: float,
    f1_hz: float,
    f2_hz: float,
) -> RayleighCoefficients:
    """Convenience wrapper accepting frequencies in Hz instead of rad/s."""
    two_pi = 2.0 * math.pi
    return rayleigh_from_modal_damping(
        zeta1=zeta1,
        zeta2=zeta2,
        omega1=two_pi * f1_hz,
        omega2=two_pi * f2_hz,
    )


__all__ = [
    "RayleighCoefficients",
    "rayleigh_from_modal_damping",
    "rayleigh_from_modal_damping_hz",
]
