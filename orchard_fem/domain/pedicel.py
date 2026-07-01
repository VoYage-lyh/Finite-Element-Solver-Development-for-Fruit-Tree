"""Pedicel (fruit-stem) mechanics — the lateral stiffness of a fruit swinging
on its stalk.

A fruit hangs from a branch on a slender pedicel.  For lateral (swing) motion
the pedicel acts as a short cantilever beam carrying the fruit as a tip mass,
with two restoring mechanisms that BOTH soften as the pedicel gets longer:

    k_pedicel = 3·E·I / L³   (cantilever bending, I = π·d⁴/64)
              + m·g / L      (gravitational pendulum restoring)

This is the physically-grounded replacement for the old ``k = F_detach /
d_detach`` shortcut, which conflated the elastic stiffness with the
secant-to-breaking stiffness and left the fruit-pedicel resonance ~10× too high
(≈100 Hz instead of the ≈5–14 Hz that puts fruit swing inside the harvest
band).  The breaking force ``F_detach`` is now an *independent* threshold, not a
by-product of the stiffness.

These three module constants are the SINGLE SOURCE for the default pedicel
geometry; ``FruitDistributionPolicy`` and the JSON loaders import them so the
default lives in exactly one place.  Defaults are an order-of-magnitude estimate
for Prunus cerasifera (L≈25 mm, d≈1.3 mm, green-stem E≈2 GPa → resonance ≈10 Hz)
pending a measured value.
"""

from __future__ import annotations

import math

# Gravitational acceleration used for the pendulum restoring term [m/s²].
GRAVITY = 9.81

# Default pedicel geometry for a ripe Prunus cerasifera fruit (single source).
DEFAULT_PEDICEL_LENGTH_M = 0.025
DEFAULT_PEDICEL_DIAMETER_M = 0.0013
DEFAULT_PEDICEL_YOUNGS_MODULUS_PA = 2.0e9


def pedicel_stiffness_n_per_m(
    fruit_mass_kg: float,
    length_m: float = DEFAULT_PEDICEL_LENGTH_M,
    diameter_m: float = DEFAULT_PEDICEL_DIAMETER_M,
    youngs_modulus_pa: float = DEFAULT_PEDICEL_YOUNGS_MODULUS_PA,
) -> float:
    """Lateral swing stiffness of one fruit on its pedicel [N/m].

    Cantilever bending (``3EI/L³``) plus the gravitational pendulum restoring
    (``m·g/L``).  Both terms shrink with pedicel length, so longer / thinner
    pedicels give a softer, lower-frequency fruit swing.
    """
    second_moment = math.pi * diameter_m**4 / 64.0
    k_bending = 3.0 * youngs_modulus_pa * second_moment / length_m**3
    k_pendulum = fruit_mass_kg * GRAVITY / length_m
    return k_bending + k_pendulum


def pedicel_resonance_hz(
    fruit_mass_kg: float,
    length_m: float = DEFAULT_PEDICEL_LENGTH_M,
    diameter_m: float = DEFAULT_PEDICEL_DIAMETER_M,
    youngs_modulus_pa: float = DEFAULT_PEDICEL_YOUNGS_MODULUS_PA,
) -> float:
    """Undamped fruit-swing natural frequency ``√(k/m)/2π`` [Hz] — diagnostic."""
    k = pedicel_stiffness_n_per_m(fruit_mass_kg, length_m, diameter_m, youngs_modulus_pa)
    return math.sqrt(k / fruit_mass_kg) / (2.0 * math.pi)
