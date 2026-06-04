"""Beam-element stress recovery from nodal displacements.

Upgrades the harvest fracture/clamp-damage tiers
(:mod:`orchard_fem.harvest.objective`) from amplitude-scaled stress *proxies* to
physically recovered bending stress.

For a Euler-Bernoulli/Timoshenko beam element the internal end forces follow
directly from the element stiffness and its nodal displacements,

    f_local = K_local · u_local,

which is exact (no finite differencing of the displacement field).  The peak
fibre stress over the section is then

    σ = |N| / A  +  |M_z|·c_y / I_z  +  |M_y|·c_z / I_y,

combining axial and biaxial bending at the extreme fibre (conservative: it sums
the worst-case fibre of each contribution).  For harmonic (FRF) analysis the
nodal displacements are complex amplitudes; the recovered forces are complex and
their magnitudes give the stress amplitude.

Local DOF ordering (matching
:func:`orchard_fem.discretization.beam.local_matrices.build_local_stiffness_matrix`)::

    node 1: [ux, uy, uz, rx, ry, rz] = indices 0..5
    node 2: [ux, uy, uz, rx, ry, rz] = indices 6..11

so axial force = f[0], torsion = f[3], bending moment about z (in-plane y
bending) = f[5]/f[11], bending moment about y = f[4]/f[10].

Section extreme-fibre distances ``c_y``, ``c_z`` are not carried by
:class:`~orchard_fem.discretization.beam.types.BeamElementProperties`; for a
solid circular section ``c = sqrt(4·I / A)`` exactly (see
:func:`extreme_fibre_distance`), which is used as the default when explicit
distances are not supplied.

Network-level wiring (peak stress over a whole tree from an FRF solve, to feed
``branch_peak_stress`` / ``clamp_stress`` in ``evaluate_harvest_objective``)
additionally requires the solver to retain the full complex solution vector and
the assembler to thread per-element section geometry through
``BranchElementState``; see :doc:`/docs/pinn_harvest_research_plan` §5.2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from orchard_fem.discretization.beam.local_matrices import build_local_stiffness_matrix
from orchard_fem.discretization.beam.types import BeamElementProperties

Number = complex | float


@dataclass(frozen=True)
class ElementEndForces:
    """Internal end forces/moments of a beam element in its local frame.

    All quantities are taken at the two element nodes.  For harmonic analysis
    they are complex amplitudes; use ``abs(...)`` for the amplitude.

    Parameters
    ----------
    axial:
        Axial force ``N`` at node 1 (``-N`` at node 2 for equilibrium) [N].
    torsion:
        Torsional moment [N·m].
    moment_y_node1 / moment_y_node2:
        Bending moment about the local y-axis at each node [N·m].
    moment_z_node1 / moment_z_node2:
        Bending moment about the local z-axis at each node [N·m].
    """

    axial: Number
    torsion: Number
    moment_y_node1: Number
    moment_y_node2: Number
    moment_z_node1: Number
    moment_z_node2: Number


def _matvec12(matrix: list[list[float]], vector: list[Number]) -> list[Number]:
    if len(vector) != 12:
        raise ValueError("Local displacement vector must have 12 components.")
    return [sum(matrix[r][c] * vector[c] for c in range(12)) for r in range(12)]


def recover_element_end_forces(
    properties: BeamElementProperties,
    local_displacements: list[Number],
) -> ElementEndForces:
    """Recover element end forces from local nodal displacements.

    Parameters
    ----------
    properties:
        Element properties (supplies the local stiffness matrix).
    local_displacements:
        Length-12 nodal displacement vector in the element local frame
        (real for static, complex amplitudes for harmonic analysis).

    Returns
    -------
    ElementEndForces
    """
    k_local = build_local_stiffness_matrix(properties)
    f = _matvec12(k_local, local_displacements)
    return ElementEndForces(
        axial=f[0],
        torsion=f[3],
        moment_y_node1=f[4],
        moment_y_node2=f[10],
        moment_z_node1=f[5],
        moment_z_node2=f[11],
    )


def extreme_fibre_distance(area: float, second_moment: float) -> float:
    """Extreme-fibre distance ``c`` from area and second moment of area.

    Exact for a solid circular section (``I = π r⁴/4``, ``A = π r²`` ⇒
    ``sqrt(4 I / A) = r``); an approximation for non-circular / multi-tissue
    sections until the true section outer radius is threaded through.

    Pass the second moment about *z* (``I_z = ∫ y² dA``) to obtain the y-fibre
    distance ``c_y``, and about *y* to obtain ``c_z``.
    """
    if area <= 0.0:
        raise ValueError("area must be positive.")
    if second_moment < 0.0:
        raise ValueError("second_moment must be non-negative.")
    return math.sqrt(4.0 * second_moment / area)


def element_peak_stress(
    properties: BeamElementProperties,
    local_displacements: list[Number],
    *,
    c_y: float | None = None,
    c_z: float | None = None,
) -> float:
    """Peak fibre stress amplitude over a beam element [Pa].

    Combines axial and biaxial-bending contributions at the worst-case extreme
    fibre::

        σ = |N|/A + |M_z|·c_y/I_z + |M_y|·c_z/I_y

    using the larger of the two end moments for each bending axis.

    Parameters
    ----------
    properties:
        Element properties (needs ``area``, ``iy``, ``iz``).
    local_displacements:
        Length-12 local nodal displacement vector (real or complex).
    c_y / c_z:
        Extreme-fibre distances [m].  Default to
        :func:`extreme_fibre_distance` from ``(area, iz)`` and ``(area, iy)``.

    Returns
    -------
    float
        Peak stress amplitude [Pa] (non-negative).
    """
    forces = recover_element_end_forces(properties, local_displacements)

    area = properties.area
    iy = properties.iy
    iz = properties.iz
    cy = c_y if c_y is not None else extreme_fibre_distance(area, iz)
    cz = c_z if c_z is not None else extreme_fibre_distance(area, iy)

    axial_stress = abs(forces.axial) / area if area > 0.0 else 0.0
    m_z = max(abs(forces.moment_z_node1), abs(forces.moment_z_node2))
    m_y = max(abs(forces.moment_y_node1), abs(forces.moment_y_node2))
    bending_z_stress = (m_z * cy / iz) if iz > 0.0 else 0.0
    bending_y_stress = (m_y * cz / iy) if iy > 0.0 else 0.0

    return axial_stress + bending_z_stress + bending_y_stress
