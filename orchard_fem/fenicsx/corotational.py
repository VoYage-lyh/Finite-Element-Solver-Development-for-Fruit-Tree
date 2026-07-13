"""Corotational large-deformation 3-D Timoshenko beam formulation.

Implements geometric nonlinearity via a corotational frame: for each element
the current element frame is tracked, local deformations are measured relative to
that frame, and element forces / tangent stiffness are assembled from the
standard linear stiffness in the current configuration.

The tangent stiffness is:

    K_tang = T^T K_mat T + K_geo

where ``K_geo`` is the linear geometric stiffness from axial force.  The
rotational geometric stiffness term (K_rot) is omitted in this first
implementation; it is typically small for tree-branch bending-dominated motion.

DOF ordering per element (12 total, 6 per node):
    [ux_A, uy_A, uz_A, rx_A, ry_A, rz_A,
     ux_B, uy_B, uz_B, rx_B, ry_B, rz_B]

This matches FEniCSx Lagrange P1 vector-6 ordering for a 2-node 1-D element.

All numpy/PETSc imports are deferred to function bodies so this module can be
imported cleanly in environments without those packages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class CorotationalElementData:
    """Runtime state of one corotational beam element.

    Parameters
    ----------
    node_a_ref:
        Reference-configuration coordinates of node A [3].
    node_b_ref:
        Reference-configuration coordinates of node B [3].
    disp_a:
        Current displacements at node A [6]: (ux,uy,uz,rx,ry,rz).
    disp_b:
        Current displacements at node B [6].
    axial_force:
        Current axial force [N] (positive = tension).
    """

    node_a_ref: Any  # np.ndarray [3]
    node_b_ref: Any  # np.ndarray [3]
    disp_a: Any  # np.ndarray [6]
    disp_b: Any  # np.ndarray [6]
    axial_force: float


# ---------------------------------------------------------------------------
# Core corotational math (pure numpy)
# ---------------------------------------------------------------------------

def compute_corotational_current_frame(
    X_a: Any,
    X_b: Any,
    u_a: Any,
    u_b: Any,
) -> tuple[Any, Any]:
    """Compute the current element frame axes.

    Parameters
    ----------
    X_a, X_b:
        Reference-configuration node positions [3].
    u_a, u_b:
        Current displacement vectors [6] (first 3 = translations).

    Returns
    -------
    e1 : np.ndarray [3]
        Current axial direction (unit vector from A to B).
    R : np.ndarray [3×3]
        Rotation matrix whose columns are (e1, e2, e3).
    """
    import numpy as np

    x_a = X_a + u_a[:3]
    x_b = X_b + u_b[:3]
    chord = x_b - x_a
    length = float(np.linalg.norm(chord))
    if length < 1.0e-12:
        raise RuntimeError(
            "Corotational element has zero or near-zero current length — "
            "the mesh may be overly deformed."
        )
    e1 = chord / length

    # Reference axial direction (needed to build a consistent e2, e3)
    X_chord = X_b - X_a
    L0 = float(np.linalg.norm(X_chord))
    e1_ref = X_chord / L0 if L0 > 1.0e-12 else np.array([1.0, 0.0, 0.0])

    # Build e2 perpendicular to current e1 in the plane spanned by (e1, e1_ref)
    # Fall back to global Y or X if the reference axial is parallel to current axial
    proj = np.dot(e1_ref, e1)
    candidate = e1_ref - proj * e1
    cand_norm = float(np.linalg.norm(candidate))
    if cand_norm > 1.0e-8:
        e2 = candidate / cand_norm
    else:
        # e1 ≈ e1_ref; pick a perpendicular
        perp = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(e1, perp)) > 0.9:
            perp = np.array([0.0, 0.0, 1.0])
        e2 = perp - np.dot(perp, e1) * e1
        e2 = e2 / np.linalg.norm(e2)

    e3 = np.cross(e1, e2)
    R = np.column_stack([e1, e2, e3])
    return e1, R


def compute_corotational_local_deformations(
    X_a: Any,
    X_b: Any,
    u_a: Any,
    u_b: Any,
    R: Any,
) -> Any:
    """Compute the 12-DOF local displacement vector in the current element frame.

    Parameters
    ----------
    X_a, X_b:
        Reference-configuration node positions [3].
    u_a, u_b:
        Current global displacements [6].
    R : np.ndarray [3×3]
        Current element rotation matrix (columns are e1, e2, e3).

    Returns
    -------
    u_local : np.ndarray [12]
        Local DOFs: translations and rotations in the current element frame.
        Node A translations are zero by construction (corotational origin).
    """
    import numpy as np

    x_a = X_a + u_a[:3]
    x_b = X_b + u_b[:3]
    L0 = float(np.linalg.norm(X_b - X_a))
    current_length = float(np.linalg.norm(x_b - x_a))

    # Local translations at A: zero (corotational frame is attached to A)
    # Local axial deformation at B: current length - reference length
    t_a_loc = np.zeros(3)
    t_b_loc = np.array([current_length - L0, 0.0, 0.0])

    # Rotate global rotations into the current element frame
    r_a_loc = R.T @ u_a[3:]
    r_b_loc = R.T @ u_b[3:]

    u_local = np.concatenate([t_a_loc, r_a_loc, t_b_loc, r_b_loc])
    return u_local


def _build_12x12_transformation(R: Any) -> Any:
    """Build the 12×12 block-diagonal element transformation matrix from R [3×3]."""
    import numpy as np

    T = np.zeros((12, 12))
    for block in range(4):
        off = block * 3
        T[off:off + 3, off:off + 3] = R
    return T


def _local_stiffness_from_cell_data(
    cell_data: Any,
    cell_index: int,
    length: float,
) -> Any:
    """Build the 12×12 local linear stiffness from ``EmbeddedBeamCellData``."""
    import numpy as np
    from orchard_fem.discretization.beam.local_matrices import build_local_stiffness_matrix
    from orchard_fem.discretization.beam.types import BeamElementProperties

    cd = cell_data
    # Provide combined rigidities directly; scalar field values are not needed.
    props = BeamElementProperties(
        youngs_modulus=1.0,      # unused when *_rigidity overrides are given
        shear_modulus=1.0,       # unused when *_rigidity overrides are given
        area=1.0,
        iy=1.0,
        iz=1.0,
        torsion_constant=1.0,
        density=1.0,
        length=length,
        axial_rigidity=float(cd.axial_rigidity[cell_index]),
        torsional_rigidity=float(cd.torsional_rigidity[cell_index]),
        bending_rigidity_y=float(cd.bending_rigidity_y[cell_index]),
        bending_rigidity_z=float(cd.bending_rigidity_z[cell_index]),
    )
    K_loc_list = build_local_stiffness_matrix(props)
    return np.array(K_loc_list, dtype=float)


def _local_geometric_stiffness(axial_force: float, length: float) -> Any:
    import numpy as np
    from orchard_fem.discretization.beam.local_matrices import build_local_geometric_stiffness_matrix

    K_geo_list = build_local_geometric_stiffness_matrix(axial_force, length)
    return np.array(K_geo_list, dtype=float)


def compute_corotational_element_forces_and_stiffness(
    X_a: Any,
    X_b: Any,
    u_a: Any,
    u_b: Any,
    cell_data: Any,
    cell_index: int,
) -> tuple[Any, Any]:
    """Compute corotational element internal forces and tangent stiffness.

    Parameters
    ----------
    X_a, X_b:
        Reference-configuration node positions [3].
    u_a, u_b:
        Current global displacements [6].
    cell_data:
        :class:`~orchard_fem.fenicsx.beam_forms.EmbeddedBeamCellData`.
    cell_index:
        Index of this cell in ``cell_data`` arrays.

    Returns
    -------
    f_int : np.ndarray [12]
        Element internal force vector in **global** coordinates.
    K_tang : np.ndarray [12×12]
        Element tangent stiffness in **global** coordinates.
    """
    import numpy as np

    _, R = compute_corotational_current_frame(X_a, X_b, u_a, u_b)
    u_loc = compute_corotational_local_deformations(X_a, X_b, u_a, u_b, R)

    L0 = float(np.linalg.norm(X_b - X_a))
    x_a = X_a + u_a[:3]
    x_b = X_b + u_b[:3]
    l_current = float(np.linalg.norm(x_b - x_a))

    K_loc = _local_stiffness_from_cell_data(cell_data, cell_index, l_current)

    # Internal forces in local frame
    f_loc = K_loc @ u_loc

    # Axial force for geometric stiffness
    EA = float(cell_data.axial_rigidity[cell_index])
    axial_strain = (l_current - L0) / L0 if L0 > 1.0e-12 else 0.0
    axial_force = EA * axial_strain

    K_geo = _local_geometric_stiffness(axial_force, l_current)

    # Build full 12×12 rotation block
    T = _build_12x12_transformation(R)

    # Material tangent in global: T^T K_mat T
    K_mat_global = T.T @ K_loc @ T
    K_geo_global = T.T @ K_geo @ T
    K_tang = K_mat_global + K_geo_global

    # Internal forces in global frame
    f_int = T.T @ f_loc

    return f_int, K_tang


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

def assemble_corotational_internal_force(
    assembly: Any,
    displacement_vector: Any,
) -> Any:
    """Assemble the global internal force vector using the corotational formulation.

    Parameters
    ----------
    assembly:
        :class:`~orchard_fem.fenicsx.assembly.FenicsCxAssemblyResult` (the return
        value of ``assemble_fenicsx_system``).
    displacement_vector:
        PETSc Vec containing the current global displacements.

    Returns
    -------
    PETSc Vec
        Global internal force vector.
    """
    from petsc4py import PETSc

    experiment = assembly.experiment
    cell_data = experiment.cell_data
    space_bundle = experiment.space_bundle
    mesh = space_bundle.mesh
    V = space_bundle.function_space

    u_array = displacement_vector.getArray(readonly=True)

    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)
    cell_node_map = topology.connectivity(topology.dim, 0)
    coords = mesh.geometry.x

    f_global = displacement_vector.duplicate()
    f_global.set(0.0)

    n_cells = cell_data.cell_count
    for cell_index in range(n_cells):
        cell_dofs = V.dofmap.cell_dofs(cell_index)  # [12] global DOF indices
        nodes = cell_node_map.links(cell_index)     # [2] node indices

        X_a = coords[nodes[0]]
        X_b = coords[nodes[1]]

        u_a = u_array[cell_dofs[:6]]
        u_b = u_array[cell_dofs[6:]]

        f_elem, _ = compute_corotational_element_forces_and_stiffness(
            X_a, X_b, u_a, u_b, cell_data, cell_index
        )

        for local_dof, global_dof in enumerate(cell_dofs):
            f_global.setValue(
                int(global_dof),
                f_elem[local_dof],
                addv=PETSc.InsertMode.ADD_VALUES,
            )

    f_global.assemblyBegin()
    f_global.assemblyEnd()
    return f_global


def assemble_corotational_tangent_stiffness(
    assembly: Any,
    displacement_vector: Any,
) -> Any:
    """Assemble the global tangent stiffness matrix using the corotational formulation.

    Parameters
    ----------
    assembly:
        :class:`~orchard_fem.fenicsx.assembly.FenicsCxAssemblyResult`.
    displacement_vector:
        PETSc Vec containing the current global displacements.

    Returns
    -------
    PETSc Mat
        Global tangent stiffness matrix.
    """
    from petsc4py import PETSc

    from orchard_fem.fenicsx.petsc_ops import copy_matrix_like

    experiment = assembly.experiment
    cell_data = experiment.cell_data
    space_bundle = experiment.space_bundle
    mesh = space_bundle.mesh
    V = space_bundle.function_space

    u_array = displacement_vector.getArray(readonly=True)

    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)
    cell_node_map = topology.connectivity(topology.dim, 0)
    coords = mesh.geometry.x

    K_global = copy_matrix_like(experiment.operator_bundle.stiffness_matrix)
    K_global.zeroEntries()

    n_cells = cell_data.cell_count
    for cell_index in range(n_cells):
        cell_dofs = V.dofmap.cell_dofs(cell_index)
        nodes = cell_node_map.links(cell_index)

        X_a = coords[nodes[0]]
        X_b = coords[nodes[1]]

        u_a = u_array[cell_dofs[:6]]
        u_b = u_array[cell_dofs[6:]]

        _, K_elem = compute_corotational_element_forces_and_stiffness(
            X_a, X_b, u_a, u_b, cell_data, cell_index
        )

        rows = [int(d) for d in cell_dofs]
        cols = [int(d) for d in cell_dofs]
        K_global.setValues(rows, cols, K_elem.ravel(), addv=PETSc.InsertMode.ADD_VALUES)

    K_global.assemblyBegin()
    K_global.assemblyEnd()
    return K_global
