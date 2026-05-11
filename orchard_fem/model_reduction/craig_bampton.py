"""Craig-Bampton component mode synthesis (CMS).

Reduces the full finite-element system to a small ROM consisting of:
- *Interface DOFs* (branch root/junction nodes, preserved exactly)
- *Fixed-interface normal modes* (internal vibration shapes with boundaries fixed)

The reduction basis is:

    T_CB = [ Φ_n   Φ_c ]
           [  0    I_b ]

where ``Φ_n`` are the first *n_internal_modes* fixed-interface eigenvectors
(interior partition), ``Φ_c = -K_ii⁻¹ K_ib`` are the static constraint modes,
and ``I_b`` is the identity on the interface partition.

Reduced matrices:  ``K_r = T_CB^T K T_CB``,  ``M_r = T_CB^T M T_CB``.

Because orchard FEM models typically have < 2 000 DOFs, dense extraction via
PETSc ``getValues()`` is appropriate and avoids a separate sparse partitioning
library.

Example::

    from orchard_fem.model_reduction.craig_bampton import (
        CraigBamptonReductor,
        solve_frequency_response_reduced,
    )

    reductor = CraigBamptonReductor(
        interface_branch_ids=["trunk", "primary_a"],
        n_internal_modes=15,
    )
    rom = reductor.reduce(model)
    result = solve_frequency_response_reduced(
        rom,
        excitation_dof=rom.interface_dofs[0],
        frequencies_hz=[1.0, 2.0, 5.0, 10.0],
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-4,
    )
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING, Any

from orchard_fem.model_reduction.strategies import ReductionStrategy

if TYPE_CHECKING:
    from orchard_fem.domain import OrchardModel
    from orchard_fem.dynamics.frequency_response import FrequencyResponseResult


@dataclass
class CraigBamptonReducedModel:
    """Craig-Bampton ROM for an orchard FEM model.

    Parameters
    ----------
    T_cb:
        Transformation matrix ``[N_full × N_reduced]``.
    K_reduced:
        Reduced stiffness matrix ``[N_reduced × N_reduced]``.
    M_reduced:
        Reduced mass matrix ``[N_reduced × N_reduced]``.
    interface_dofs:
        Global DOF indices in the *full* system that are kept as interface DOFs.
    n_internal_modes:
        Number of fixed-interface normal modes retained.
    full_system_size:
        Total number of DOFs in the unreduced system (*N_full*).
    """

    T_cb: Any  # np.ndarray [N_full × N_reduced]
    K_reduced: Any  # np.ndarray [N_reduced × N_reduced]
    M_reduced: Any  # np.ndarray [N_reduced × N_reduced]
    interface_dofs: list[int]
    n_internal_modes: int
    full_system_size: int


class CraigBamptonReductor(ReductionStrategy):
    """Builds a Craig-Bampton ROM for an :class:`~orchard_fem.domain.OrchardModel`.

    Parameters
    ----------
    interface_branch_ids:
        IDs of branches whose root nodes are treated as interface DOFs (i.e.
        junction / clamp points).  All 6 DOFs of each selected node are kept.
    n_internal_modes:
        Number of fixed-interface normal modes to include in the basis.
    """

    def __init__(
        self,
        interface_branch_ids: list[str],
        n_internal_modes: int = 20,
    ) -> None:
        self._interface_branch_ids = list(interface_branch_ids)
        self._n_internal_modes = n_internal_modes

    def name(self) -> str:
        return "craig_bampton"

    def reduce(self, model: "OrchardModel") -> CraigBamptonReducedModel:
        """Build the Craig-Bampton ROM for *model*.

        All FEniCSx / PETSc / numpy imports are deferred to here so that the
        class can be imported in environments without those packages.

        Parameters
        ----------
        model:
            Assembled :class:`~orchard_fem.domain.OrchardModel`.

        Returns
        -------
        CraigBamptonReducedModel
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for Craig-Bampton reduction.") from exc
        try:
            from scipy.linalg import cho_factor, cho_solve, eigh
        except ImportError as exc:
            raise ImportError("scipy is required for Craig-Bampton reduction.") from exc

        from orchard_fem.fenicsx.assembly import assemble_fenicsx_system
        from orchard_fem.fenicsx.branch_dofs import resolve_branch_node_dofs

        assembly = assemble_fenicsx_system(model)
        op = assembly.experiment.operator_bundle
        K_petsc = op.stiffness_matrix
        M_petsc = op.mass_matrix
        N = K_petsc.getSize()[0]

        K_full = _petsc_to_dense(K_petsc, N)
        M_full = _petsc_to_dense(M_petsc, N)

        branch_dof_map = resolve_branch_node_dofs(
            model,
            assembly.experiment.space_bundle,
            assembly.experiment.mesh_spec,
        )

        interface_dofs = _collect_interface_dofs(
            self._interface_branch_ids, branch_dof_map, N
        )

        interior_dofs = [i for i in range(N) if i not in set(interface_dofs)]

        if not interior_dofs:
            raise ValueError(
                "Craig-Bampton: no interior DOFs remain after removing interface DOFs. "
                "Reduce the number of interface branches or use a coarser mesh."
            )

        b = interface_dofs
        i_idx = interior_dofs
        n_i = len(i_idx)
        n_b = len(b)

        K_ii = K_full[np.ix_(i_idx, i_idx)]
        K_ib = K_full[np.ix_(i_idx, b)]
        K_bb = K_full[np.ix_(b, b)]
        M_ii = M_full[np.ix_(i_idx, i_idx)]
        M_ib = M_full[np.ix_(i_idx, b)]
        M_bb = M_full[np.ix_(b, b)]

        # Constraint modes: Φ_c = -K_ii⁻¹ K_ib
        try:
            cho = cho_factor(K_ii)
            Phi_c = -cho_solve(cho, K_ib)   # [n_i × n_b]
        except Exception:
            # Fall back to lstsq if K_ii is nearly singular (e.g. insufficient BCs)
            Phi_c, _, _, _ = np.linalg.lstsq(K_ii, -K_ib, rcond=None)

        # Fixed-interface normal modes: solve K_ii Φ = λ M_ii Φ
        n_modes = min(self._n_internal_modes, n_i)
        eigenvalues, Phi_n = eigh(K_ii, M_ii, subset_by_index=[0, n_modes - 1])
        _ = eigenvalues  # frequencies can be computed as sqrt(λ)/(2π) if needed

        # Assemble T_CB
        T_cb = np.zeros((N, n_modes + n_b))
        for local_row, global_row in enumerate(i_idx):
            T_cb[global_row, :n_modes] = Phi_n[local_row, :]
            T_cb[global_row, n_modes:] = Phi_c[local_row, :]
        for local_row, global_row in enumerate(b):
            T_cb[global_row, n_modes + local_row] = 1.0

        # Reduced matrices
        K_r = T_cb.T @ K_full @ T_cb
        M_r = T_cb.T @ M_full @ T_cb

        return CraigBamptonReducedModel(
            T_cb=T_cb,
            K_reduced=K_r,
            M_reduced=M_r,
            interface_dofs=interface_dofs,
            n_internal_modes=n_modes,
            full_system_size=N,
        )


def _petsc_to_dense(mat: Any, N: int) -> Any:
    """Extract a PETSc Mat to a dense numpy array via getValues()."""
    import numpy as np

    rows = list(range(N))
    cols = list(range(N))
    values = mat.getValues(rows, cols)  # returns [N×N] numpy array
    return np.asarray(values, dtype=float)


def _collect_interface_dofs(
    branch_ids: list[str],
    branch_dof_map: dict,
    N: int,
) -> list[int]:
    """Collect all 6 root-node DOFs for each named branch."""
    seen: set[int] = set()
    result: list[int] = []
    for bid in branch_ids:
        if bid not in branch_dof_map:
            continue
        node_dofs_list = branch_dof_map[bid]
        if not node_dofs_list:
            continue
        root_dofs = node_dofs_list[0]  # first node = root
        for dof in root_dofs:
            if 0 <= dof < N and dof not in seen:
                seen.add(dof)
                result.append(dof)
    return result


def solve_frequency_response_reduced(
    rom: CraigBamptonReducedModel,
    excitation_dof: int,
    frequencies_hz: list[float],
    *,
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
) -> "FrequencyResponseResult":
    """Solve the FRF on the Craig-Bampton reduced system.

    Parameters
    ----------
    rom:
        Reduced model from :meth:`CraigBamptonReductor.reduce`.
    excitation_dof:
        Global DOF (in the *full* system) where the unit harmonic force is applied.
        Must be one of ``rom.interface_dofs``.
    frequencies_hz:
        Frequency grid [Hz].
    rayleigh_alpha:
        Mass-proportional Rayleigh damping coefficient.
    rayleigh_beta:
        Stiffness-proportional Rayleigh damping coefficient.

    Returns
    -------
    FrequencyResponseResult
        FRF evaluated at each frequency.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for reduced FRF.") from exc

    from orchard_fem.dynamics.frequency_response import (
        FrequencyResponsePoint,
        FrequencyResponseResult,
    )

    K_r = rom.K_reduced
    M_r = rom.M_reduced
    C_r = rayleigh_alpha * M_r + rayleigh_beta * K_r

    # Find the reduced-system column that corresponds to excitation_dof.
    n_modes = rom.n_internal_modes
    if excitation_dof in rom.interface_dofs:
        col_in_reduced = n_modes + rom.interface_dofs.index(excitation_dof)
    else:
        raise ValueError(
            f"excitation_dof={excitation_dof} is not an interface DOF. "
            "Only interface DOFs can be excited in the Craig-Bampton reduced system."
        )

    N_r = K_r.shape[0]
    points: list[FrequencyResponsePoint] = []

    for freq in frequencies_hz:
        omega = 2.0 * pi * freq
        # Build real-block system [N_r×N_r] → 2N_r × 2N_r
        A = K_r - omega ** 2 * M_r
        B = omega * C_r
        block = np.zeros((2 * N_r, 2 * N_r))
        block[:N_r, :N_r] = A
        block[:N_r, N_r:] = -B
        block[N_r:, :N_r] = B
        block[N_r:, N_r:] = A

        rhs = np.zeros(2 * N_r)
        rhs[col_in_reduced] = 1.0  # unit real load

        try:
            sol = np.linalg.solve(block, rhs)
        except np.linalg.LinAlgError:
            sol = np.zeros(2 * N_r)

        real_part = sol[col_in_reduced]
        imag_part = sol[N_r + col_in_reduced]
        magnitude = math.hypot(real_part, imag_part)

        points.append(
            FrequencyResponsePoint(
                frequency_hz=freq,
                excitation_response_magnitude=magnitude,
                observation_magnitudes=[magnitude],
            )
        )

    return FrequencyResponseResult(
        observation_names=["reduced_excitation_dof"],
        points=points,
    )
