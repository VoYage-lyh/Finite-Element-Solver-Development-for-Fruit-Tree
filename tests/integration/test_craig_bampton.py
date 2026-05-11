"""Tests for Craig-Bampton model reduction.

Pure-Python tests (no FEniCSx) use synthetic dense matrices.
FEniCSx-dependent tests are skipped if petsc4py is not available.
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Pure-numpy Craig-Bampton math (synthetic 2-DOF partition)
# ---------------------------------------------------------------------------

def _build_synthetic_system(n_interior: int, n_interface: int):
    """Build trivial K, M matrices for testing partitioning logic."""
    import numpy as np

    n = n_interior + n_interface
    K = np.diag([float(i + 1) * 1000.0 for i in range(n)])
    M = np.eye(n) * 0.5
    return K, M


def test_constraint_modes_shape():
    """Constraint modes Φ_c should be [n_interior × n_interface]."""
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve

    n_i, n_b = 6, 2
    K, M = _build_synthetic_system(n_i, n_b)

    K_ii = K[:n_i, :n_i]
    K_ib = K[:n_i, n_i:]   # will be zero for diagonal K — add off-diag coupling

    # Add coupling
    K_ib = np.random.default_rng(0).uniform(0.1, 1.0, (n_i, n_b))
    cho = cho_factor(K_ii)
    Phi_c = -cho_solve(cho, K_ib)

    assert Phi_c.shape == (n_i, n_b)


def test_reduced_matrix_shape():
    """K_reduced and M_reduced must be [n_modes+n_b × n_modes+n_b]."""
    import numpy as np
    from scipy.linalg import eigh, cho_factor, cho_solve

    n_i, n_b = 8, 3
    n_modes_keep = 4
    K, M = _build_synthetic_system(n_i, n_b)
    n = n_i + n_b

    i_idx = list(range(n_i))
    b_idx = list(range(n_i, n))

    K_ii = K[np.ix_(i_idx, i_idx)]
    K_ib = np.eye(n_i, n_b) * 5.0   # add coupling
    M_ii = M[np.ix_(i_idx, i_idx)]

    cho = cho_factor(K_ii)
    Phi_c = -cho_solve(cho, K_ib)

    _, Phi_n = eigh(K_ii, M_ii, subset_by_index=[0, n_modes_keep - 1])

    T_cb = np.zeros((n, n_modes_keep + n_b))
    for local, glob in enumerate(i_idx):
        T_cb[glob, :n_modes_keep] = Phi_n[local, :]
        T_cb[glob, n_modes_keep:] = Phi_c[local, :]
    for local, glob in enumerate(b_idx):
        T_cb[glob, n_modes_keep + local] = 1.0

    K_r = T_cb.T @ K @ T_cb
    M_r = T_cb.T @ M @ T_cb

    expected_size = n_modes_keep + n_b
    assert K_r.shape == (expected_size, expected_size)
    assert M_r.shape == (expected_size, expected_size)


def test_reduced_matrices_symmetric():
    """Reduced matrices must be symmetric."""
    import numpy as np
    from scipy.linalg import eigh, cho_factor, cho_solve

    n_i, n_b = 6, 2
    n_modes_keep = 3
    K, M = _build_synthetic_system(n_i, n_b)
    n = n_i + n_b

    i_idx = list(range(n_i))
    b_idx = list(range(n_i, n))

    K_ii = K[np.ix_(i_idx, i_idx)]
    K_ib = np.ones((n_i, n_b)) * 0.5
    M_ii = M[np.ix_(i_idx, i_idx)]

    cho = cho_factor(K_ii)
    Phi_c = -cho_solve(cho, K_ib)
    _, Phi_n = eigh(K_ii, M_ii, subset_by_index=[0, n_modes_keep - 1])

    T_cb = np.zeros((n, n_modes_keep + n_b))
    for local, glob in enumerate(i_idx):
        T_cb[glob, :n_modes_keep] = Phi_n[local, :]
        T_cb[glob, n_modes_keep:] = Phi_c[local, :]
    for local, glob in enumerate(b_idx):
        T_cb[glob, n_modes_keep + local] = 1.0

    K_r = T_cb.T @ K @ T_cb
    M_r = T_cb.T @ M @ T_cb

    assert np.allclose(K_r, K_r.T, atol=1e-10)
    assert np.allclose(M_r, M_r.T, atol=1e-10)


def test_reduced_frf_returns_correct_length():
    """solve_frequency_response_reduced must return one point per frequency."""
    import numpy as np
    from orchard_fem.model_reduction.craig_bampton import (
        CraigBamptonReducedModel,
        solve_frequency_response_reduced,
    )

    n_modes = 3
    n_b = 2
    n_r = n_modes + n_b

    K_r = np.diag([100.0, 400.0, 900.0, 1600.0, 2500.0])
    M_r = np.eye(n_r) * 0.1
    T_cb = np.zeros((10, n_r))
    interface_dofs = [0, 1]

    rom = CraigBamptonReducedModel(
        T_cb=T_cb,
        K_reduced=K_r,
        M_reduced=M_r,
        interface_dofs=interface_dofs,
        n_internal_modes=n_modes,
        full_system_size=10,
    )

    freqs = [0.5, 1.0, 2.0, 5.0]
    result = solve_frequency_response_reduced(
        rom,
        excitation_dof=interface_dofs[0],   # DOF 0 is interface DOF
        frequencies_hz=freqs,
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-3,
    )
    assert len(result.points) == len(freqs)
    for point in result.points:
        assert point.excitation_response_magnitude >= 0.0


def test_reduced_frf_excitation_not_interface_raises():
    """Exciting a non-interface DOF must raise ValueError."""
    import numpy as np
    from orchard_fem.model_reduction.craig_bampton import (
        CraigBamptonReducedModel,
        solve_frequency_response_reduced,
    )

    n_r = 4
    K_r = np.eye(n_r) * 1000.0
    M_r = np.eye(n_r) * 0.1

    rom = CraigBamptonReducedModel(
        T_cb=np.zeros((10, n_r)),
        K_reduced=K_r,
        M_reduced=M_r,
        interface_dofs=[8, 9],   # DOFs 8 and 9 are interface
        n_internal_modes=2,
        full_system_size=10,
    )

    with pytest.raises(ValueError, match="interface DOF"):
        solve_frequency_response_reduced(
            rom,
            excitation_dof=5,   # not in interface_dofs
            frequencies_hz=[1.0],
        )


# ---------------------------------------------------------------------------
# CraigBamptonReductor — FEniCSx-dependent tests
# ---------------------------------------------------------------------------

def _make_simple_fenicsx_model():
    """Build a minimal 1-branch model for FEniCSx-dependent tests."""
    import json
    import tempfile
    from orchard_fem.io.loaders import load_orchard_model
    from orchard_fem.io.skeleton_import import convert_skeleton_payload_to_orchard_payload

    skeleton_payload = {
        "branches": [{
            "id": "trunk",
            "parent_branch_id": None,
            "level": 0,
            "points": [[0, 0, 0], [0, 0, 1.0]],
            "outer_radius_root": 0.03,
            "outer_radius_tip": 0.025,
            "num_elements": 4,
        }],
        "clamps": [{"branch_id": "trunk", "support_stiffness": 1.0, "support_damping": 0.0}],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "trunk",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "target_node": "tip",
            "target_component": "ux",
            "driving_frequency_hz": 5.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "solver_backend": "fenicsx",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 20.0,
            "frequency_steps": 5,
            "output_csv": "unused.csv",
        },
    }
    payload = convert_skeleton_payload_to_orchard_payload(skeleton_payload)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    return load_orchard_model(path)


def test_craig_bampton_reductor_fenicsx(tmp_path):
    """Full reduce() pipeline requires FEniCSx; skip otherwise."""
    pytest.importorskip("petsc4py")
    pytest.importorskip("dolfinx")

    from orchard_fem.model_reduction.craig_bampton import CraigBamptonReductor

    model = _make_simple_fenicsx_model()
    reductor = CraigBamptonReductor(
        interface_branch_ids=["trunk"],
        n_internal_modes=3,
    )
    rom = reductor.reduce(model)

    assert rom.full_system_size > 0
    assert rom.K_reduced.shape[0] == rom.K_reduced.shape[1]
    assert rom.K_reduced.shape[0] == rom.n_internal_modes + len(rom.interface_dofs)
