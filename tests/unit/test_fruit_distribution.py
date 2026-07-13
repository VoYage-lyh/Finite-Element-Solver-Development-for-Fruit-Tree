from __future__ import annotations

import json

import pytest

from orchard_fem.io.fruit_distribution import (
    _BranchFruitSpec,
    generate_fruit_attachments_for_model,
    generate_fruit_parameters,
)
from orchard_fem.io.loaders.treeqsm import convert_treeqsm_payload
from orchard_fem.io.skeleton_import import (
    _infer_terminal_branch_ids,
    convert_skeleton_payload_to_orchard_payload,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_spec(
    branch_id: str,
    level: int,
    *,
    length_m: float = 1.0,
    is_terminal: bool = True,
) -> _BranchFruitSpec:
    return _BranchFruitSpec(
        branch_id=branch_id,
        level=level,
        length_m=length_m,
        is_terminal=is_terminal,
    )


def _multi_branch_skeleton_payload(
    *, with_policy: bool = False, total_fruit_count: int = 30
) -> dict:
    """Minimal skeleton with trunk → primary → sec_a, sec_b (both terminal level-2)."""
    payload: dict = {
        "branches": [
            {
                "id": "trunk",
                "parent_branch_id": None,
                "level": 0,
                "points": [[0, 0, 0], [0, 0, 1.5]],
                "outer_radius_root": 0.04,
                "outer_radius_tip": 0.03,
                "num_elements": 3,
            },
            {
                "id": "primary",
                "parent_branch_id": "trunk",
                "level": 1,
                "points": [[0, 0, 1.5], [0.4, 0, 2.0]],
                "outer_radius": 0.02,
                "num_elements": 2,
            },
            {
                "id": "sec_a",
                "parent_branch_id": "primary",
                "level": 2,
                "points": [[0.4, 0, 2.0], [0.7, 0.3, 2.3]],
                "outer_radius": 0.01,
                "num_elements": 2,
            },
            {
                "id": "sec_b",
                "parent_branch_id": "primary",
                "level": 2,
                "points": [[0.4, 0, 2.0], [0.7, -0.3, 2.3]],
                "outer_radius": 0.01,
                "num_elements": 2,
            },
        ],
        "clamps": [{"branch_id": "trunk", "support_stiffness": 1.0, "support_damping": 0.0}],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "trunk",
            "target_node": "root",
            "target_component": "ux",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 5.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "solver_backend": "native",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 10.0,
            "frequency_steps": 5,
            "include_gravity_prestress": False,
            "output_csv": "unused.csv",
        },
    }
    if with_policy:
        payload["fruit_distribution_policy"] = {
            "total_fruit_count": total_fruit_count,
            "seed": 2026,
        }
    return payload


def _load_model_from_payload(tmp_path, payload: dict):
    from orchard_fem.io import load_orchard_model

    converted = convert_skeleton_payload_to_orchard_payload(payload)
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(converted), encoding="utf-8")
    return load_orchard_model(str(model_path))


# ── Tests for generate_fruit_parameters ──────────────────────────────────────

def test_generate_fruit_parameters_produces_exact_count() -> None:
    specs = [
        _make_spec("sec_a", level=2, length_m=0.8),
        _make_spec("sec_b", level=2, length_m=0.6),
    ]
    total = 50
    fruits, summaries = generate_fruit_parameters(specs, total_fruit_count=total, seed=42)

    assert len(fruits) == total


def test_generate_fruit_parameters_allocates_only_to_eligible_branches() -> None:
    specs = [
        _make_spec("trunk", level=0, is_terminal=False),
        _make_spec("primary", level=1, is_terminal=False),
        _make_spec("sec_a", level=2, is_terminal=True),
        _make_spec("sec_b", level=2, is_terminal=True),
    ]
    fruits, _ = generate_fruit_parameters(specs, total_fruit_count=40, seed=7)

    ineligible_ids = {"trunk", "primary"}
    for fruit in fruits:
        assert fruit.branch_id not in ineligible_ids, (
            f"Fruit allocated to ineligible branch '{fruit.branch_id}'"
        )


# ── Tests for _infer_terminal_branch_ids ─────────────────────────────────────

def test_infer_terminal_branch_ids_returns_leaf_nodes() -> None:
    branches = [
        {"id": "trunk", "parent_branch_id": None},
        {"id": "primary", "parent_branch_id": "trunk"},
        {"id": "sec_a", "parent_branch_id": "primary"},
        {"id": "sec_b", "parent_branch_id": "primary"},
    ]
    terminals = _infer_terminal_branch_ids(branches)

    assert terminals == {"sec_a", "sec_b"}
    assert "trunk" not in terminals
    assert "primary" not in terminals


# ── Tests for generate_fruit_attachments_for_model ───────────────────────────

def test_generate_fruit_attachments_location_s_values(tmp_path) -> None:
    model = _load_model_from_payload(tmp_path, _multi_branch_skeleton_payload())

    from orchard_fem.domain import FruitDistributionPolicy

    policy = FruitDistributionPolicy(total_fruit_count=30, seed=2026)
    attachments = generate_fruit_attachments_for_model(model, policy)

    valid_s = {0.0, 0.5, 1.0}
    for att in attachments:
        assert att.location_s in valid_s, (
            f"Attachment '{att.fruit_id}' has unexpected location_s={att.location_s}"
        )


def test_generate_fruit_attachments_stiffness_matches_formula(tmp_path) -> None:
    model = _load_model_from_payload(tmp_path, _multi_branch_skeleton_payload())

    from orchard_fem.domain import FruitDistributionPolicy

    policy = FruitDistributionPolicy(
        total_fruit_count=20,
        seed=99,
        detachment_displacement_m=0.010,
    )

    parent_ids = {b.parent_branch_id for b in model.branches if b.parent_branch_id is not None}
    specs = [
        _BranchFruitSpec(
            branch_id=b.branch_id,
            level=b.level,
            length_m=b.path.length(),
            is_terminal=b.branch_id not in parent_ids,
        )
        for b in model.branches
    ]
    _, summaries = generate_fruit_parameters(
        specs,
        total_fruit_count=policy.total_fruit_count,
        seed=policy.seed,
        include_terminal_primary=policy.include_terminal_primary,
        count_weight_cv=policy.count_weight_cv,
        long_axis_cv=policy.long_axis_cv,
        short_axis_cv=policy.short_axis_cv,
        mass_residual_cv=policy.mass_residual_cv,
        detachment_force_cv=policy.detachment_force_cv,
        crack_probability=policy.crack_probability,
    )
    summary_by_node = {s.node_id: s for s in summaries}

    from orchard_fem.domain.pedicel import pedicel_stiffness_n_per_m

    attachments = generate_fruit_attachments_for_model(model, policy)
    for att in attachments:
        summary = summary_by_node[att.fruit_id]
        n = max(summary.fruit_count, 1)
        # Stiffness = n parallel pedicels (physical cantilever+pendulum model);
        # the breaking force is stored separately on detach_force = n·mean force.
        expected_stiffness = n * pedicel_stiffness_n_per_m(
            summary.total_fruit_mass_kg / n,
            policy.pedicel_length_m,
            policy.pedicel_diameter_m,
            policy.pedicel_youngs_modulus_pa,
        )
        assert att.stiffness == pytest.approx(expected_stiffness, rel=1.0e-9)
        expected_force = n * summary.mean_detachment_force_N
        assert att.detach_force == pytest.approx(expected_force, rel=1.0e-9)


# ── Tests for FruitDistributionPolicy round-trip ─────────────────────────────

def test_fruit_distribution_policy_roundtrip_expands_correct_count(tmp_path) -> None:
    total = 30
    payload = _multi_branch_skeleton_payload(with_policy=True, total_fruit_count=total)
    model = _load_model_from_payload(tmp_path, payload)

    # The loader should have expanded the policy to FruitAttachment objects
    assert len(model.fruits) > 0, "No fruits generated from policy"
    # Each node attachment aggregates multiple individual fruits.
    # Verify total count stored in fruit_policy matches.
    assert model.fruit_policy is not None
    assert model.fruit_policy.total_fruit_count == total


# ── Tests for convert_treeqsm_payload ────────────────────────────────────────

def test_convert_treeqsm_payload_raises_on_missing_branches() -> None:
    payload = {
        "excitation": {"kind": "harmonic_force", "target_branch_id": "trunk"},
        "analysis": {"mode": "frequency_response"},
    }
    with pytest.raises(ValueError, match="branches"):
        convert_treeqsm_payload(payload)


def test_convert_treeqsm_payload_raises_on_missing_excitation() -> None:
    payload = {
        "branches": [
            {"id": "trunk", "parent_branch_id": None, "level": 0,
             "points": [[0, 0, 0], [0, 0, 1]], "outer_radius_root": 0.04, "outer_radius_tip": 0.03}
        ],
        "analysis": {"mode": "frequency_response"},
    }
    with pytest.raises(ValueError, match="excitation"):
        convert_treeqsm_payload(payload)


def test_convert_treeqsm_payload_raises_on_branch_missing_id() -> None:
    payload = {
        "branches": [
            {"parent_branch_id": None, "level": 0,
             "points": [[0, 0, 0], [0, 0, 1]], "outer_radius_root": 0.04, "outer_radius_tip": 0.03}
        ],
        "excitation": {"kind": "harmonic_force", "target_branch_id": "trunk"},
        "analysis": {"mode": "frequency_response"},
    }
    with pytest.raises(ValueError, match="id"):
        convert_treeqsm_payload(payload)


# ── FEniCSx-dependent test ────────────────────────────────────────────────────

def test_gravity_static_displacement_stored_after_prestress(tmp_path) -> None:
    pytest.importorskip("petsc4py")
    pytest.importorskip("dolfinx")

    from orchard_fem.fenicsx.assembly import assemble_fenicsx_system
    from orchard_fem.io import load_orchard_model

    payload = {
        "branches": [
            {
                "id": "trunk",
                "parent_branch_id": None,
                "level": 0,
                "points": [[0, 0, 0], [0, 0, 1.5]],
                "outer_radius_root": 0.04,
                "outer_radius_tip": 0.03,
                "num_elements": 3,
            }
        ],
        "clamps": [{"branch_id": "trunk", "support_stiffness": 1.0, "support_damping": 0.0}],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "trunk",
            "target_node": "tip",
            "target_component": "ux",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 5.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "solver_backend": "fenicsx",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 10.0,
            "frequency_steps": 3,
            "include_gravity_prestress": True,
            "output_csv": "unused.csv",
        },
    }
    converted = convert_skeleton_payload_to_orchard_payload(payload)
    model_path = tmp_path / "gravity_model.json"
    model_path.write_text(json.dumps(converted), encoding="utf-8")
    model = load_orchard_model(str(model_path))

    assembly = assemble_fenicsx_system(model)
    op = assembly.experiment.operator_bundle

    assert op.gravity_static_displacement is not None, (
        "gravity_static_displacement should be stored after prestress assembly"
    )
