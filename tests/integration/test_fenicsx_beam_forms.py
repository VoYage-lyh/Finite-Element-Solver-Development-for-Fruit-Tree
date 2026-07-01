from __future__ import annotations

import os

import numpy as np
import pytest

from orchard_fem.fenicsx import (
    assemble_embedded_beam_operators,
    assemble_fenicsx_system,
    build_embedded_beam_cell_data,
    build_embedded_line_mesh_spec,
    build_embedded_timoshenko_forms,
    build_embedded_timoshenko_experiment,
    create_embedded_beam_coefficient_functions,
    create_embedded_beam_function_space,
    fenicsx_stack_available,
)
from orchard_fem.io import load_orchard_model


def test_embedded_beam_cell_data_matches_mesh_spec_and_stays_positive() -> None:
    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    spec = build_embedded_line_mesh_spec(model)
    cell_data = build_embedded_beam_cell_data(model, spec=spec)

    assert cell_data.cell_count == spec.cell_count
    assert np.all(cell_data.axial_rigidity > 0.0)
    assert np.all(cell_data.torsional_rigidity > 0.0)
    assert np.all(cell_data.bending_rigidity_y > 0.0)
    assert np.all(cell_data.bending_rigidity_z > 0.0)
    assert np.all(cell_data.mass_per_length > 0.0)


def test_fenicsx_system_assembly_facade_is_exported() -> None:
    assert callable(assemble_fenicsx_system)


def test_fenicsx_system_assembly_reports_pipeline_stages() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx assembly tests")
    if not fenicsx_stack_available():
        pytest.skip("FEniCSx stack not available in this environment")

    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    assembly = assemble_fenicsx_system(model, polynomial_degree=1)
    stage_names = [stage.name for stage in assembly.stages]

    assert stage_names == [
        "mesh",
        "function_space",
        "cell_data",
        "coefficients",
        "ufl_forms",
        "boundary_conditions",
        "branch_connection_mpc",
        "base_operators",
        "branch_joint_constraints",
        "auto_nonlinear_links",
        "nonlinear_clamps",
        "fruit_attachments",
        "gravity_prestress",
    ]
    assert assembly.stiffness_matrix.getSize() == assembly.mass_matrix.getSize()
    assert assembly.experiment.operator_bundle is not None


def test_embedded_beam_cell_frames_are_unit_and_orthogonal() -> None:
    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    cell_data = build_embedded_beam_cell_data(model)

    tangent = np.stack(
        [cell_data.tangent_x, cell_data.tangent_y, cell_data.tangent_z],
        axis=1,
    )
    local_y = np.stack(
        [cell_data.normal_y_x, cell_data.normal_y_y, cell_data.normal_y_z],
        axis=1,
    )
    local_z = np.stack(
        [cell_data.normal_z_x, cell_data.normal_z_y, cell_data.normal_z_z],
        axis=1,
    )

    assert np.allclose(np.linalg.norm(tangent, axis=1), 1.0, atol=1.0e-10)
    assert np.allclose(np.linalg.norm(local_y, axis=1), 1.0, atol=1.0e-10)
    assert np.allclose(np.linalg.norm(local_z, axis=1), 1.0, atol=1.0e-10)
    assert np.allclose(np.sum(tangent * local_y, axis=1), 0.0, atol=1.0e-10)
    assert np.allclose(np.sum(tangent * local_z, axis=1), 0.0, atol=1.0e-10)
    assert np.allclose(np.sum(local_y * local_z, axis=1), 0.0, atol=1.0e-10)


def test_embedded_beam_form_bundle_smoke() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx beam-form tests")
    if not fenicsx_stack_available():
        pytest.skip("FEniCSx stack not available in this environment")

    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    space_bundle = create_embedded_beam_function_space(model, polynomial_degree=1)
    cell_data = build_embedded_beam_cell_data(model)
    coefficients = create_embedded_beam_coefficient_functions(space_bundle.mesh, cell_data)
    form_bundle = build_embedded_timoshenko_forms(space_bundle, coefficients)

    from dolfinx import fem as dolfinx_fem

    assert form_bundle.stiffness_form is not None
    assert form_bundle.mass_form is not None
    assert form_bundle.state_function is not None
    assert form_bundle.residual_form is not None
    assert form_bundle.jacobian_form is not None
    dolfinx_fem.form(form_bundle.stiffness_form)
    dolfinx_fem.form(form_bundle.mass_form)
    dolfinx_fem.form(form_bundle.residual_form)
    dolfinx_fem.form(form_bundle.jacobian_form)


def test_embedded_beam_operator_bundle_smoke() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx beam-form tests")
    if not fenicsx_stack_available():
        pytest.skip("FEniCSx stack not available in this environment")

    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    experiment = build_embedded_timoshenko_experiment(model, polynomial_degree=1)
    operator_bundle = experiment.operator_bundle

    stiffness_size = operator_bundle.stiffness_matrix.getSize()
    mass_size = operator_bundle.mass_matrix.getSize()
    assert stiffness_size[0] == stiffness_size[1]
    assert mass_size[0] == mass_size[1]
    assert stiffness_size == mass_size
    assert stiffness_size[0] > 0


def test_embedded_beam_operator_bundle_matches_manual_pipeline() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx beam-form tests")
    if not fenicsx_stack_available():
        pytest.skip("FEniCSx stack not available in this environment")

    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    space_bundle = create_embedded_beam_function_space(model, polynomial_degree=1)
    cell_data = build_embedded_beam_cell_data(model)
    coefficients = create_embedded_beam_coefficient_functions(space_bundle.mesh, cell_data)
    form_bundle = build_embedded_timoshenko_forms(space_bundle, coefficients)
    operator_bundle = assemble_embedded_beam_operators(form_bundle)

    assert operator_bundle.compiled_stiffness_form is not None
    assert operator_bundle.compiled_mass_form is not None
    assert operator_bundle.stiffness_matrix.getSize() == operator_bundle.mass_matrix.getSize()


def test_embedded_beam_ufl_jacobian_matches_linear_stiffness_form() -> None:
    if os.environ.get("ORCHARD_RUN_DOLFINX_TESTS") != "1":
        pytest.skip("Set ORCHARD_RUN_DOLFINX_TESTS=1 to run DOLFINx beam-form tests")
    if not fenicsx_stack_available():
        pytest.skip("FEniCSx stack not available in this environment")

    model = load_orchard_model("tests/fixtures/demo_orchard.json")
    space_bundle = create_embedded_beam_function_space(model, polynomial_degree=1)
    cell_data = build_embedded_beam_cell_data(model)
    coefficients = create_embedded_beam_coefficient_functions(space_bundle.mesh, cell_data)
    form_bundle = build_embedded_timoshenko_forms(space_bundle, coefficients)

    from dolfinx import fem as dolfinx_fem
    from dolfinx.fem import petsc as dolfinx_fem_petsc
    from petsc4py import PETSc

    stiffness_matrix = dolfinx_fem_petsc.assemble_matrix(
        dolfinx_fem.form(form_bundle.stiffness_form)
    )
    stiffness_matrix.assemble()
    jacobian_matrix = dolfinx_fem_petsc.assemble_matrix(
        dolfinx_fem.form(form_bundle.jacobian_form)
    )
    jacobian_matrix.assemble()

    jacobian_matrix.axpy(
        -1.0,
        stiffness_matrix,
        structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
    )
    jacobian_matrix.assemble()
    assert jacobian_matrix.norm() <= 1.0e-7 * max(stiffness_matrix.norm(), 1.0)
