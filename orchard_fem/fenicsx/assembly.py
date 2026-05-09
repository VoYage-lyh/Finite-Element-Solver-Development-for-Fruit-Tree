from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.beam_forms import (
    build_embedded_beam_cell_data,
    build_embedded_timoshenko_forms,
    create_embedded_beam_coefficient_functions,
)
from orchard_fem.fenicsx.boundary_conditions import build_model_clamp_boundary_conditions
from orchard_fem.fenicsx.embedded_mesh import (
    EmbeddedLineMeshSpec,
    build_embedded_line_mesh_spec,
)
from orchard_fem.fenicsx.fields import create_embedded_beam_function_space
from orchard_fem.fenicsx.operators import (
    EmbeddedBeamExperimentBundle,
    _augment_operators_with_auto_nonlinear_links,
    _augment_operators_with_fruits,
    _augment_operators_with_gravity_prestress,
    _augment_operators_with_joints,
    _augment_operators_with_nonlinear_clamps,
    assemble_embedded_beam_operators,
    EmbeddedBeamOperatorBundle,
)


@dataclass(frozen=True)
class FenicsxAssemblyStage:
    name: str
    enabled: bool = True
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class FenicsxSystemAssembly:
    model: OrchardModel
    experiment: EmbeddedBeamExperimentBundle
    stages: tuple[FenicsxAssemblyStage, ...] = field(default_factory=tuple)

    @property
    def stiffness_matrix(self):
        return self.experiment.operator_bundle.stiffness_matrix

    @property
    def mass_matrix(self):
        return self.experiment.operator_bundle.mass_matrix

    @property
    def nonlinear_links(self):
        return self.experiment.operator_bundle.nonlinear_links


def _stage(name: str, enabled: bool = True, *details: str) -> FenicsxAssemblyStage:
    return FenicsxAssemblyStage(name=name, enabled=enabled, details=tuple(details))


def _has_branch_connections(model: OrchardModel) -> bool:
    return any(branch.parent_branch_id is not None for branch in model.branches)


def _has_nonlinear_clamps(model: OrchardModel) -> bool:
    return any(abs(clamp.cubic_stiffness) > 1.0e-14 for clamp in model.clamps)


def _operator_size(operator_bundle: EmbeddedBeamOperatorBundle) -> int:
    return int(operator_bundle.stiffness_matrix.getSize()[0])


def assemble_fenicsx_system(
    model: OrchardModel,
    *,
    polynomial_degree: int = 1,
    spec: EmbeddedLineMeshSpec | None = None,
    shear_correction: float = 0.4,
    comm: object | None = None,
    partitioner: object | None = None,
    max_facet_to_cell_links: int = 2,
    use_model_clamps: bool = True,
    clamp_tolerance: float = 1.0e-8,
    bcs: Sequence[Any] | None = None,
    diag: float = 1.0,
) -> FenicsxSystemAssembly:
    require_dolfinx()
    stages: list[FenicsxAssemblyStage] = []

    resolved_spec = spec or build_embedded_line_mesh_spec(model)
    stages.append(_stage("mesh", True, f"cells={resolved_spec.cell_count}"))

    space_bundle = create_embedded_beam_function_space(
        resolved_spec,
        polynomial_degree=polynomial_degree,
        comm=comm,
        partitioner=partitioner,
        max_facet_to_cell_links=max_facet_to_cell_links,
    )
    stages.append(_stage("function_space", True, f"degree={polynomial_degree}"))

    cell_data = build_embedded_beam_cell_data(
        model,
        spec=resolved_spec,
        shear_correction=shear_correction,
    )
    stages.append(_stage("cell_data", True, f"cells={cell_data.cell_count}"))

    coefficient_functions = create_embedded_beam_coefficient_functions(
        space_bundle.mesh,
        cell_data,
    )
    stages.append(_stage("coefficients"))

    form_bundle = build_embedded_timoshenko_forms(
        space_bundle,
        coefficient_functions,
    )
    stages.append(_stage("ufl_forms", True, "stiffness", "mass", "residual", "jacobian"))

    if bcs is not None:
        effective_bcs = bcs
    elif use_model_clamps:
        effective_bcs = build_model_clamp_boundary_conditions(
            model,
            space_bundle,
            atol=clamp_tolerance,
            include_nonlinear_clamps=False,
        )
    else:
        effective_bcs = ()
    stages.append(_stage("boundary_conditions", bool(effective_bcs), f"count={len(effective_bcs)}"))

    operator_bundle = assemble_embedded_beam_operators(
        form_bundle,
        bcs=effective_bcs,
        diag=diag,
    )
    stages.append(_stage("base_operators", True, f"dofs={_operator_size(operator_bundle)}"))

    operator_bundle = _augment_operators_with_joints(
        model,
        space_bundle,
        resolved_spec,
        operator_bundle,
    )
    stages.append(
        _stage(
            "branch_joint_constraints",
            _has_branch_connections(model),
            f"nonlinear_links={len(operator_bundle.nonlinear_links)}",
        )
    )

    operator_bundle = _augment_operators_with_auto_nonlinear_links(
        model,
        space_bundle,
        resolved_spec,
        operator_bundle,
    )
    stages.append(
        _stage(
            "auto_nonlinear_links",
            bool(model.analysis.auto_nonlinear_levels),
            f"levels={list(model.analysis.auto_nonlinear_levels)}",
            f"nonlinear_links={len(operator_bundle.nonlinear_links)}",
        )
    )

    operator_bundle = _augment_operators_with_nonlinear_clamps(
        model,
        space_bundle,
        resolved_spec,
        operator_bundle,
    )
    stages.append(
        _stage(
            "nonlinear_clamps",
            _has_nonlinear_clamps(model),
            f"nonlinear_links={len(operator_bundle.nonlinear_links)}",
        )
    )

    operator_bundle = _augment_operators_with_fruits(
        model,
        space_bundle,
        resolved_spec,
        operator_bundle,
    )
    stages.append(
        _stage(
            "fruit_attachments",
            bool(model.fruits),
            f"fruits={len(model.fruits)}",
            f"dofs={_operator_size(operator_bundle)}",
        )
    )

    operator_bundle = _augment_operators_with_gravity_prestress(
        model,
        space_bundle,
        resolved_spec,
        cell_data,
        operator_bundle,
    )
    stages.append(
        _stage(
            "gravity_prestress",
            bool(model.analysis.include_gravity_prestress),
            f"branches={len(operator_bundle.prestress_axial_forces)}",
        )
    )

    experiment = EmbeddedBeamExperimentBundle(
        mesh_spec=resolved_spec,
        space_bundle=space_bundle,
        cell_data=cell_data,
        coefficient_functions=coefficient_functions,
        form_bundle=form_bundle,
        operator_bundle=operator_bundle,
        fruit_dofs=operator_bundle.fruit_dofs,
    )
    return FenicsxSystemAssembly(
        model=model,
        experiment=experiment,
        stages=tuple(stages),
    )
