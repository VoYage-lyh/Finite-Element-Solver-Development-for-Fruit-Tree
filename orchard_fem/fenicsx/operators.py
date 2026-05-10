from __future__ import annotations

from typing import Any, Sequence

from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.beam_forms import EmbeddedBeamFormBundle
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.operator_bundle import (
    EmbeddedBeamExperimentBundle,
    EmbeddedBeamOperatorBundle,
)


def assemble_embedded_beam_operators(
    form_bundle: EmbeddedBeamFormBundle,
    *,
    bcs: Sequence[Any] = (),
    diag: float = 1.0,
    mpc: Any | None = None,
    mpc_constrained_branch_ids: frozenset[str] = frozenset(),
) -> EmbeddedBeamOperatorBundle:
    require_dolfinx()

    from dolfinx import fem as dolfinx_fem
    from dolfinx.fem import petsc as dolfinx_fem_petsc

    compiled_stiffness_form = dolfinx_fem.form(form_bundle.stiffness_form)
    compiled_mass_form = dolfinx_fem.form(form_bundle.mass_form)

    if mpc is None:
        stiffness_matrix = dolfinx_fem_petsc.assemble_matrix(
            compiled_stiffness_form,
            bcs=bcs,
            diag=diag,
        )
    else:
        import dolfinx_mpc

        stiffness_matrix = dolfinx_mpc.assemble_matrix(
            compiled_stiffness_form,
            mpc,
            bcs=bcs,
            diagval=diag,
        )
    stiffness_matrix.assemble()

    if mpc is None:
        mass_matrix = dolfinx_fem_petsc.assemble_matrix(
            compiled_mass_form,
            bcs=bcs,
            diag=0.0,
        )
    else:
        import dolfinx_mpc

        mass_matrix = dolfinx_mpc.assemble_matrix(
            compiled_mass_form,
            mpc,
            bcs=bcs,
            diagval=0.0,
        )
    mass_matrix.assemble()

    return EmbeddedBeamOperatorBundle(
        compiled_stiffness_form=compiled_stiffness_form,
        compiled_mass_form=compiled_mass_form,
        stiffness_matrix=stiffness_matrix,
        mass_matrix=mass_matrix,
        mpc=mpc,
        mpc_constrained_branch_ids=mpc_constrained_branch_ids,
    )


def build_embedded_timoshenko_experiment(
    model: OrchardModel,
    *,
    polynomial_degree: int = 1,
    spec: EmbeddedLineMeshSpec | None = None,
    shear_correction: float = 0.4,
    comm: object | None = None,
    partitioner: object | None = None,
    max_facet_to_cell_links: int = 2,
    bcs: Sequence[Any] | None = None,
    use_model_clamps: bool = False,
    clamp_tolerance: float = 1.0e-8,
    diag: float = 1.0,
) -> EmbeddedBeamExperimentBundle:
    from orchard_fem.fenicsx.assembly import assemble_fenicsx_system

    return assemble_fenicsx_system(
        model,
        polynomial_degree=polynomial_degree,
        spec=spec,
        shear_correction=shear_correction,
        comm=comm,
        partitioner=partitioner,
        max_facet_to_cell_links=max_facet_to_cell_links,
        use_model_clamps=use_model_clamps,
        clamp_tolerance=clamp_tolerance,
        bcs=bcs,
        diag=diag,
    ).experiment
