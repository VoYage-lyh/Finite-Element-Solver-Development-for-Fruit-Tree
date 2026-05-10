from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orchard_fem.discretization.types import NonlinearLinkDefinition
from orchard_fem.fenicsx.beam_forms import (
    EmbeddedBeamCellData,
    EmbeddedBeamCoefficientFunctions,
    EmbeddedBeamFormBundle,
)
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle


@dataclass(frozen=True)
class EmbeddedBeamOperatorBundle:
    compiled_stiffness_form: Any
    compiled_mass_form: Any
    stiffness_matrix: Any
    mass_matrix: Any
    attachment_damping_matrix: Any | None = None
    gravity_load_vector: Any | None = None
    gravity_static_displacement: Any | None = None
    geometric_stiffness_matrix: Any | None = None
    prestress_axial_forces: dict[str, list[float]] = field(default_factory=dict)
    fruit_dofs: dict[str, int] = field(default_factory=dict)
    nonlinear_links: list[NonlinearLinkDefinition] = field(default_factory=list)
    mpc: Any | None = None
    mpc_constrained_branch_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EmbeddedBeamExperimentBundle:
    mesh_spec: EmbeddedLineMeshSpec
    space_bundle: EmbeddedBeamFunctionSpaceBundle
    cell_data: EmbeddedBeamCellData
    coefficient_functions: EmbeddedBeamCoefficientFunctions
    form_bundle: EmbeddedBeamFormBundle
    operator_bundle: EmbeddedBeamOperatorBundle
    fruit_dofs: dict[str, int]
