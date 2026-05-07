from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orchard_fem.discretization.beam.transforms import build_local_frame
from orchard_fem.domain import OrchardModel
from orchard_fem.fenicsx.availability import require_dolfinx
from orchard_fem.fenicsx.embedded_mesh import EmbeddedLineMeshSpec, build_embedded_line_mesh_spec
from orchard_fem.fenicsx.fields import EmbeddedBeamFunctionSpaceBundle
from orchard_fem.materials import build_material_lookup, evaluate_branch_section_state


@dataclass(frozen=True)
class EmbeddedBeamCellData:
    axial_rigidity: np.ndarray
    shear_rigidity_y: np.ndarray
    shear_rigidity_z: np.ndarray
    torsional_rigidity: np.ndarray
    bending_rigidity_y: np.ndarray
    bending_rigidity_z: np.ndarray
    mass_per_length: np.ndarray
    rotary_inertia_t: np.ndarray
    rotary_inertia_y: np.ndarray
    rotary_inertia_z: np.ndarray
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    tangent_z: np.ndarray
    normal_y_x: np.ndarray
    normal_y_y: np.ndarray
    normal_y_z: np.ndarray
    normal_z_x: np.ndarray
    normal_z_y: np.ndarray
    normal_z_z: np.ndarray

    @property
    def cell_count(self) -> int:
        return int(self.axial_rigidity.shape[0])


@dataclass(frozen=True)
class EmbeddedBeamCoefficientFunctions:
    axial_rigidity: Any
    shear_rigidity_y: Any
    shear_rigidity_z: Any
    torsional_rigidity: Any
    bending_rigidity_y: Any
    bending_rigidity_z: Any
    mass_per_length: Any
    rotary_inertia_t: Any
    rotary_inertia_y: Any
    rotary_inertia_z: Any
    tangent_x: Any
    tangent_y: Any
    tangent_z: Any
    normal_y_x: Any
    normal_y_y: Any
    normal_y_z: Any
    normal_z_x: Any
    normal_z_y: Any
    normal_z_z: Any


@dataclass(frozen=True)
class EmbeddedBeamFormBundle:
    stiffness_form: Any
    mass_form: Any


def _zeros(count: int) -> np.ndarray:
    return np.zeros(count, dtype=np.float64)


def build_embedded_beam_cell_data(
    model: OrchardModel,
    *,
    spec: EmbeddedLineMeshSpec | None = None,
    shear_correction: float = 0.4,
) -> EmbeddedBeamCellData:
    if shear_correction <= 0.0:
        raise ValueError("shear_correction must be positive.")

    spec = spec or build_embedded_line_mesh_spec(model)
    material_lookup = build_material_lookup(model.materials)
    count = spec.cell_count

    axial_rigidity = _zeros(count)
    shear_rigidity_y = _zeros(count)
    shear_rigidity_z = _zeros(count)
    torsional_rigidity = _zeros(count)
    bending_rigidity_y = _zeros(count)
    bending_rigidity_z = _zeros(count)
    mass_per_length = _zeros(count)
    rotary_inertia_t = _zeros(count)
    rotary_inertia_y = _zeros(count)
    rotary_inertia_z = _zeros(count)
    tangent_x = _zeros(count)
    tangent_y = _zeros(count)
    tangent_z = _zeros(count)
    normal_y_x = _zeros(count)
    normal_y_y = _zeros(count)
    normal_y_z = _zeros(count)
    normal_z_x = _zeros(count)
    normal_z_y = _zeros(count)
    normal_z_z = _zeros(count)

    cell_index = 0
    for branch in model.branches:
        num_elements = max(branch.discretization.num_elements, 1)
        for element_index in range(num_elements):
            first_station = element_index / num_elements
            second_station = (element_index + 1) / num_elements
            first_point = branch.path.point_at(first_station)
            second_point = branch.path.point_at(second_station)
            first_state = evaluate_branch_section_state(branch, material_lookup, first_station)
            second_state = evaluate_branch_section_state(branch, material_lookup, second_station)

            area = 0.5 * (first_state.area + second_state.area)
            iy = 0.5 * (first_state.ix + second_state.ix)
            iz = 0.5 * (first_state.iy + second_state.iy)
            polar_moment = max(
                0.5 * (first_state.polar_moment + second_state.polar_moment),
                iy + iz,
            )
            mass_line = 0.5 * (
                first_state.mass_per_length + second_state.mass_per_length
            )
            youngs_modulus = 0.5 * (
                first_state.effective_youngs_modulus
                + second_state.effective_youngs_modulus
            )
            shear_modulus = 0.5 * (
                first_state.effective_shear_modulus
                + second_state.effective_shear_modulus
            )
            density = mass_line / area if area > 0.0 else 0.0
            tangent, local_y, local_z = build_local_frame(first_point, second_point)

            axial_rigidity[cell_index] = youngs_modulus * area
            shear_rigidity_y[cell_index] = shear_correction * shear_modulus * area
            shear_rigidity_z[cell_index] = shear_correction * shear_modulus * area
            torsional_rigidity[cell_index] = shear_modulus * polar_moment
            bending_rigidity_y[cell_index] = youngs_modulus * iy
            bending_rigidity_z[cell_index] = youngs_modulus * iz
            mass_per_length[cell_index] = mass_line
            rotary_inertia_t[cell_index] = density * polar_moment
            rotary_inertia_y[cell_index] = density * iy
            rotary_inertia_z[cell_index] = density * iz
            tangent_x[cell_index] = tangent.x
            tangent_y[cell_index] = tangent.y
            tangent_z[cell_index] = tangent.z
            normal_y_x[cell_index] = local_y.x
            normal_y_y[cell_index] = local_y.y
            normal_y_z[cell_index] = local_y.z
            normal_z_x[cell_index] = local_z.x
            normal_z_y[cell_index] = local_z.y
            normal_z_z[cell_index] = local_z.z

            cell_index += 1

    if cell_index != count:
        raise ValueError(
            "Embedded beam cell data count mismatch between orchard topology and mesh spec."
        )

    return EmbeddedBeamCellData(
        axial_rigidity=axial_rigidity,
        shear_rigidity_y=shear_rigidity_y,
        shear_rigidity_z=shear_rigidity_z,
        torsional_rigidity=torsional_rigidity,
        bending_rigidity_y=bending_rigidity_y,
        bending_rigidity_z=bending_rigidity_z,
        mass_per_length=mass_per_length,
        rotary_inertia_t=rotary_inertia_t,
        rotary_inertia_y=rotary_inertia_y,
        rotary_inertia_z=rotary_inertia_z,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        tangent_z=tangent_z,
        normal_y_x=normal_y_x,
        normal_y_y=normal_y_y,
        normal_y_z=normal_y_z,
        normal_z_x=normal_z_x,
        normal_z_y=normal_z_y,
        normal_z_z=normal_z_z,
    )


def _create_cellwise_scalar_function(mesh: Any, values: np.ndarray, name: str) -> Any:
    from dolfinx import fem as dolfinx_fem

    function_space = dolfinx_fem.functionspace(mesh, ("DG", 0))
    function = dolfinx_fem.Function(function_space, name=name)
    function.x.array[:] = np.asarray(values, dtype=function.x.array.dtype)
    return function


def create_embedded_beam_coefficient_functions(
    mesh: Any,
    cell_data: EmbeddedBeamCellData,
) -> EmbeddedBeamCoefficientFunctions:
    require_dolfinx()

    local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    if cell_data.cell_count != local_cells:
        raise ValueError(
            f"Cell data size ({cell_data.cell_count}) does not match local mesh cells ({local_cells})."
        )

    return EmbeddedBeamCoefficientFunctions(
        axial_rigidity=_create_cellwise_scalar_function(mesh, cell_data.axial_rigidity, "EA"),
        shear_rigidity_y=_create_cellwise_scalar_function(mesh, cell_data.shear_rigidity_y, "kGA_y"),
        shear_rigidity_z=_create_cellwise_scalar_function(mesh, cell_data.shear_rigidity_z, "kGA_z"),
        torsional_rigidity=_create_cellwise_scalar_function(mesh, cell_data.torsional_rigidity, "GJ"),
        bending_rigidity_y=_create_cellwise_scalar_function(mesh, cell_data.bending_rigidity_y, "EIy"),
        bending_rigidity_z=_create_cellwise_scalar_function(mesh, cell_data.bending_rigidity_z, "EIz"),
        mass_per_length=_create_cellwise_scalar_function(mesh, cell_data.mass_per_length, "rhoA"),
        rotary_inertia_t=_create_cellwise_scalar_function(mesh, cell_data.rotary_inertia_t, "rhoJ"),
        rotary_inertia_y=_create_cellwise_scalar_function(mesh, cell_data.rotary_inertia_y, "rhoIy"),
        rotary_inertia_z=_create_cellwise_scalar_function(mesh, cell_data.rotary_inertia_z, "rhoIz"),
        tangent_x=_create_cellwise_scalar_function(mesh, cell_data.tangent_x, "t_x"),
        tangent_y=_create_cellwise_scalar_function(mesh, cell_data.tangent_y, "t_y"),
        tangent_z=_create_cellwise_scalar_function(mesh, cell_data.tangent_z, "t_z"),
        normal_y_x=_create_cellwise_scalar_function(mesh, cell_data.normal_y_x, "ey_x"),
        normal_y_y=_create_cellwise_scalar_function(mesh, cell_data.normal_y_y, "ey_y"),
        normal_y_z=_create_cellwise_scalar_function(mesh, cell_data.normal_y_z, "ey_z"),
        normal_z_x=_create_cellwise_scalar_function(mesh, cell_data.normal_z_x, "ez_x"),
        normal_z_y=_create_cellwise_scalar_function(mesh, cell_data.normal_z_y, "ez_y"),
        normal_z_z=_create_cellwise_scalar_function(mesh, cell_data.normal_z_z, "ez_z"),
    )


def build_embedded_timoshenko_forms(
    space_bundle: EmbeddedBeamFunctionSpaceBundle,
    coefficients: EmbeddedBeamCoefficientFunctions,
) -> EmbeddedBeamFormBundle:
    require_dolfinx()

    import ufl

    displacement_rotation = ufl.TrialFunction(space_bundle.mixed_space)
    test_field = ufl.TestFunction(space_bundle.mixed_space)
    displacement, rotation = ufl.split(displacement_rotation)
    displacement_test, rotation_test = ufl.split(test_field)

    tangent = ufl.as_vector(
        [
            coefficients.tangent_x,
            coefficients.tangent_y,
            coefficients.tangent_z,
        ]
    )
    local_y = ufl.as_vector(
        [
            coefficients.normal_y_x,
            coefficients.normal_y_y,
            coefficients.normal_y_z,
        ]
    )
    local_z = ufl.as_vector(
        [
            coefficients.normal_z_x,
            coefficients.normal_z_y,
            coefficients.normal_z_z,
        ]
    )

    def derivative(value: Any) -> Any:
        return ufl.dot(ufl.grad(value), tangent)

    def local_components(vector: Any) -> tuple[Any, Any, Any]:
        return (
            ufl.dot(vector, tangent),
            ufl.dot(vector, local_y),
            ufl.dot(vector, local_z),
        )

    u_t, u_y, u_z = local_components(displacement)
    v_t, v_y, v_z = local_components(displacement_test)
    theta_t, theta_y, theta_z = local_components(rotation)
    psi_t, psi_y, psi_z = local_components(rotation_test)

    axial_strain = derivative(u_t)
    axial_test = derivative(v_t)
    torsion_strain = derivative(theta_t)
    torsion_test = derivative(psi_t)
    shear_y = derivative(u_y) - theta_z
    shear_y_test = derivative(v_y) - psi_z
    shear_z = derivative(u_z) + theta_y
    shear_z_test = derivative(v_z) + psi_y
    curvature_y = derivative(theta_y)
    curvature_y_test = derivative(psi_y)
    curvature_z = derivative(theta_z)
    curvature_z_test = derivative(psi_z)

    stiffness_form = (
        coefficients.axial_rigidity * axial_strain * axial_test
        + coefficients.torsional_rigidity * torsion_strain * torsion_test
        + coefficients.shear_rigidity_y * shear_y * shear_y_test
        + coefficients.shear_rigidity_z * shear_z * shear_z_test
        + coefficients.bending_rigidity_y * curvature_y * curvature_y_test
        + coefficients.bending_rigidity_z * curvature_z * curvature_z_test
    ) * ufl.dx

    mass_form = (
        coefficients.mass_per_length * ufl.dot(displacement, displacement_test)
        + coefficients.rotary_inertia_t * theta_t * psi_t
        + coefficients.rotary_inertia_y * theta_y * psi_y
        + coefficients.rotary_inertia_z * theta_z * psi_z
    ) * ufl.dx

    return EmbeddedBeamFormBundle(
        stiffness_form=stiffness_form,
        mass_form=mass_form,
    )
