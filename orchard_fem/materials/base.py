from __future__ import annotations

from dataclasses import dataclass

from orchard_fem.cross_section.profile import MeasuredSectionSeries
from orchard_fem.domain import BranchDefinition, MaterialModelKind, MaterialProperties


@dataclass(frozen=True)
class BranchAverageProperties:
    length: float
    average_area: float
    average_ix: float
    average_iy: float
    average_polar_moment: float
    average_mass_per_length: float
    average_youngs_modulus: float
    average_shear_modulus: float
    average_damping_ratio: float


@dataclass(frozen=True)
class BranchSectionState:
    station: float
    area: float
    ix: float
    iy: float
    polar_moment: float
    mass_per_length: float
    effective_youngs_modulus: float
    effective_shear_modulus: float
    effective_poisson_ratio: float
    damping_ratio: float


def build_material_lookup(materials: list[MaterialProperties]) -> dict[str, MaterialProperties]:
    return {material.material_id: material for material in materials}


class MaterialLibrary:
    def __init__(self, materials: list[MaterialProperties] | None = None) -> None:
        self._materials = build_material_lookup(materials or [])

    def add_linear_elastic(self, properties: MaterialProperties) -> None:
        self._materials[properties.material_id] = properties

    def add_nonlinear_elastic(self, properties: MaterialProperties) -> None:
        self._materials[properties.material_id] = properties

    def add_orthotropic_placeholder(self, properties: MaterialProperties) -> None:
        self._materials[properties.material_id] = properties

    def contains(self, material_id: str) -> bool:
        return material_id in self._materials

    def require(self, material_id: str) -> MaterialProperties:
        try:
            return self._materials[material_id]
        except KeyError as exc:
            raise KeyError(f"Unknown material id: {material_id}") from exc

    def ids(self) -> list[str]:
        return list(self._materials)


class SpatialMaterialField:
    def __init__(self, materials: MaterialLibrary) -> None:
        self._materials = materials

    def resolve(self, material_id: str, station: float) -> MaterialProperties:
        del station
        return self._materials.require(material_id)


def _tangent_modulus(material: MaterialProperties, generalized_strain: float = 0.0) -> float:
    if material.model == MaterialModelKind.LINEAR:
        return material.youngs_modulus
    if material.model == MaterialModelKind.NONLINEAR:
        return material.youngs_modulus * (1.0 + material.nonlinear_alpha * generalized_strain * generalized_strain)
    if material.model == MaterialModelKind.ORTHOTROPIC_PLACEHOLDER:
        return material.youngs_modulus
    raise ValueError(f"Unsupported material model kind: {material.model}")


def _evaluate_profile_state(
    station: float,
    section_series: MeasuredSectionSeries,
    material_lookup: dict[str, MaterialProperties],
) -> BranchSectionState:
    for profile in section_series.profiles:
        if abs(profile.station - station) <= 1.0e-12:
            properties = profile.evaluate()
            break
    else:
        raise ValueError(f"No section profile exactly defined at station {station}")

    modulus_area_sum = 0.0
    poisson_area_sum = 0.0
    damping_weight = 0.0
    mass_per_length = 0.0
    density_area_sum = 0.0

    for region in properties.regions:
        material = material_lookup[region.material_id]
        region_mass = material.density * region.area
        mass_per_length += region_mass
        modulus_area_sum += _tangent_modulus(material) * region.area
        poisson_area_sum += material.poisson_ratio * region.area
        damping_weight += region_mass * material.damping_ratio
        density_area_sum += region.area

    if density_area_sum <= 0.0 or properties.total_area <= 0.0:
        raise ValueError("Branch section state requires positive area")

    effective_youngs_modulus = modulus_area_sum / density_area_sum
    effective_poisson_ratio = poisson_area_sum / density_area_sum
    effective_shear_modulus = effective_youngs_modulus / (2.0 * (1.0 + effective_poisson_ratio))
    damping_ratio = damping_weight / mass_per_length

    return BranchSectionState(
        station=station,
        area=properties.total_area,
        ix=properties.ix_centroid,
        iy=properties.iy_centroid,
        polar_moment=properties.ix_centroid + properties.iy_centroid,
        mass_per_length=mass_per_length,
        effective_youngs_modulus=effective_youngs_modulus,
        effective_shear_modulus=effective_shear_modulus,
        effective_poisson_ratio=effective_poisson_ratio,
        damping_ratio=damping_ratio,
    )


def _interpolate(left: float, right: float, alpha: float) -> float:
    return (1.0 - alpha) * left + alpha * right


def _evaluate_profile_states(
    section_series: MeasuredSectionSeries,
    material_lookup: dict[str, MaterialProperties],
) -> list[BranchSectionState]:
    return [
        _evaluate_profile_state(profile.station, section_series, material_lookup)
        for profile in section_series.profiles
    ]


def evaluate_branch_section_state(
    branch: BranchDefinition,
    material_lookup: dict[str, MaterialProperties],
    station: float,
) -> BranchSectionState:
    states = _evaluate_profile_states(branch.section_series, material_lookup)
    if not states:
        raise ValueError(f"Branch {branch.branch_id} has no section stations")

    station = max(0.0, min(1.0, station))
    if len(states) == 1 or station <= states[0].station:
        return states[0]
    if station >= states[-1].station:
        return states[-1]

    for index in range(len(states) - 1):
        left = states[index]
        right = states[index + 1]
        if left.station <= station <= right.station:
            span = right.station - left.station
            alpha = 0.0 if span <= 1.0e-12 else (station - left.station) / span
            return BranchSectionState(
                station=station,
                area=_interpolate(left.area, right.area, alpha),
                ix=_interpolate(left.ix, right.ix, alpha),
                iy=_interpolate(left.iy, right.iy, alpha),
                polar_moment=_interpolate(left.polar_moment, right.polar_moment, alpha),
                mass_per_length=_interpolate(left.mass_per_length, right.mass_per_length, alpha),
                effective_youngs_modulus=_interpolate(left.effective_youngs_modulus, right.effective_youngs_modulus, alpha),
                effective_shear_modulus=_interpolate(left.effective_shear_modulus, right.effective_shear_modulus, alpha),
                effective_poisson_ratio=_interpolate(left.effective_poisson_ratio, right.effective_poisson_ratio, alpha),
                damping_ratio=_interpolate(left.damping_ratio, right.damping_ratio, alpha),
            )

    return states[-1]


def report_branch_average_properties(
    branch: BranchDefinition,
    material_lookup: dict[str, MaterialProperties],
) -> BranchAverageProperties:
    states = _evaluate_profile_states(branch.section_series, material_lookup)
    if not states:
        raise ValueError(f"Branch {branch.branch_id} has no section stations")

    count = float(len(states))
    return BranchAverageProperties(
        length=branch.path.length(),
        average_area=sum(state.area for state in states) / count,
        average_ix=sum(state.ix for state in states) / count,
        average_iy=sum(state.iy for state in states) / count,
        average_polar_moment=sum(state.polar_moment for state in states) / count,
        average_mass_per_length=sum(state.mass_per_length for state in states) / count,
        average_youngs_modulus=sum(state.effective_youngs_modulus for state in states) / count,
        average_shear_modulus=sum(state.effective_shear_modulus for state in states) / count,
        average_damping_ratio=sum(state.damping_ratio for state in states) / count,
    )
