from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from orchard_fem.cross_section.profile import (
    ContourSectionProfile,
    CrossSectionProfile,
    MeasuredSectionSeries,
    ParameterizedSectionProfile,
)
from orchard_fem.cross_section.defaults import (
    DEFAULT_PHLOEM_MATERIAL_ID,
    DEFAULT_PITH_MATERIAL_ID,
    DEFAULT_XYLEM_MATERIAL_ID,
    make_circular_section,
)
from orchard_fem.cross_section.tissue import RegionGeometry, SectionShapeKind, TissueRegion, TissueType
from orchard_fem.topology.tree import BranchPath, ObservationPoint, TreeTopology, Vec3


class MaterialModelKind(str, Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    ORTHOTROPIC_PLACEHOLDER = "orthotropic_placeholder"


class JointLawKind(str, Enum):
    NONE = "none"
    POLYNOMIAL = "polynomial"
    GAP_FRICTION = "gap_friction"


class ExcitationKind(str, Enum):
    HARMONIC_FORCE = "harmonic_force"
    HARMONIC_DISPLACEMENT = "harmonic_displacement"
    HARMONIC_ACCELERATION = "harmonic_acceleration"


class AnalysisMode(str, Enum):
    FREQUENCY_RESPONSE = "frequency_response"
    TIME_HISTORY = "time_history"


@dataclass(frozen=True)
class OrchardMetadata:
    name: str = ""
    cultivar: str = ""


@dataclass(frozen=True)
class MaterialProperties:
    material_id: str
    tissue: TissueType
    model: MaterialModelKind
    density: float
    youngs_modulus: float
    poisson_ratio: float = 0.3
    damping_ratio: float = 0.0
    nonlinear_alpha: float = 0.0


@dataclass(frozen=True)
class BranchDiscretizationHint:
    num_elements: int = 4
    hotspot: bool = False


@dataclass(frozen=True)
class BranchDefinition:
    branch_id: str
    parent_branch_id: str | None
    level: int
    path: BranchPath
    section_series: MeasuredSectionSeries
    discretization: BranchDiscretizationHint


@dataclass(frozen=True)
class JointLawDefinition:
    kind: JointLawKind = JointLawKind.NONE
    linear_scale: float = 1.0
    cubic_scale: float = 0.0
    open_scale: float = 1.0
    gap_threshold: float = 0.0


@dataclass(frozen=True)
class JointDefinition:
    joint_id: str
    parent_branch_id: str
    child_branch_id: str
    linear_stiffness_scale: float = 1.0
    law: JointLawDefinition = JointLawDefinition()


@dataclass(frozen=True)
class FruitAttachment:
    fruit_id: str
    branch_id: str
    location_s: float
    mass: float
    stiffness: float
    damping: float


@dataclass(frozen=True)
class ClampBoundaryCondition:
    branch_id: str
    support_stiffness: float = 0.0
    support_damping: float = 0.0
    cubic_stiffness: float = 0.0


@dataclass(frozen=True)
class HarmonicExcitation:
    kind: ExcitationKind
    target_branch_id: str
    amplitude: float
    phase_degrees: float
    target_node: str = "tip"
    target_component: str = "ux"
    driving_frequency_hz: float = 0.0


@dataclass(frozen=True)
class AnalysisSettings:
    mode: AnalysisMode = AnalysisMode.FREQUENCY_RESPONSE
    frequency_start_hz: float = 1.0
    frequency_end_hz: float = 25.0
    frequency_steps: int = 50
    time_step_seconds: float = 0.002
    total_time_seconds: float = 1.0
    output_stride: int = 1
    max_nonlinear_iterations: int = 12
    nonlinear_tolerance: float = 1.0e-8
    rayleigh_alpha: float = 0.0
    rayleigh_beta: float = 1.0e-4
    auto_nonlinear_levels: list[int] = field(default_factory=list)
    auto_nonlinear_cubic_scale: float = 0.0
    include_gravity_prestress: bool = False
    gravity_direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    output_csv: str = "frequency_response.csv"


@dataclass(frozen=True)
class OrchardModel:
    metadata: OrchardMetadata
    materials: list[MaterialProperties]
    topology: TreeTopology
    branches: list[BranchDefinition]
    joints: list[JointDefinition]
    fruits: list[FruitAttachment]
    clamps: list[ClampBoundaryCondition]
    excitation: HarmonicExcitation
    analysis: AnalysisSettings
    observations: list[ObservationPoint]

    def require_branch(self, branch_id: str) -> BranchDefinition:
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        raise KeyError(f"Unknown branch id: {branch_id}")


def parse_vec3(values: list[float]) -> Vec3:
    if len(values) != 3:
        raise ValueError("Expected a 3D vector")
    return Vec3(float(values[0]), float(values[1]), float(values[2]))


def parse_shape(payload: dict) -> RegionGeometry:
    kind = SectionShapeKind(payload["type"])
    samples = int(payload.get("samples", 48))

    if kind == SectionShapeKind.SOLID_ELLIPSE:
        return RegionGeometry(
            kind=kind,
            center=(float(payload["center"][0]), float(payload["center"][1])),
            radii=(float(payload["radii"][0]), float(payload["radii"][1])),
            samples=samples,
        )

    if kind == SectionShapeKind.ELLIPTIC_RING:
        return RegionGeometry(
            kind=kind,
            outer_center=(float(payload["outer_center"][0]), float(payload["outer_center"][1])),
            outer_radii=(float(payload["outer_radii"][0]), float(payload["outer_radii"][1])),
            inner_center=(float(payload["inner_center"][0]), float(payload["inner_center"][1])),
            inner_radii=(float(payload["inner_radii"][0]), float(payload["inner_radii"][1])),
            samples=samples,
        )

    if kind == SectionShapeKind.POLYGON:
        return RegionGeometry(
            kind=kind,
            outer_points=[(float(point[0]), float(point[1])) for point in payload["outer_points"]],
            inner_points=[(float(point[0]), float(point[1])) for point in payload.get("inner_points", [])],
            samples=samples,
        )

    raise ValueError(f"Unsupported shape kind: {payload['type']}")


def parse_regions(payload: list[dict]) -> list[TissueRegion]:
    return [
        TissueRegion(
            tissue=TissueType(region["tissue"]),
            material_id=str(region["material_id"]),
            geometry=parse_shape(region["shape"]),
        )
        for region in payload
    ]


def parse_section_series(
    payload: list[dict],
    available_material_ids: set[str] | None = None,
) -> MeasuredSectionSeries:
    series = MeasuredSectionSeries()
    for station in payload:
        station_value = float(station["s"])
        shorthand = station.get("shorthand")
        if shorthand is not None:
            if str(shorthand) != "circular":
                raise ValueError(f"Unsupported station shorthand: {shorthand}")

            required_material_ids = {
                DEFAULT_XYLEM_MATERIAL_ID,
                DEFAULT_PITH_MATERIAL_ID,
                DEFAULT_PHLOEM_MATERIAL_ID,
            }
            if available_material_ids is not None:
                missing = sorted(required_material_ids - available_material_ids)
                if missing:
                    raise ValueError(
                        "Circular shorthand requires materials to be defined: "
                        + ", ".join(missing)
                    )

            series.add_profile(
                make_circular_section(
                    station=station_value,
                    outer_radius=float(station["outer_radius"]),
                )
            )
            continue

        regions = parse_regions(station["regions"])
        profile_type = str(station.get("profile_type", "parameterized"))
        profile: CrossSectionProfile
        if profile_type == "parameterized":
            profile = ParameterizedSectionProfile(station_value, regions)
        elif profile_type == "contour":
            profile = ContourSectionProfile(station_value, regions)
        else:
            raise ValueError(f"Unsupported profile_type: {profile_type}")
        series.add_profile(profile)
    return series
