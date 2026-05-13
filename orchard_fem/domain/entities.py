from __future__ import annotations

from dataclasses import dataclass, field

from orchard_fem.cross_section.profile import MeasuredSectionSeries
from orchard_fem.cross_section.tissue import TissueType
from orchard_fem.domain.enums import (
    AnalysisMode,
    ExcitationKind,
    JointLawKind,
    MaterialModelKind,
    SolverBackendKind,
)
from orchard_fem.topology import BranchPath, ObservationPoint, TreeTopology


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
    target_component: str = "ux"


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
    target_s: float | None = None   # 0..1 normalized arc-length; overrides target_node when set


@dataclass(frozen=True)
class AnalysisSettings:
    mode: AnalysisMode = AnalysisMode.FREQUENCY_RESPONSE
    solver_backend: SolverBackendKind = SolverBackendKind.FENICSX
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
    use_corotational: bool = False
    output_csv: str = "frequency_response.csv"


@dataclass(frozen=True)
class FruitDistributionPolicy:
    """Regression-based fruit placement policy.

    When present on an OrchardModel, fruit attachments are auto-generated
    at load time via `generate_fruit_attachments_for_model`.  Any explicit
    `fruits` list in the JSON takes precedence over this policy.

    Stiffness derivation:
        k_total = fruit_count * mean_detachment_force_N / detachment_displacement_m
    Damping derivation:
        c = 2 * attachment_damping_ratio * sqrt(mass * k_total)
    """
    total_fruit_count: int
    seed: int = 2026
    include_terminal_primary: bool = True
    detachment_displacement_m: float = 0.010
    attachment_damping_ratio: float = 0.05
    attachment_component: str = "ux"
    count_weight_cv: float = 0.25
    long_axis_cv: float = 0.12
    short_axis_cv: float = 0.12
    mass_residual_cv: float = 0.12
    detachment_force_cv: float = 0.15
    crack_probability: float = 0.65


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
    fruit_policy: FruitDistributionPolicy | None = None

    def require_branch(self, branch_id: str) -> BranchDefinition:
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        raise KeyError(f"Unknown branch id: {branch_id}")

    def find_joint_for_child(self, child_branch_id: str) -> JointDefinition | None:
        for joint in self.joints:
            if joint.child_branch_id == child_branch_id:
                return joint
        return None

    def find_clamp(self, branch_id: str) -> ClampBoundaryCondition | None:
        for clamp in self.clamps:
            if clamp.branch_id == branch_id:
                return clamp
        return None

    def find_observation(self, observation_id: str) -> ObservationPoint | None:
        for observation in self.observations:
            if observation.id == observation_id:
                return observation
        return None
