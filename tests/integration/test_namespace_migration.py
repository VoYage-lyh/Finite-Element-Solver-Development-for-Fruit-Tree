from orchard_fem.discretization import LinearDynamicAssemblyResult, NonlinearLinkDefinition
from orchard_fem.branches import (
    BranchAverageProperties,
    BranchDefinition,
    BranchDiscretizationHint,
    BranchPath,
    BranchSectionState,
)
from orchard_fem.discretization import AssembledModel, BeamElementProperties, DOFManager
from orchard_fem.error_metrics import ErrorMetrics
from orchard_fem.excitation_and_bc import (
    AnalysisMode,
    AnalysisSettings,
    ClampBoundaryCondition,
    ExcitationKind,
    HarmonicExcitation,
)
from orchard_fem.fruits import FruitAttachment
from orchard_fem.geometry_topology import ObservationPoint, TopologyNode, TreeTopology, Vec3
from orchard_fem.joints_and_bifurcations import (
    JointDefinition,
    JointLawDefinition,
    JointLawKind,
    NonlinearLink,
    NonlinearLinkKind,
)
from orchard_fem.model_reduction import ReducedBasis, ReductionStrategy
from orchard_fem.materials import MaterialLibrary, SpatialMaterialField
from orchard_fem.solver_core import DynamicSystem


def test_migration_namespaces_cover_archived_cpp_module_surface() -> None:
    assert AssembledModel is LinearDynamicAssemblyResult
    assert BeamElementProperties is not None
    assert BranchDefinition is not None
    assert BranchAverageProperties is not None
    assert BranchDiscretizationHint is not None
    assert BranchPath is not None
    assert BranchSectionState is not None
    assert DOFManager is not None
    assert AnalysisMode is not None
    assert AnalysisSettings is not None
    assert ClampBoundaryCondition is not None
    assert ExcitationKind is not None
    assert HarmonicExcitation is not None
    assert FruitAttachment is not None
    assert JointDefinition is not None
    assert JointLawDefinition is not None
    assert JointLawKind is not None
    assert MaterialLibrary is not None
    assert ObservationPoint is not None
    assert SpatialMaterialField is not None
    assert TopologyNode is not None
    assert TreeTopology is not None
    assert Vec3 is not None
    assert DynamicSystem is LinearDynamicAssemblyResult
    assert NonlinearLink is NonlinearLinkDefinition
    assert NonlinearLinkKind is not None
    assert ReducedBasis(vectors=[]).vectors == []
    assert issubclass(ReductionStrategy, object)
    assert ErrorMetrics().runtime_speedup == 1.0
