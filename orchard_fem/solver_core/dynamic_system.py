from orchard_fem.discretization import (
    LinearDynamicAssemblyResult,
    NonlinearLinkDefinition,
    NonlinearLinkKind,
)
from orchard_fem.dynamics import (
    FrequencyResponsePoint,
    FrequencyResponseResult,
    TimeHistoryPoint,
    TimeHistoryResult,
)

DynamicSystem = LinearDynamicAssemblyResult
NonlinearLink = NonlinearLinkDefinition

__all__ = [
    "DynamicSystem",
    "FrequencyResponsePoint",
    "FrequencyResponseResult",
    "NonlinearLink",
    "NonlinearLinkDefinition",
    "NonlinearLinkKind",
    "TimeHistoryPoint",
    "TimeHistoryResult",
]
