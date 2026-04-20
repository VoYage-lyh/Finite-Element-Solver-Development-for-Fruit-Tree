from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass, field

from orchard_fem.cross_section.integrator import SectionIntegrator
from orchard_fem.cross_section.tissue import SectionProperties, TissueRegion


class CrossSectionProfile(ABC):
    def __init__(self, station: float) -> None:
        self._station = station

    @property
    def station(self) -> float:
        return self._station

    @abstractmethod
    def evaluate(self) -> SectionProperties:
        raise NotImplementedError

    @abstractmethod
    def descriptor(self) -> str:
        raise NotImplementedError


class ParameterizedSectionProfile(CrossSectionProfile):
    def __init__(self, station: float, regions: list[TissueRegion]) -> None:
        super().__init__(station)
        self._regions = list(regions)

    @property
    def regions(self) -> list[TissueRegion]:
        return self._regions

    def evaluate(self) -> SectionProperties:
        return SectionIntegrator.integrate(self._regions)

    def descriptor(self) -> str:
        return "parameterized"


class ContourSectionProfile(CrossSectionProfile):
    def __init__(self, station: float, regions: list[TissueRegion]) -> None:
        super().__init__(station)
        self._regions = list(regions)

    @property
    def regions(self) -> list[TissueRegion]:
        return self._regions

    def evaluate(self) -> SectionProperties:
        return SectionIntegrator.integrate(self._regions)

    def descriptor(self) -> str:
        return "contour"


@dataclass
class MeasuredSectionSeries:
    profiles: list[CrossSectionProfile] = field(default_factory=list)

    def add_profile(self, profile: CrossSectionProfile) -> None:
        stations = [existing.station for existing in self.profiles]
        insert_at = bisect_left(stations, profile.station)
        self.profiles.insert(insert_at, profile)

    def stations(self) -> list[float]:
        return [profile.station for profile in self.profiles]
