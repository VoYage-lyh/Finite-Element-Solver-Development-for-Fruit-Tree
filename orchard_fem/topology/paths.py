from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt

from orchard_fem.topology.geometry import Vec3, distance, lerp, normalize


@dataclass(frozen=True)
class BranchPath:
    start: Vec3
    end: Vec3

    def length(self) -> float:
        return distance(self.start, self.end)

    def point_at(self, station: float) -> Vec3:
        return lerp(self.start, self.end, max(0.0, min(1.0, station)))

    def direction(self) -> Vec3:
        return normalize(self.end - self.start)

    def inclination_angle_rad(self) -> float:
        direction = self.direction()
        horizontal_magnitude = sqrt((direction.x * direction.x) + (direction.y * direction.y))
        return atan2(abs(direction.z), horizontal_magnitude)
