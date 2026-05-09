from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt

from orchard_fem.topology.geometry import Vec3, distance, lerp, normalize


@dataclass(frozen=True)
class BranchPath:
    start: Vec3
    end: Vec3
    control_points: tuple[Vec3, ...] = ()

    def points(self) -> tuple[Vec3, ...]:
        return (self.start, *self.control_points, self.end)

    def length(self) -> float:
        points = self.points()
        return sum(
            distance(points[index], points[index + 1])
            for index in range(len(points) - 1)
        )

    def point_at(self, station: float) -> Vec3:
        clamped_station = max(0.0, min(1.0, station))
        points = self.points()
        if len(points) == 2:
            return lerp(self.start, self.end, clamped_station)

        total_length = self.length()
        if total_length <= 1.0e-12:
            return self.start
        target_length = clamped_station * total_length
        traversed = 0.0
        for index in range(len(points) - 1):
            first = points[index]
            second = points[index + 1]
            segment_length = distance(first, second)
            if segment_length <= 1.0e-12:
                continue
            if traversed + segment_length >= target_length:
                return lerp(first, second, (target_length - traversed) / segment_length)
            traversed += segment_length
        return self.end

    def direction(self) -> Vec3:
        return normalize(self.end - self.start)

    def inclination_angle_rad(self) -> float:
        direction = self.direction()
        horizontal_magnitude = sqrt((direction.x * direction.x) + (direction.y * direction.y))
        return atan2(abs(direction.z), horizontal_magnitude)
