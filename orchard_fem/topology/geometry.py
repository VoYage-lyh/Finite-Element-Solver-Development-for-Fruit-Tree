from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Vec3":
        return Vec3(self.x * factor, self.y * factor, self.z * factor)


def dot(left: Vec3, right: Vec3) -> float:
    return (left.x * right.x) + (left.y * right.y) + (left.z * right.z)


def norm(value: Vec3) -> float:
    return sqrt(dot(value, value))


def distance(left: Vec3, right: Vec3) -> float:
    return norm(left - right)


def normalize(value: Vec3) -> Vec3:
    magnitude = norm(value)
    if magnitude <= 1.0e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return value.scale(1.0 / magnitude)


def lerp(left: Vec3, right: Vec3, alpha: float) -> Vec3:
    return left.scale(1.0 - alpha) + right.scale(alpha)
