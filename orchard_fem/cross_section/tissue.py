from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


Vec2 = tuple[float, float]


class TissueType(str, Enum):
    XYLEM = "xylem"
    PITH = "pith"
    PHLOEM = "phloem"


class SectionShapeKind(str, Enum):
    SOLID_ELLIPSE = "solid_ellipse"
    ELLIPTIC_RING = "elliptic_ring"
    POLYGON = "polygon"


@dataclass(frozen=True)
class RegionGeometry:
    kind: SectionShapeKind = SectionShapeKind.SOLID_ELLIPSE
    center: Vec2 = (0.0, 0.0)
    radii: Vec2 = (0.0, 0.0)
    outer_center: Vec2 = (0.0, 0.0)
    outer_radii: Vec2 = (0.0, 0.0)
    inner_center: Vec2 = (0.0, 0.0)
    inner_radii: Vec2 = (0.0, 0.0)
    outer_points: list[Vec2] = field(default_factory=list)
    inner_points: list[Vec2] = field(default_factory=list)
    samples: int = 48


@dataclass(frozen=True)
class TissueRegion:
    tissue: TissueType
    material_id: str
    geometry: RegionGeometry


@dataclass(frozen=True)
class SectionRegionProperties:
    tissue: TissueType
    material_id: str
    area: float
    centroid: Vec2
    ix_centroid: float
    iy_centroid: float


@dataclass(frozen=True)
class SectionProperties:
    total_area: float
    centroid: Vec2
    ix_centroid: float
    iy_centroid: float
    regions: list[SectionRegionProperties]
