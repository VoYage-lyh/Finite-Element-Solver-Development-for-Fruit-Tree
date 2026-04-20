from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from orchard_fem.cross_section.tissue import (
    RegionGeometry,
    SectionProperties,
    SectionRegionProperties,
    SectionShapeKind,
    TissueRegion,
    Vec2,
)


@dataclass(frozen=True)
class _LoopProperties:
    area: float
    centroid: Vec2
    ix_origin: float
    iy_origin: float


def _signed_area(points: list[Vec2]) -> float:
    twice_area = 0.0
    for index, point_a in enumerate(points):
        point_b = points[(index + 1) % len(points)]
        twice_area += (point_a[0] * point_b[1]) - (point_b[0] * point_a[1])
    return 0.5 * twice_area


def _ensure_positive_orientation(points: list[Vec2]) -> list[Vec2]:
    if len(points) < 3:
        raise ValueError("A polygon loop needs at least three points")
    return list(reversed(points)) if _signed_area(points) < 0.0 else points


def _sample_ellipse(center: Vec2, radii: Vec2, requested_samples: int) -> list[Vec2]:
    samples = max(requested_samples, 16)
    return [
        (
            center[0] + radii[0] * cos((2.0 * pi * index) / samples),
            center[1] + radii[1] * sin((2.0 * pi * index) / samples),
        )
        for index in range(samples)
    ]


def _integrate_loop(points: list[Vec2]) -> _LoopProperties:
    ordered_points = _ensure_positive_orientation(points)
    cross_sum = 0.0
    centroid_x_sum = 0.0
    centroid_y_sum = 0.0
    ix_sum = 0.0
    iy_sum = 0.0

    for index, point_a in enumerate(ordered_points):
        point_b = ordered_points[(index + 1) % len(ordered_points)]
        cross = (point_a[0] * point_b[1]) - (point_b[0] * point_a[1])
        cross_sum += cross
        centroid_x_sum += (point_a[0] + point_b[0]) * cross
        centroid_y_sum += (point_a[1] + point_b[1]) * cross
        ix_sum += ((point_a[1] ** 2) + (point_a[1] * point_b[1]) + (point_b[1] ** 2)) * cross
        iy_sum += ((point_a[0] ** 2) + (point_a[0] * point_b[0]) + (point_b[0] ** 2)) * cross

    area = 0.5 * cross_sum
    if abs(area) < 1.0e-12:
        raise ValueError("Degenerate polygon loop has near-zero area")

    return _LoopProperties(
        area=area,
        centroid=(centroid_x_sum / (6.0 * area), centroid_y_sum / (6.0 * area)),
        ix_origin=ix_sum / 12.0,
        iy_origin=iy_sum / 12.0,
    )


def _build_region_properties(
    region: TissueRegion,
    area: float,
    centroid: Vec2,
    ix_origin: float,
    iy_origin: float,
) -> SectionRegionProperties:
    ix_centroid = max(ix_origin - area * centroid[1] * centroid[1], 0.0)
    iy_centroid = max(iy_origin - area * centroid[0] * centroid[0], 0.0)
    return SectionRegionProperties(
        tissue=region.tissue,
        material_id=region.material_id,
        area=area,
        centroid=centroid,
        ix_centroid=ix_centroid,
        iy_centroid=iy_centroid,
    )


def _integrate_region(region: TissueRegion) -> SectionRegionProperties:
    geometry: RegionGeometry = region.geometry

    if geometry.kind == SectionShapeKind.SOLID_ELLIPSE:
        outer = _integrate_loop(_sample_ellipse(geometry.center, geometry.radii, geometry.samples))
        return _build_region_properties(region, outer.area, outer.centroid, outer.ix_origin, outer.iy_origin)

    if geometry.kind == SectionShapeKind.ELLIPTIC_RING:
        outer_points = _sample_ellipse(geometry.outer_center, geometry.outer_radii, geometry.samples)
        inner_points = _sample_ellipse(geometry.inner_center, geometry.inner_radii, geometry.samples)
    elif geometry.kind == SectionShapeKind.POLYGON:
        outer_points = geometry.outer_points
        inner_points = geometry.inner_points
    else:
        raise ValueError(f"Unsupported section shape kind: {geometry.kind}")

    outer = _integrate_loop(list(outer_points))
    if not inner_points:
        return _build_region_properties(region, outer.area, outer.centroid, outer.ix_origin, outer.iy_origin)

    inner = _integrate_loop(list(inner_points))
    area = outer.area - inner.area
    if area <= 0.0:
        raise ValueError("Ring/polygon region has non-positive area")

    centroid = (
        ((outer.area * outer.centroid[0]) - (inner.area * inner.centroid[0])) / area,
        ((outer.area * outer.centroid[1]) - (inner.area * inner.centroid[1])) / area,
    )
    return _build_region_properties(
        region=region,
        area=area,
        centroid=centroid,
        ix_origin=outer.ix_origin - inner.ix_origin,
        iy_origin=outer.iy_origin - inner.iy_origin,
    )


class SectionIntegrator:
    @staticmethod
    def integrate(regions: list[TissueRegion]) -> SectionProperties:
        if not regions:
            raise ValueError("Cross-section profile requires at least one tissue region")

        total_area = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        ix_origin_sum = 0.0
        iy_origin_sum = 0.0
        region_properties: list[SectionRegionProperties] = []

        for region in regions:
            properties = _integrate_region(region)
            region_properties.append(properties)
            total_area += properties.area
            weighted_x += properties.area * properties.centroid[0]
            weighted_y += properties.area * properties.centroid[1]
            ix_origin_sum += properties.ix_centroid + properties.area * properties.centroid[1] ** 2
            iy_origin_sum += properties.iy_centroid + properties.area * properties.centroid[0] ** 2

        if total_area <= 0.0:
            raise ValueError("Cross-section total area must be positive")

        centroid = (weighted_x / total_area, weighted_y / total_area)
        ix_centroid = max(ix_origin_sum - total_area * centroid[1] ** 2, 0.0)
        iy_centroid = max(iy_origin_sum - total_area * centroid[0] ** 2, 0.0)
        return SectionProperties(
            total_area=total_area,
            centroid=centroid,
            ix_centroid=ix_centroid,
            iy_centroid=iy_centroid,
            regions=region_properties,
        )
