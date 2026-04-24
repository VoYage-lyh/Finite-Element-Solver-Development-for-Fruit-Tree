from __future__ import annotations

from orchard_fem.cross_section.defaults import (
    DEFAULT_PHLOEM_MATERIAL_ID,
    DEFAULT_PITH_MATERIAL_ID,
    DEFAULT_XYLEM_MATERIAL_ID,
    make_circular_section,
)
from orchard_fem.cross_section.profile import (
    ContourSectionProfile,
    CrossSectionProfile,
    MeasuredSectionSeries,
    ParameterizedSectionProfile,
)
from orchard_fem.cross_section.tissue import (
    RegionGeometry,
    SectionShapeKind,
    TissueRegion,
    TissueType,
)
from orchard_fem.topology import Vec3


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
            inner_points=[
                (float(point[0]), float(point[1]))
                for point in payload.get("inner_points", [])
            ],
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
