from __future__ import annotations

from orchard_fem.cross_section.profile import MeasuredSectionSeries, ParameterizedSectionProfile
from orchard_fem.cross_section.tissue import RegionGeometry, SectionShapeKind, TissueRegion, TissueType


DEFAULT_XYLEM_MATERIAL_ID = "xylem_default"
DEFAULT_PITH_MATERIAL_ID = "pith_default"
DEFAULT_PHLOEM_MATERIAL_ID = "phloem_default"
DEFAULT_CIRCULAR_SAMPLES = 96


def make_circular_section(
    station: float,
    outer_radius: float,
    xylem_material_id: str = DEFAULT_XYLEM_MATERIAL_ID,
    pith_material_id: str = DEFAULT_PITH_MATERIAL_ID,
    phloem_material_id: str = DEFAULT_PHLOEM_MATERIAL_ID,
    pith_radius_fraction: float = 0.15,
    phloem_thickness_fraction: float = 0.08,
) -> ParameterizedSectionProfile:
    """
    Generate a concentric three-layer circular section.
    """
    outer_radius = float(outer_radius)
    if outer_radius <= 0.0:
        raise ValueError("outer_radius must be positive")
    if not 0.0 < pith_radius_fraction < 1.0:
        raise ValueError("pith_radius_fraction must be between 0 and 1")
    if not 0.0 < phloem_thickness_fraction < 1.0:
        raise ValueError("phloem_thickness_fraction must be between 0 and 1")

    pith_radius = outer_radius * pith_radius_fraction
    xylem_outer_radius = outer_radius * (1.0 - phloem_thickness_fraction)
    if xylem_outer_radius <= pith_radius:
        raise ValueError("phloem_thickness_fraction leaves no room for the xylem ring")

    pith_region = TissueRegion(
        tissue=TissueType.PITH,
        material_id=pith_material_id,
        geometry=RegionGeometry(
            kind=SectionShapeKind.SOLID_ELLIPSE,
            center=(0.0, 0.0),
            radii=(pith_radius, pith_radius),
            samples=DEFAULT_CIRCULAR_SAMPLES,
        ),
    )
    xylem_region = TissueRegion(
        tissue=TissueType.XYLEM,
        material_id=xylem_material_id,
        geometry=RegionGeometry(
            kind=SectionShapeKind.ELLIPTIC_RING,
            outer_center=(0.0, 0.0),
            outer_radii=(xylem_outer_radius, xylem_outer_radius),
            inner_center=(0.0, 0.0),
            inner_radii=(pith_radius, pith_radius),
            samples=DEFAULT_CIRCULAR_SAMPLES,
        ),
    )
    phloem_region = TissueRegion(
        tissue=TissueType.PHLOEM,
        material_id=phloem_material_id,
        geometry=RegionGeometry(
            kind=SectionShapeKind.ELLIPTIC_RING,
            outer_center=(0.0, 0.0),
            outer_radii=(outer_radius, outer_radius),
            inner_center=(0.0, 0.0),
            inner_radii=(xylem_outer_radius, xylem_outer_radius),
            samples=DEFAULT_CIRCULAR_SAMPLES,
        ),
    )

    return ParameterizedSectionProfile(station, [pith_region, xylem_region, phloem_region])


def make_default_branch_sections(
    outer_radius_root: float,
    outer_radius_tip: float,
    num_stations: int = 2,
    **kwargs,
) -> MeasuredSectionSeries:
    """
    Generate a linearly tapered series of default circular sections.
    """
    if num_stations < 2:
        raise ValueError("num_stations must be at least 2")

    series = MeasuredSectionSeries()
    for index in range(num_stations):
        alpha = index / (num_stations - 1)
        radius = float(outer_radius_root) + alpha * (float(outer_radius_tip) - float(outer_radius_root))
        series.add_profile(make_circular_section(station=float(alpha), outer_radius=radius, **kwargs))
    return series
