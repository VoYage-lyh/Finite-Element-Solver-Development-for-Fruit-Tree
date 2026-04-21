from __future__ import annotations

from orchard_fem.cross_section.profile import ContourSectionProfile


def load_scan_profiles(csv_path: str) -> list[ContourSectionProfile]:
    """
    Placeholder hook for future 3D-scan-driven contour profile loading.

    Expected CSV columns:
        station, x, y, tissue
    """
    raise NotImplementedError(
        "Scan profile loading is reserved for future 3D-scan integration. "
        "Provide contour CSVs with columns: station, x, y, tissue."
    )
