from __future__ import annotations

import numpy as np
import pytest

from orchard_fem.visualization.scene3d import _physical_axis_layout, _tube_surface


def test_tube_surface_preserves_circular_sections() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.7],
            [0.2, 0.0, 1.4],
            [0.5, 0.1, 2.0],
        ]
    )
    radii = np.array([0.20, 0.16, 0.11, 0.06])

    result = _tube_surface(points, radii, np, n_theta=24)

    assert result is not None
    x_surface, y_surface, z_surface = result
    assert x_surface.shape == (4, 25)
    surface = np.stack([x_surface, y_surface, z_surface], axis=1)
    ring_distances = np.linalg.norm(surface - points[:, :, None], axis=1)
    expected_radii = np.repeat(radii[:, None], ring_distances.shape[1], axis=1)
    assert ring_distances == pytest.approx(expected_radii, rel=1.0e-12, abs=1.0e-12)
    assert surface[:, :, 0] == pytest.approx(surface[:, :, -1], abs=1.0e-12)


def test_tube_surface_removes_duplicate_centreline_points() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = _tube_surface(points, [0.2, 0.2, 0.1], np, n_theta=12)

    assert result is not None
    assert result[0].shape == (2, 13)


def test_physical_axis_layout_widens_limits_without_distorting_units() -> None:
    x_limits, y_limits, z_limits, box_aspect = _physical_axis_layout(
        -1.36,
        0.79,
        0.0,
        0.0,
        -0.04,
        2.96,
        0.068,
    )
    spans = np.array(
        [
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        ]
    )

    assert np.asarray(box_aspect) == pytest.approx(spans)
    assert spans[1] >= 0.35 * max(spans[0], spans[2])
    assert (y_limits[0] + y_limits[1]) / 2.0 == pytest.approx(0.0)
