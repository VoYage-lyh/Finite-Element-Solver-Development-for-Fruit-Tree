from __future__ import annotations

import pytest

from orchard_fem.dynamics.continuation import solve_frequency_continuation


def test_frequency_continuation_inserts_substeps_without_changing_output_grid() -> None:
    solved_frequencies: list[float] = []

    def solve_point(frequency_hz: float, previous_state: float | None) -> float:
        del previous_state
        solved_frequencies.append(frequency_hz)
        return frequency_hz

    points = solve_frequency_continuation(
        [1.0, 2.0],
        solve_point,
        lambda current, previous: float("inf") if previous is None else abs(current - previous),
        max_relative_step_change=0.30,
        max_bisections=8,
    )

    assert [point.frequency_hz for point in points] == [1.0, 2.0]
    assert points[-1].inserted_substeps > 0
    assert len(solved_frequencies) > len(points)
    assert solved_frequencies[0] == pytest.approx(1.0)
    assert solved_frequencies[-1] == pytest.approx(2.0)


def test_frequency_continuation_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="max_relative_step_change"):
        solve_frequency_continuation(
            [1.0],
            lambda frequency, state: frequency,
            lambda a, b: 0.0,
            max_relative_step_change=0.0,
        )
