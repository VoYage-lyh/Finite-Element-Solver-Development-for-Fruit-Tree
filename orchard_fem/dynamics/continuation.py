from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class ContinuationPoint(Generic[StateT]):
    frequency_hz: float
    state: StateT
    inserted_substeps: int = 0


def solve_frequency_continuation(
    target_frequencies_hz: list[float],
    solve_point: Callable[[float, StateT | None], StateT],
    relative_change: Callable[[StateT, StateT | None], float],
    *,
    max_relative_step_change: float = 0.35,
    max_bisections: int = 8,
) -> list[ContinuationPoint[StateT]]:
    if not target_frequencies_hz:
        return []
    if max_relative_step_change <= 0.0:
        raise ValueError("max_relative_step_change must be positive.")
    if max_bisections < 0:
        raise ValueError("max_bisections must be non-negative.")

    output: list[ContinuationPoint[StateT]] = []
    current_frequency = float(target_frequencies_hz[0])
    current_state = solve_point(current_frequency, None)
    output.append(
        ContinuationPoint(
            frequency_hz=current_frequency,
            state=current_state,
            inserted_substeps=0,
        )
    )

    def advance(
        start_frequency: float,
        start_state: StateT,
        target_frequency: float,
        *,
        depth: int,
    ) -> tuple[StateT, int]:
        try:
            target_state = solve_point(target_frequency, start_state)
        except Exception:
            if depth >= max_bisections:
                raise
            midpoint = 0.5 * (start_frequency + target_frequency)
            midpoint_state, first_insertions = advance(
                start_frequency,
                start_state,
                midpoint,
                depth=depth + 1,
            )
            final_state, second_insertions = advance(
                midpoint,
                midpoint_state,
                target_frequency,
                depth=depth + 1,
            )
            return final_state, first_insertions + second_insertions + 1

        change = relative_change(target_state, start_state)
        if change <= max_relative_step_change or depth >= max_bisections:
            return target_state, 0

        midpoint = 0.5 * (start_frequency + target_frequency)
        midpoint_state, first_insertions = advance(
            start_frequency,
            start_state,
            midpoint,
            depth=depth + 1,
        )
        final_state, second_insertions = advance(
            midpoint,
            midpoint_state,
            target_frequency,
            depth=depth + 1,
        )
        return final_state, first_insertions + second_insertions + 1

    for target_frequency in target_frequencies_hz[1:]:
        final_state, inserted_substeps = advance(
            current_frequency,
            current_state,
            float(target_frequency),
            depth=0,
        )
        current_frequency = float(target_frequency)
        current_state = final_state
        output.append(
            ContinuationPoint(
                frequency_hz=current_frequency,
                state=current_state,
                inserted_substeps=inserted_substeps,
            )
        )

    return output
