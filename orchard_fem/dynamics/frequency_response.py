from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi

from orchard_fem.discretization import LinearDynamicAssemblyResult, OrchardSystemAssembler
from orchard_fem.dynamics.excitation import build_frequency_excitation_load
from orchard_fem.dynamics.time_history import _solve_time_history_execution
from orchard_fem.io import load_orchard_model
from orchard_fem.io.csv_writer import FrequencyResponseRow, write_frequency_response_csv
from orchard_fem.numerics import create_aij_matrix, require_petsc, solve_linear_system


@dataclass(frozen=True)
class FrequencyResponseRequest:
    model_path: str
    output_csv: str


@dataclass(frozen=True)
class FrequencyResponsePoint:
    frequency_hz: float
    excitation_response_magnitude: float
    observation_magnitudes: list[float]


@dataclass(frozen=True)
class FrequencyResponseResult:
    observation_names: list[str]
    points: list[FrequencyResponsePoint]

    def write_csv(self, file_path: str) -> None:
        write_frequency_response_csv(
            file_path,
            self.observation_names,
            [
                FrequencyResponseRow(
                    frequency_hz=point.frequency_hz,
                    excitation_response=point.excitation_response_magnitude,
                    observation_values=point.observation_magnitudes,
                )
                for point in self.points
            ],
        )


def _frequency_grid(analysis) -> list[float]:
    steps = max(analysis.frequency_steps, 1)
    grid: list[float] = []
    for step_index in range(steps):
        alpha = 0.0 if steps == 1 else step_index / (steps - 1)
        grid.append(
            analysis.frequency_start_hz
            + (alpha * (analysis.frequency_end_hz - analysis.frequency_start_hz))
        )
    return grid


def _estimate_steady_state_amplitude(values: list[float]) -> float:
    if not values:
        return 0.0
    start_index = len(values) // 2
    return max((abs(value) for value in values[start_index:]), default=0.0)


def _build_real_block_matrix(
    stiffness_matrix: list[list[float]],
    mass_matrix: list[list[float]],
    damping_matrix: list[list[float]],
    omega: float,
) -> list[list[float]]:
    size = len(stiffness_matrix)
    block = [[0.0 for _ in range(2 * size)] for _ in range(2 * size)]

    for row_index in range(size):
        for column_index in range(size):
            real_value = (
                stiffness_matrix[row_index][column_index]
                - ((omega * omega) * mass_matrix[row_index][column_index])
            )
            imag_value = omega * damping_matrix[row_index][column_index]

            block[row_index][column_index] = real_value
            block[row_index][size + column_index] = -imag_value
            block[size + row_index][column_index] = imag_value
            block[size + row_index][size + column_index] = real_value

    return block


def solve_frequency_response_system(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
) -> FrequencyResponseResult:
    require_petsc()

    if assembled.nonlinear_links:
        points: list[FrequencyResponsePoint] = []
        continuation_state = None
        for frequency_hz in _frequency_grid(analysis):
            sweep_excitation = replace(excitation, driving_frequency_hz=frequency_hz)
            execution = _solve_time_history_execution(
                assembled,
                sweep_excitation,
                analysis,
                initial_state=continuation_state,
            )
            response = execution.result
            continuation_state = execution.final_state
            points.append(
                FrequencyResponsePoint(
                    frequency_hz=frequency_hz,
                    excitation_response_magnitude=_estimate_steady_state_amplitude(
                        [point.excitation_response_value for point in response.points]
                    ),
                    observation_magnitudes=[
                        _estimate_steady_state_amplitude(
                            [
                                point.observation_values[observation_index]
                                for point in response.points
                            ]
                        )
                        for observation_index in range(len(response.observation_names))
                    ],
                )
            )
        return FrequencyResponseResult(
            observation_names=assembled.observation_names,
            points=points,
        )

    points: list[FrequencyResponsePoint] = []
    for frequency_hz in _frequency_grid(analysis):
        omega = 2.0 * pi * frequency_hz
        block_matrix = _build_real_block_matrix(
            assembled.stiffness_matrix,
            assembled.mass_matrix,
            assembled.damping_matrix,
            omega,
        )
        load_real, load_imag = build_frequency_excitation_load(
            assembled.stiffness_matrix,
            assembled.mass_matrix,
            assembled.damping_matrix,
            assembled.excitation_dof,
            excitation,
            omega,
        )

        rhs = [0.0 for _ in range(2 * len(assembled.dof_labels))]
        rhs[assembled.excitation_dof] = load_real
        rhs[len(assembled.dof_labels) + assembled.excitation_dof] = load_imag
        response = solve_linear_system(create_aij_matrix(block_matrix), rhs)

        real = response[: len(assembled.dof_labels)]
        imag = response[len(assembled.dof_labels) :]
        excitation_magnitude = (
            (real[assembled.excitation_dof] ** 2) + (imag[assembled.excitation_dof] ** 2)
        ) ** 0.5
        observation_magnitudes = [
            ((real[dof] ** 2) + (imag[dof] ** 2)) ** 0.5
            for dof in assembled.observation_dofs
        ]
        points.append(
            FrequencyResponsePoint(
                frequency_hz=frequency_hz,
                excitation_response_magnitude=excitation_magnitude,
                observation_magnitudes=observation_magnitudes,
            )
        )

    return FrequencyResponseResult(
        observation_names=assembled.observation_names,
        points=points,
    )


class PETScFrequencyResponseSolver:
    def solve(self, request: FrequencyResponseRequest) -> FrequencyResponseResult:
        require_petsc()

        model = load_orchard_model(request.model_path)
        assembled = OrchardSystemAssembler().assemble(model)
        result = solve_frequency_response_system(
            assembled,
            model.excitation,
            model.analysis,
        )
        result.write_csv(request.output_csv)
        return result
