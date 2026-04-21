from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from orchard_fem.io.csv_writer import FrequencyResponseRow, write_frequency_response_csv
from orchard_fem.io.legacy_loader import load_orchard_model
from orchard_fem.model import ExcitationKind
from orchard_fem.solvers._petsc import create_aij_matrix, require_petsc, solve_linear_system
from orchard_fem.solvers.modal_assembler import OrchardSystemAssembler


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


def _build_frequency_excitation_load(
    stiffness_matrix: list[list[float]],
    mass_matrix: list[list[float]],
    damping_matrix: list[list[float]],
    excitation_dof: int,
    excitation,
    omega: float,
) -> tuple[float, float]:
    phase_radians = excitation.phase_degrees * (pi / 180.0)
    base_real = excitation.amplitude * cos(phase_radians)
    base_imag = excitation.amplitude * sin(phase_radians)

    if excitation.kind == ExcitationKind.HARMONIC_FORCE:
        return base_real, base_imag

    if excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        real_scale = stiffness_matrix[excitation_dof][excitation_dof]
        imag_scale = omega * damping_matrix[excitation_dof][excitation_dof]
        return (
            (base_real * real_scale) - (base_imag * imag_scale),
            (base_real * imag_scale) + (base_imag * real_scale),
        )

    if excitation.kind == ExcitationKind.HARMONIC_ACCELERATION:
        scale = mass_matrix[excitation_dof][excitation_dof]
        return base_real * scale, base_imag * scale

    raise ValueError(f"Unsupported excitation kind: {excitation.kind}")


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


class PETScFrequencyResponseSolver:
    def solve(self, request: FrequencyResponseRequest) -> FrequencyResponseResult:
        require_petsc()

        model = load_orchard_model(request.model_path)
        assembled = OrchardSystemAssembler().assemble(model)
        steps = max(model.analysis.frequency_steps, 1)
        points: list[FrequencyResponsePoint] = []

        for step_index in range(steps):
            alpha = 0.0 if steps == 1 else step_index / (steps - 1)
            frequency_hz = model.analysis.frequency_start_hz + (
                alpha * (model.analysis.frequency_end_hz - model.analysis.frequency_start_hz)
            )
            omega = 2.0 * pi * frequency_hz
            block_matrix = _build_real_block_matrix(
                assembled.stiffness_matrix,
                assembled.mass_matrix,
                assembled.damping_matrix,
                omega,
            )
            load_real, load_imag = _build_frequency_excitation_load(
                assembled.stiffness_matrix,
                assembled.mass_matrix,
                assembled.damping_matrix,
                assembled.excitation_dof,
                model.excitation,
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

        result = FrequencyResponseResult(
            observation_names=assembled.observation_names,
            points=points,
        )
        result.write_csv(request.output_csv)
        return result
