from __future__ import annotations

from dataclasses import dataclass
from math import pi

from orchard_fem.discretization import LinearDynamicAssemblyResult, OrchardSystemAssembler
from orchard_fem.discretization.types import Matrix
from orchard_fem.dynamics.continuation import solve_frequency_continuation
from orchard_fem.dynamics.harmonic_balance import (
    first_harmonic_link_response,
    relative_complex_components,
)
from orchard_fem.dynamics.excitation import build_frequency_excitation_load
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


def _copy_matrix(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def _relative_response_change(current: list[float], previous: list[float] | None) -> float:
    if previous is None:
        return float("inf")
    numerator = max((abs(a - b) for a, b in zip(current, previous)), default=0.0)
    denominator = max(max((abs(value) for value in current), default=0.0), 1.0)
    return numerator / denominator


def _solve_frequency_point(
    stiffness_matrix: Matrix,
    assembled: LinearDynamicAssemblyResult,
    excitation,
    omega: float,
) -> list[float]:
    block_matrix = _build_real_block_matrix(
        stiffness_matrix,
        assembled.mass_matrix,
        assembled.damping_matrix,
        omega,
    )
    load_real, load_imag = build_frequency_excitation_load(
        stiffness_matrix,
        assembled.mass_matrix,
        assembled.damping_matrix,
        assembled.excitation_dof,
        excitation,
        omega,
    )

    dof_count = len(assembled.dof_labels)
    rhs = [0.0 for _ in range(2 * dof_count)]
    rhs[assembled.excitation_dof] = load_real
    rhs[dof_count + assembled.excitation_dof] = load_imag
    return solve_linear_system(create_aij_matrix(block_matrix), rhs)


def _build_frequency_rhs(
    stiffness_matrix: Matrix,
    assembled: LinearDynamicAssemblyResult,
    excitation,
    omega: float,
) -> list[float]:
    load_real, load_imag = build_frequency_excitation_load(
        stiffness_matrix,
        assembled.mass_matrix,
        assembled.damping_matrix,
        assembled.excitation_dof,
        excitation,
        omega,
    )
    dof_count = len(assembled.dof_labels)
    rhs = [0.0 for _ in range(2 * dof_count)]
    rhs[assembled.excitation_dof] = load_real
    rhs[dof_count + assembled.excitation_dof] = load_imag
    return rhs


def _matrix_vector_product(matrix: Matrix, vector: list[float]) -> list[float]:
    return [sum(value * vector[column] for column, value in enumerate(row)) for row in matrix]


def _infinity_norm(vector: list[float]) -> float:
    return max((abs(value) for value in vector), default=0.0)


def _scatter_first_harmonic_link(
    residual: list[float],
    jacobian: Matrix,
    *,
    dof_count: int,
    response: list[float],
    link,
) -> None:
    real = response[:dof_count]
    imag = response[dof_count:]
    relative_real, relative_imag = relative_complex_components(real, imag, link)
    link_response = first_harmonic_link_response(link, relative_real, relative_imag)

    signed_dofs = [(link.first_dof, 1.0)]
    if link.second_dof >= 0:
        signed_dofs.append((link.second_dof, -1.0))

    for row_dof, row_sign in signed_dofs:
        residual[row_dof] += row_sign * link_response.force_real
        residual[dof_count + row_dof] += row_sign * link_response.force_imag
        for column_dof, column_sign in signed_dofs:
            sign = row_sign * column_sign
            jacobian[row_dof][column_dof] += sign * link_response.tangent_rr
            jacobian[row_dof][dof_count + column_dof] += (
                sign * link_response.tangent_ri
            )
            jacobian[dof_count + row_dof][column_dof] += (
                sign * link_response.tangent_ir
            )
            jacobian[dof_count + row_dof][dof_count + column_dof] += (
                sign * link_response.tangent_ii
            )


def _build_harmonic_balance_residual_and_jacobian(
    linear_block_matrix: Matrix,
    rhs: list[float],
    assembled: LinearDynamicAssemblyResult,
    response: list[float],
) -> tuple[list[float], Matrix]:
    residual = [
        value - rhs[row_index]
        for row_index, value in enumerate(
            _matrix_vector_product(linear_block_matrix, response)
        )
    ]
    jacobian = _copy_matrix(linear_block_matrix)
    dof_count = len(assembled.dof_labels)
    for link in assembled.nonlinear_links:
        _scatter_first_harmonic_link(
            residual,
            jacobian,
            dof_count=dof_count,
            response=response,
            link=link,
        )
    return residual, jacobian


def _newton_harmonic_balance_from_start(
    linear_block_matrix: Matrix,
    rhs: list[float],
    assembled: LinearDynamicAssemblyResult,
    analysis,
    start_response: list[float],
) -> list[float]:
    response = start_response[:]
    tolerance = float(analysis.nonlinear_tolerance)
    target_norm = tolerance * max(_infinity_norm(rhs), 1.0)

    for _ in range(max(int(analysis.max_nonlinear_iterations), 1)):
        residual, jacobian = _build_harmonic_balance_residual_and_jacobian(
            linear_block_matrix,
            rhs,
            assembled,
            response,
        )
        residual_norm = _infinity_norm(residual)
        if residual_norm <= target_norm:
            return response

        correction = solve_linear_system(
            create_aij_matrix(jacobian),
            [-value for value in residual],
        )
        if _infinity_norm(correction) <= tolerance * max(_infinity_norm(response), 1.0):
            return response

        accepted_response: list[float] | None = None
        accepted_norm = residual_norm
        step_scale = 1.0
        for _line_search_index in range(8):
            trial = [
                value + (step_scale * correction[index])
                for index, value in enumerate(response)
            ]
            trial_residual, _ = _build_harmonic_balance_residual_and_jacobian(
                linear_block_matrix,
                rhs,
                assembled,
                trial,
            )
            trial_norm = _infinity_norm(trial_residual)
            if trial_norm < accepted_norm:
                accepted_response = trial
                accepted_norm = trial_norm
                break
            step_scale *= 0.5

        response = accepted_response if accepted_response is not None else [
            value + correction[index] for index, value in enumerate(response)
        ]

    final_residual, _ = _build_harmonic_balance_residual_and_jacobian(
        linear_block_matrix,
        rhs,
        assembled,
        response,
    )
    raise RuntimeError(
        "Harmonic-balance frequency response did not converge "
        f"(residual={_infinity_norm(final_residual):.6e}, "
        f"target={target_norm:.6e})."
    )


def _solve_harmonic_balance_frequency_point(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
    omega: float,
    initial_response: list[float] | None,
) -> list[float]:
    linear_block_matrix = _build_real_block_matrix(
        assembled.stiffness_matrix,
        assembled.mass_matrix,
        assembled.damping_matrix,
        omega,
    )
    rhs = _build_frequency_rhs(assembled.stiffness_matrix, assembled, excitation, omega)
    linear_response = solve_linear_system(create_aij_matrix(linear_block_matrix), rhs)

    starts: list[list[float]] = []
    if initial_response is not None:
        starts.append(initial_response)
    starts.append(linear_response)

    errors: list[str] = []
    for start_response in starts:
        try:
            return _newton_harmonic_balance_from_start(
                linear_block_matrix,
                rhs,
                assembled,
                analysis,
                start_response,
            )
        except RuntimeError as exc:
            errors.append(str(exc))

    raise RuntimeError("; ".join(errors))


def _response_point_from_complex_response(
    assembled: LinearDynamicAssemblyResult,
    *,
    frequency_hz: float,
    response: list[float],
) -> FrequencyResponsePoint:
    dof_count = len(assembled.dof_labels)
    real = response[:dof_count]
    imag = response[dof_count:]
    excitation_magnitude = (
        (real[assembled.excitation_dof] ** 2) + (imag[assembled.excitation_dof] ** 2)
    ) ** 0.5
    observation_magnitudes = [
        ((real[dof] ** 2) + (imag[dof] ** 2)) ** 0.5
        for dof in assembled.observation_dofs
    ]
    return FrequencyResponsePoint(
        frequency_hz=frequency_hz,
        excitation_response_magnitude=excitation_magnitude,
        observation_magnitudes=observation_magnitudes,
    )


def solve_frequency_response_system(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
) -> FrequencyResponseResult:
    require_petsc()

    points: list[FrequencyResponsePoint] = []
    target_frequencies = _frequency_grid(analysis)
    if assembled.nonlinear_links:
        continuation_points = solve_frequency_continuation(
            target_frequencies,
            lambda frequency_hz, initial_response: _solve_harmonic_balance_frequency_point(
                assembled,
                excitation,
                analysis,
                2.0 * pi * frequency_hz,
                initial_response,
            ),
            _relative_response_change,
        )
        return FrequencyResponseResult(
            observation_names=assembled.observation_names,
            points=[
                _response_point_from_complex_response(
                    assembled,
                    frequency_hz=point.frequency_hz,
                    response=point.state,
                )
                for point in continuation_points
            ],
        )

    for frequency_hz in target_frequencies:
        omega = 2.0 * pi * frequency_hz
        response = _solve_frequency_point(
            assembled.stiffness_matrix,
            assembled,
            excitation,
            omega,
        )
        points.append(
            _response_point_from_complex_response(
                assembled,
                frequency_hz=frequency_hz,
                response=response,
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
