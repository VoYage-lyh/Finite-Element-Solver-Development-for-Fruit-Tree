from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from orchard_fem.io.csv_writer import TimeHistoryRow, write_time_history_csv
from orchard_fem.io.legacy_loader import load_orchard_model
from orchard_fem.model import ExcitationKind
from orchard_fem.solvers._petsc import create_aij_matrix, require_petsc, solve_linear_system
from orchard_fem.solvers.modal_assembler import (
    LinearDynamicAssemblyResult,
    NonlinearLinkDefinition,
    NonlinearLinkKind,
    OrchardSystemAssembler,
)


@dataclass(frozen=True)
class TimeHistoryRequest:
    model_path: str
    output_csv: str


@dataclass(frozen=True)
class TimeExcitationState:
    signal_value: float
    equivalent_load: float


@dataclass(frozen=True)
class TimeHistoryPoint:
    time_seconds: float
    excitation_signal_value: float
    excitation_load_value: float
    excitation_response_value: float
    observation_values: list[float]


@dataclass(frozen=True)
class TimeHistoryResult:
    observation_names: list[str]
    points: list[TimeHistoryPoint]

    def write_csv(self, file_path: str) -> None:
        write_time_history_csv(
            file_path,
            self.observation_names,
            [
                TimeHistoryRow(
                    time_seconds=point.time_seconds,
                    excitation_signal=point.excitation_signal_value,
                    excitation_load=point.excitation_load_value,
                    excitation_response=point.excitation_response_value,
                    observation_values=point.observation_values,
                )
                for point in self.points
            ],
        )


def _default_driving_frequency_hz(excitation, analysis) -> float:
    if excitation.driving_frequency_hz > 0.0:
        return excitation.driving_frequency_hz
    return max(analysis.frequency_start_hz, 0.1)


def _build_time_excitation_state(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
    time_seconds: float,
) -> TimeExcitationState:
    excitation_dof = assembled.excitation_dof
    phase_radians = excitation.phase_degrees * (pi / 180.0)
    omega = 2.0 * pi * _default_driving_frequency_hz(excitation, analysis)
    angle = (omega * time_seconds) + phase_radians
    displacement = excitation.amplitude * sin(angle)
    velocity = excitation.amplitude * omega * cos(angle)
    acceleration = -excitation.amplitude * omega * omega * sin(angle)

    if excitation.kind == ExcitationKind.HARMONIC_FORCE:
        return TimeExcitationState(signal_value=displacement, equivalent_load=displacement)

    if excitation.kind == ExcitationKind.HARMONIC_DISPLACEMENT:
        equivalent_load = (
            (assembled.stiffness_matrix[excitation_dof][excitation_dof] * displacement)
            + (assembled.damping_matrix[excitation_dof][excitation_dof] * velocity)
            + (assembled.mass_matrix[excitation_dof][excitation_dof] * acceleration)
        )
        return TimeExcitationState(signal_value=displacement, equivalent_load=equivalent_load)

    if excitation.kind == ExcitationKind.HARMONIC_ACCELERATION:
        equivalent_load = assembled.mass_matrix[excitation_dof][excitation_dof] * acceleration
        return TimeExcitationState(signal_value=acceleration, equivalent_load=equivalent_load)

    raise ValueError(f"Unsupported excitation kind: {excitation.kind}")


def _build_load_vector(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
    time_seconds: float,
) -> list[float]:
    load = [0.0 for _ in assembled.dof_labels]
    load[assembled.excitation_dof] = _build_time_excitation_state(
        assembled,
        excitation,
        analysis,
        time_seconds,
    ).equivalent_load
    return load


def _matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(row[column_index] * vector[column_index] for column_index in range(len(vector)))
        for row in matrix
    ]


def _infinity_norm(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def _nonlinear_force(link: NonlinearLinkDefinition, relative_displacement: float) -> float:
    if link.kind == NonlinearLinkKind.CUBIC_SPRING:
        return link.cubic_stiffness * relative_displacement * relative_displacement * relative_displacement

    if link.kind == NonlinearLinkKind.GAP_SPRING:
        magnitude = abs(relative_displacement)
        if magnitude <= link.gap_threshold:
            return 0.0
        return (
            (link.open_stiffness - link.linear_stiffness)
            * (magnitude - link.gap_threshold)
            * (1.0 if relative_displacement >= 0.0 else -1.0)
        )

    raise ValueError(f"Unsupported nonlinear link kind: {link.kind}")


def _nonlinear_tangent(link: NonlinearLinkDefinition, relative_displacement: float) -> float:
    if link.kind == NonlinearLinkKind.CUBIC_SPRING:
        return 3.0 * link.cubic_stiffness * relative_displacement * relative_displacement

    if link.kind == NonlinearLinkKind.GAP_SPRING:
        return (
            0.0
            if abs(relative_displacement) <= link.gap_threshold
            else (link.open_stiffness - link.linear_stiffness)
        )

    raise ValueError(f"Unsupported nonlinear link kind: {link.kind}")


def _evaluate_nonlinear_tangent_and_force(
    dof_count: int,
    nonlinear_links: list[NonlinearLinkDefinition],
    displacement: list[float],
) -> tuple[list[list[float]], list[float]]:
    tangent = [[0.0 for _ in range(dof_count)] for _ in range(dof_count)]
    nonlinear_force = [0.0 for _ in range(dof_count)]

    for link in nonlinear_links:
        first = link.first_dof
        second = link.second_dof
        second_value = displacement[second] if second >= 0 else 0.0
        relative_displacement = displacement[first] - second_value
        scalar_force = _nonlinear_force(link, relative_displacement)
        scalar_tangent = _nonlinear_tangent(link, relative_displacement)

        nonlinear_force[first] += scalar_force
        tangent[first][first] += scalar_tangent

        if second >= 0:
            nonlinear_force[second] -= scalar_force
            tangent[first][second] -= scalar_tangent
            tangent[second][first] -= scalar_tangent
            tangent[second][second] += scalar_tangent

    return tangent, nonlinear_force


def _build_effective_matrix(
    assembled: LinearDynamicAssemblyResult,
    nonlinear_tangent: list[list[float]],
    mass_scale: float,
    damping_scale: float,
) -> list[list[float]]:
    dof_count = len(assembled.dof_labels)
    matrix = [[0.0 for _ in range(dof_count)] for _ in range(dof_count)]

    for row_index in range(dof_count):
        for column_index in range(dof_count):
            matrix[row_index][column_index] = (
                (mass_scale * assembled.mass_matrix[row_index][column_index])
                + (damping_scale * assembled.damping_matrix[row_index][column_index])
                + assembled.stiffness_matrix[row_index][column_index]
                + nonlinear_tangent[row_index][column_index]
            )

    return matrix


def _compute_initial_acceleration(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
) -> list[float]:
    external_force = _build_load_vector(assembled, excitation, analysis, 0.0)
    _, nonlinear_force = _evaluate_nonlinear_tangent_and_force(
        len(assembled.dof_labels),
        assembled.nonlinear_links,
        [0.0 for _ in assembled.dof_labels],
    )
    rhs = [
        external_force[index] - nonlinear_force[index]
        for index in range(len(external_force))
    ]
    return solve_linear_system(create_aij_matrix(assembled.mass_matrix), rhs)


class PETScTimeHistorySolver:
    def solve(self, request: TimeHistoryRequest) -> TimeHistoryResult:
        require_petsc()

        model = load_orchard_model(request.model_path)
        assembled = OrchardSystemAssembler().assemble(model)

        if not assembled.dof_labels:
            raise RuntimeError("Dynamic system is empty")
        if model.analysis.time_step_seconds <= 0.0 or model.analysis.total_time_seconds <= 0.0:
            raise RuntimeError("Time-history analysis requires positive time step and total time")

        dof_count = len(assembled.dof_labels)
        dt = model.analysis.time_step_seconds
        total_steps = max(1, round(model.analysis.total_time_seconds / dt))
        output_stride = max(model.analysis.output_stride, 1)
        beta = 0.25
        gamma = 0.5
        mass_scale = 1.0 / (beta * dt * dt)
        damping_scale = gamma / (beta * dt)

        displacement = [0.0 for _ in range(dof_count)]
        velocity = [0.0 for _ in range(dof_count)]
        acceleration = _compute_initial_acceleration(assembled, model.excitation, model.analysis)
        initial_excitation_state = _build_time_excitation_state(
            assembled,
            model.excitation,
            model.analysis,
            0.0,
        )

        points = [
            TimeHistoryPoint(
                time_seconds=0.0,
                excitation_signal_value=initial_excitation_state.signal_value,
                excitation_load_value=initial_excitation_state.equivalent_load,
                excitation_response_value=0.0,
                observation_values=[0.0 for _ in assembled.observation_dofs],
            )
        ]

        max_iterations = max(model.analysis.max_nonlinear_iterations, 1)
        tolerance = model.analysis.nonlinear_tolerance

        for step in range(1, total_steps + 1):
            time_seconds = float(step) * dt
            displacement_predictor = [
                displacement[index]
                + (dt * velocity[index])
                + (dt * dt * (0.5 - beta) * acceleration[index])
                for index in range(dof_count)
            ]
            velocity_predictor = [
                velocity[index] + (dt * (1.0 - gamma) * acceleration[index])
                for index in range(dof_count)
            ]

            displacement_guess = displacement_predictor.copy()
            converged = False

            for _ in range(max_iterations):
                acceleration_guess = [
                    mass_scale * (displacement_guess[index] - displacement_predictor[index])
                    for index in range(dof_count)
                ]
                velocity_guess = [
                    velocity_predictor[index] + (gamma * dt * acceleration_guess[index])
                    for index in range(dof_count)
                ]

                nonlinear_tangent, nonlinear_force = _evaluate_nonlinear_tangent_and_force(
                    dof_count,
                    assembled.nonlinear_links,
                    displacement_guess,
                )
                external_force = _build_load_vector(
                    assembled,
                    model.excitation,
                    model.analysis,
                    time_seconds,
                )
                mass_force = _matrix_vector_multiply(assembled.mass_matrix, acceleration_guess)
                damping_force = _matrix_vector_multiply(assembled.damping_matrix, velocity_guess)
                stiffness_force = _matrix_vector_multiply(assembled.stiffness_matrix, displacement_guess)

                residual = [
                    mass_force[index]
                    + damping_force[index]
                    + stiffness_force[index]
                    + nonlinear_force[index]
                    - external_force[index]
                    for index in range(dof_count)
                ]
                if _infinity_norm(residual) < tolerance:
                    converged = True
                    break

                effective_matrix = _build_effective_matrix(
                    assembled,
                    nonlinear_tangent,
                    mass_scale,
                    damping_scale,
                )
                displacement_increment = solve_linear_system(
                    create_aij_matrix(effective_matrix),
                    [-value for value in residual],
                )

                displacement_guess = [
                    displacement_guess[index] + displacement_increment[index]
                    for index in range(dof_count)
                ]

                increment_norm = _infinity_norm(displacement_increment)
                state_norm = max(_infinity_norm(displacement_guess), 1.0)
                if increment_norm < (tolerance * state_norm):
                    converged = True
                    break

            if not converged:
                raise RuntimeError(
                    "Newmark nonlinear iteration failed to converge at time "
                    f"{time_seconds:.12g}"
                )

            acceleration = [
                mass_scale * (displacement_guess[index] - displacement_predictor[index])
                for index in range(dof_count)
            ]
            velocity = [
                velocity_predictor[index] + (gamma * dt * acceleration[index])
                for index in range(dof_count)
            ]
            displacement = displacement_guess

            if step % output_stride == 0 or step == total_steps:
                excitation_state = _build_time_excitation_state(
                    assembled,
                    model.excitation,
                    model.analysis,
                    time_seconds,
                )
                points.append(
                    TimeHistoryPoint(
                        time_seconds=time_seconds,
                        excitation_signal_value=excitation_state.signal_value,
                        excitation_load_value=excitation_state.equivalent_load,
                        excitation_response_value=displacement[assembled.excitation_dof],
                        observation_values=[
                            displacement[dof] for dof in assembled.observation_dofs
                        ],
                    )
                )

        result = TimeHistoryResult(
            observation_names=assembled.observation_names,
            points=points,
        )
        result.write_csv(request.output_csv)
        return result
