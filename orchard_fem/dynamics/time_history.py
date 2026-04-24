from __future__ import annotations

from dataclasses import dataclass

from orchard_fem.discretization import LinearDynamicAssemblyResult, OrchardSystemAssembler
from orchard_fem.dynamics.excitation import (
    TimeExcitationState,
    build_time_excitation_state,
    build_time_load_vector,
)
from orchard_fem.dynamics.nonlinear import (
    evaluate_nonlinear_tangent_and_force,
    infinity_norm,
    matrix_vector_multiply,
)
from orchard_fem.io import load_orchard_model
from orchard_fem.io.csv_writer import TimeHistoryRow, write_time_history_csv
from orchard_fem.numerics import create_aij_matrix, require_petsc, solve_linear_system


@dataclass(frozen=True)
class TimeHistoryRequest:
    model_path: str
    output_csv: str


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
    external_force = build_time_load_vector(assembled, excitation, analysis, 0.0)
    _, nonlinear_force = evaluate_nonlinear_tangent_and_force(
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
        model = load_orchard_model(request.model_path)
        assembled = OrchardSystemAssembler().assemble(model)
        result = solve_time_history_system(assembled, model.excitation, model.analysis)
        result.write_csv(request.output_csv)
        return result


def solve_time_history_system(
    assembled: LinearDynamicAssemblyResult,
    excitation,
    analysis,
) -> TimeHistoryResult:
    require_petsc()

    if not assembled.dof_labels:
        raise RuntimeError("Dynamic system is empty")
    if analysis.time_step_seconds <= 0.0 or analysis.total_time_seconds <= 0.0:
        raise RuntimeError("Time-history analysis requires positive time step and total time")

    dof_count = len(assembled.dof_labels)
    dt = analysis.time_step_seconds
    total_steps = max(1, round(analysis.total_time_seconds / dt))
    output_stride = max(analysis.output_stride, 1)
    beta = 0.25
    gamma = 0.5
    mass_scale = 1.0 / (beta * dt * dt)
    damping_scale = gamma / (beta * dt)

    displacement = [0.0 for _ in range(dof_count)]
    velocity = [0.0 for _ in range(dof_count)]
    acceleration = _compute_initial_acceleration(assembled, excitation, analysis)
    initial_excitation_state = build_time_excitation_state(
        assembled,
        excitation,
        analysis,
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

    max_iterations = max(analysis.max_nonlinear_iterations, 1)
    tolerance = analysis.nonlinear_tolerance

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

            nonlinear_tangent, nonlinear_force = evaluate_nonlinear_tangent_and_force(
                dof_count,
                assembled.nonlinear_links,
                displacement_guess,
            )
            external_force = build_time_load_vector(
                assembled,
                excitation,
                analysis,
                time_seconds,
            )
            mass_force = matrix_vector_multiply(assembled.mass_matrix, acceleration_guess)
            damping_force = matrix_vector_multiply(assembled.damping_matrix, velocity_guess)
            stiffness_force = matrix_vector_multiply(assembled.stiffness_matrix, displacement_guess)

            residual = [
                mass_force[index]
                + damping_force[index]
                + stiffness_force[index]
                + nonlinear_force[index]
                - external_force[index]
                for index in range(dof_count)
            ]
            if infinity_norm(residual) < tolerance:
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

            increment_norm = infinity_norm(displacement_increment)
            state_norm = max(infinity_norm(displacement_guess), 1.0)
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
            excitation_state = build_time_excitation_state(
                assembled,
                excitation,
                analysis,
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

    return TimeHistoryResult(
        observation_names=assembled.observation_names,
        points=points,
    )
