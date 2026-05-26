from __future__ import annotations

from orchard_fem.discretization import NonlinearLinkDefinition, NonlinearLinkKind


def matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(row[column_index] * vector[column_index] for column_index in range(len(vector)))
        for row in matrix
    ]


def infinity_norm(values: list[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def nonlinear_force(link: NonlinearLinkDefinition, relative_displacement: float) -> float:
    if link.kind == NonlinearLinkKind.CUBIC_SPRING:
        return (
            link.cubic_stiffness
            * relative_displacement
            * relative_displacement
            * relative_displacement
        )

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


def nonlinear_tangent(link: NonlinearLinkDefinition, relative_displacement: float) -> float:
    if link.kind == NonlinearLinkKind.CUBIC_SPRING:
        return 3.0 * link.cubic_stiffness * relative_displacement * relative_displacement

    if link.kind == NonlinearLinkKind.GAP_SPRING:
        return (
            0.0
            if abs(relative_displacement) <= link.gap_threshold
            else (link.open_stiffness - link.linear_stiffness)
        )

    raise ValueError(f"Unsupported nonlinear link kind: {link.kind}")


def quadratic_damping_force(
    link: NonlinearLinkDefinition, relative_velocity: float
) -> float:
    coefficient = link.quadratic_damping
    if coefficient == 0.0:
        return 0.0
    return coefficient * abs(relative_velocity) * relative_velocity


def quadratic_damping_tangent(
    link: NonlinearLinkDefinition, relative_velocity: float
) -> float:
    coefficient = link.quadratic_damping
    if coefficient == 0.0:
        return 0.0
    return 2.0 * coefficient * abs(relative_velocity)


def evaluate_nonlinear_tangent_and_force(
    dof_count: int,
    nonlinear_links: list[NonlinearLinkDefinition],
    displacement: list[float],
) -> tuple[list[list[float]], list[float]]:
    stiffness_tangent, _, force = evaluate_nonlinear_force_and_tangents(
        dof_count, nonlinear_links, displacement, None
    )
    return stiffness_tangent, force


def evaluate_nonlinear_force_and_tangents(
    dof_count: int,
    nonlinear_links: list[NonlinearLinkDefinition],
    displacement: list[float],
    velocity: list[float] | None,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    stiffness_tangent = [[0.0 for _ in range(dof_count)] for _ in range(dof_count)]
    damping_tangent = [[0.0 for _ in range(dof_count)] for _ in range(dof_count)]
    force = [0.0 for _ in range(dof_count)]

    for link in nonlinear_links:
        first = link.first_dof
        second = link.second_dof
        second_disp = displacement[second] if second >= 0 else 0.0
        relative_displacement = displacement[first] - second_disp
        elastic_force = nonlinear_force(link, relative_displacement)
        elastic_tangent = nonlinear_tangent(link, relative_displacement)

        damping_contribution = 0.0
        damping_jacobian = 0.0
        if velocity is not None and link.quadratic_damping != 0.0:
            second_vel = velocity[second] if second >= 0 else 0.0
            relative_velocity = velocity[first] - second_vel
            damping_contribution = quadratic_damping_force(link, relative_velocity)
            damping_jacobian = quadratic_damping_tangent(link, relative_velocity)

        total_force = elastic_force + damping_contribution
        force[first] += total_force
        stiffness_tangent[first][first] += elastic_tangent
        damping_tangent[first][first] += damping_jacobian

        if second >= 0:
            force[second] -= total_force
            stiffness_tangent[first][second] -= elastic_tangent
            stiffness_tangent[second][first] -= elastic_tangent
            stiffness_tangent[second][second] += elastic_tangent
            damping_tangent[first][second] -= damping_jacobian
            damping_tangent[second][first] -= damping_jacobian
            damping_tangent[second][second] += damping_jacobian

    return stiffness_tangent, damping_tangent, force
