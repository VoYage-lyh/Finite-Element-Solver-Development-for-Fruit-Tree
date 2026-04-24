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


def evaluate_nonlinear_tangent_and_force(
    dof_count: int,
    nonlinear_links: list[NonlinearLinkDefinition],
    displacement: list[float],
) -> tuple[list[list[float]], list[float]]:
    tangent = [[0.0 for _ in range(dof_count)] for _ in range(dof_count)]
    force = [0.0 for _ in range(dof_count)]

    for link in nonlinear_links:
        first = link.first_dof
        second = link.second_dof
        second_value = displacement[second] if second >= 0 else 0.0
        relative_displacement = displacement[first] - second_value
        scalar_force = nonlinear_force(link, relative_displacement)
        scalar_tangent = nonlinear_tangent(link, relative_displacement)

        force[first] += scalar_force
        tangent[first][first] += scalar_tangent

        if second >= 0:
            force[second] -= scalar_force
            tangent[first][second] -= scalar_tangent
            tangent[second][first] -= scalar_tangent
            tangent[second][second] += scalar_tangent

    return tangent, force
