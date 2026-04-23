from __future__ import annotations

from typing import Dict, Sequence, Tuple


def lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> list[float]:
    return [a[index] + alpha * (b[index] - a[index]) for index in range(3)]


def branch_num_elements(branch: dict) -> int:
    return max(int(branch.get("discretization", {}).get("num_elements", 4)), 1)


def resolve_branch_station(branch: dict, target_node) -> float:
    if target_node in (None, "tip"):
        return 1.0
    if target_node == "root":
        return 0.0

    node_index = int(target_node)
    return max(0.0, min(1.0, node_index / branch_num_elements(branch)))


def resolve_branch_point(branch: dict, station: float) -> list[float]:
    return lerp(branch["start"], branch["end"], station)


def build_branch_lookup(model: dict) -> Dict[str, dict]:
    return {branch["id"]: branch for branch in model.get("branches", [])}


def build_fruit_lookup(model: dict) -> Dict[str, dict]:
    return {fruit["id"]: fruit for fruit in model.get("fruits", [])}


def observation_components(observation: dict) -> list[str]:
    if "target_components" in observation:
        value = observation["target_components"]
        if not isinstance(value, list):
            raise RuntimeError("observations[].target_components must be a list")
        components = [str(component) for component in value]
        if not components:
            raise RuntimeError("observations[].target_components must not be empty")
        return components

    return [str(observation.get("target_component", "ux"))]


def resolve_observation_point(model: dict, observation: dict) -> Tuple[list[float], str]:
    branch_lookup = build_branch_lookup(model)
    fruit_lookup = build_fruit_lookup(model)

    if observation["target_type"] == "branch":
        branch = branch_lookup[observation["target_id"]]
        station = resolve_branch_station(branch, observation.get("target_node", "tip"))
        position = resolve_branch_point(branch, station)
        label = "{0} ({1})".format(observation["id"], "/".join(observation_components(observation)))
        return position, label

    if observation["target_type"] == "fruit":
        fruit = fruit_lookup[observation["target_id"]]
        branch = branch_lookup[fruit["branch_id"]]
        position = resolve_branch_point(branch, float(fruit["location_s"]))
        return position, observation["id"]

    raise RuntimeError(
        "Unsupported observation target_type: {0}".format(observation["target_type"])
    )


def resolve_excitation_point(model: dict) -> Tuple[list[float], str]:
    excitation = model["excitation"]
    branch = build_branch_lookup(model)[excitation["target_branch_id"]]
    station = resolve_branch_station(branch, excitation.get("target_node", "tip"))
    point = resolve_branch_point(branch, station)
    label = "excitation ({0})".format(excitation.get("target_component", "ux"))
    return point, label


def project_xz(point: Sequence[float]) -> Tuple[float, float]:
    return float(point[0]), float(point[2])
