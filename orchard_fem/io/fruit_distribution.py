from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from orchard_fem.domain.entities import (
        FruitAttachment,
        FruitDistributionPolicy,
        OrchardModel,
    )

Position = Literal["Root", "Mid", "Tip"]

POSITION_XI: dict[Position, float] = {
    "Root": 0.0,
    "Mid": 0.5,
    "Tip": 1.0,
}


# ── Input / output dataclasses ───────────────────────────────────────────────

@dataclass
class _BranchFruitSpec:
    branch_id: str
    level: int
    length_m: float
    is_terminal: bool


@dataclass
class FruitInstance:
    fruit_id: str
    branch_id: str
    branch_level: int
    node_id: str
    position: Position
    xi: float
    crack_state: int
    long_axis_mm: float
    short_axis_mm: float
    canopy_height_mm: float
    mass_g: float
    mass_kg: float
    detachment_force_N: float


@dataclass
class FruitNodeSummary:
    branch_id: str
    branch_level: int
    node_id: str
    position: Position
    fruit_count: int
    crack_count: int
    total_fruit_mass_kg: float
    mean_single_fruit_mass_g: float
    mean_detachment_force_N: float
    mean_long_axis_mm: float
    mean_short_axis_mm: float
    mean_canopy_height_mm: float


# ── Regression mean functions (fitted from empirical 20-sample table) ────────

def _mean_long_axis_mm(xi: float) -> float:
    return 46.67 - 6.28 * xi + 1.39 * xi ** 2


def _mean_short_axis_mm(xi: float) -> float:
    return 33.00 + 6.50 * xi - 6.50 * xi ** 2


def _mean_fruit_mass_g(xi: float) -> float:
    return 27.07 + 2.82 * xi - 6.36 * xi ** 2


def _mean_canopy_height_mm(xi: float) -> float:
    return 1426.67 + 199.44 * xi + 189.44 * xi ** 2


def _mean_detachment_force_N(xi: float, crack_state: int) -> float:
    log_force = 3.5179 - 0.5044 * xi - 0.8697 * xi ** 2 + 0.1859 * crack_state
    return math.exp(log_force)


# ── Random helpers ───────────────────────────────────────────────────────────

def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _lognormal_multiplier(rng: random.Random, cv: float) -> float:
    if cv <= 0.0:
        return 1.0
    sigma = math.sqrt(math.log(1.0 + cv ** 2))
    mu = -0.5 * sigma ** 2
    return rng.lognormvariate(mu, sigma)


def _normal_positive(
    rng: random.Random,
    mean: float,
    cv: float,
    lower: float,
    upper: float,
) -> float:
    sd = abs(mean * cv)
    return _clamp(rng.gauss(mean, sd), lower, upper)


# ── Fruit-bearing eligibility and count allocation ───────────────────────────

def _is_fruit_bearing(spec: _BranchFruitSpec, include_terminal_primary: bool) -> bool:
    if not spec.is_terminal:
        return False
    if spec.level >= 2:
        return True
    return include_terminal_primary and spec.level == 1


def _node_weight(
    spec: _BranchFruitSpec,
    position: Position,
    rng: random.Random,
    *,
    beta_position: float,
    alpha_branch_order: float,
    length_power: float,
    count_weight_cv: float,
) -> float:
    xi = POSITION_XI[position]
    length_factor = max(spec.length_m, 1.0e-6) ** length_power
    order_factor = math.exp(alpha_branch_order * max(spec.level - 2, 0))
    position_factor = math.exp(beta_position * xi)
    return length_factor * order_factor * position_factor * _lognormal_multiplier(rng, count_weight_cv)


def _allocate_integer_counts(
    total: int,
    weights: dict[str, float],
) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        raise ValueError("weights cannot be empty")

    weight_sum = sum(max(w, 0.0) for w in weights.values())
    if weight_sum <= 0.0:
        equal = {k: 1.0 for k in weights}
        return _allocate_integer_counts(total, equal)

    raw = {k: total * max(w, 0.0) / weight_sum for k, w in weights.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = total - sum(counts.values())
    for key in sorted(raw, key=lambda k: raw[k] - math.floor(raw[k]), reverse=True)[:remaining]:
        counts[key] += 1
    return counts


# ── Individual fruit generation ──────────────────────────────────────────────

def _generate_one_fruit(
    fruit_id: str,
    spec: _BranchFruitSpec,
    position: Position,
    rng: random.Random,
    *,
    crack_probability: float,
    long_axis_cv: float,
    short_axis_cv: float,
    mass_residual_cv: float,
    detachment_force_cv: float,
    height_cv: float = 0.05,
    min_long_axis_mm: float = 25.0,
    max_long_axis_mm: float = 65.0,
    min_short_axis_mm: float = 18.0,
    max_short_axis_mm: float = 50.0,
    min_mass_g: float = 8.0,
    max_mass_g: float = 60.0,
    min_detachment_force_N: float = 3.0,
    max_detachment_force_N: float = 70.0,
    min_height_mm: float = 1000.0,
    max_height_mm: float = 2300.0,
) -> FruitInstance:
    xi = POSITION_XI[position]
    node_id = f"{spec.branch_id}-{position}"
    crack_state = 1 if rng.random() < crack_probability else 0

    height_mm = _normal_positive(
        rng,
        mean=_mean_canopy_height_mm(xi),
        cv=height_cv,
        lower=min_height_mm,
        upper=max_height_mm,
    )

    long_mean = _mean_long_axis_mm(xi)
    short_mean = _mean_short_axis_mm(xi)

    long_axis = _clamp(
        long_mean * _lognormal_multiplier(rng, long_axis_cv),
        min_long_axis_mm,
        max_long_axis_mm,
    )
    short_axis = _clamp(
        min(short_mean * _lognormal_multiplier(rng, short_axis_cv), 0.95 * long_axis),
        min_short_axis_mm,
        max_short_axis_mm,
    )

    mass_mean = _mean_fruit_mass_g(xi)
    mass_g = _clamp(
        mass_mean
        * (long_axis / long_mean)
        * (short_axis / short_mean) ** 2
        * _lognormal_multiplier(rng, mass_residual_cv),
        min_mass_g,
        max_mass_g,
    )

    force_base = _mean_detachment_force_N(xi, crack_state)
    force_N = _clamp(
        force_base
        * (mass_g / mass_mean) ** 0.25
        * (long_axis / long_mean) ** 0.15
        * (short_axis / short_mean) ** 0.15
        * _lognormal_multiplier(rng, detachment_force_cv),
        min_detachment_force_N,
        max_detachment_force_N,
    )

    return FruitInstance(
        fruit_id=fruit_id,
        branch_id=spec.branch_id,
        branch_level=spec.level,
        node_id=node_id,
        position=position,
        xi=xi,
        crack_state=crack_state,
        long_axis_mm=long_axis,
        short_axis_mm=short_axis,
        canopy_height_mm=height_mm,
        mass_g=mass_g,
        mass_kg=mass_g / 1000.0,
        detachment_force_N=force_N,
    )


# ── Node-level aggregation ───────────────────────────────────────────────────

def _summarize_nodes(fruits: list[FruitInstance]) -> list[FruitNodeSummary]:
    grouped: dict[str, list[FruitInstance]] = {}
    for fruit in fruits:
        grouped.setdefault(fruit.node_id, []).append(fruit)

    summaries: list[FruitNodeSummary] = []
    for node_id, group in grouped.items():
        count = len(group)
        summaries.append(
            FruitNodeSummary(
                branch_id=group[0].branch_id,
                branch_level=group[0].branch_level,
                node_id=node_id,
                position=group[0].position,
                fruit_count=count,
                crack_count=sum(f.crack_state for f in group),
                total_fruit_mass_kg=sum(f.mass_kg for f in group),
                mean_single_fruit_mass_g=sum(f.mass_g for f in group) / count,
                mean_detachment_force_N=sum(f.detachment_force_N for f in group) / count,
                mean_long_axis_mm=sum(f.long_axis_mm for f in group) / count,
                mean_short_axis_mm=sum(f.short_axis_mm for f in group) / count,
                mean_canopy_height_mm=sum(f.canopy_height_mm for f in group) / count,
            )
        )
    summaries.sort(key=lambda x: x.node_id)
    return summaries


# ── Main generation entry points ─────────────────────────────────────────────

def generate_fruit_parameters(
    specs: list[_BranchFruitSpec],
    total_fruit_count: int,
    *,
    seed: int = 2026,
    include_terminal_primary: bool = True,
    count_weight_cv: float = 0.25,
    long_axis_cv: float = 0.12,
    short_axis_cv: float = 0.12,
    mass_residual_cv: float = 0.12,
    detachment_force_cv: float = 0.15,
    crack_probability: float = 0.65,
    beta_position: float = math.log(4.0),
    alpha_branch_order: float = math.log(1.25),
    length_power: float = 1.0,
) -> tuple[list[FruitInstance], list[FruitNodeSummary]]:
    if total_fruit_count <= 0:
        raise ValueError("total_fruit_count must be positive")

    rng = random.Random(seed)
    bearing_specs = [s for s in specs if _is_fruit_bearing(s, include_terminal_primary)]
    if not bearing_specs:
        raise ValueError("No fruit-bearing branches found in the provided specs")

    node_weights: dict[str, float] = {}
    node_info: dict[str, tuple[_BranchFruitSpec, Position]] = {}
    for spec in bearing_specs:
        for position in ("Root", "Mid", "Tip"):
            key = f"{spec.branch_id}-{position}"
            node_weights[key] = _node_weight(
                spec,
                position,
                rng,
                beta_position=beta_position,
                alpha_branch_order=alpha_branch_order,
                length_power=length_power,
                count_weight_cv=count_weight_cv,
            )
            node_info[key] = (spec, position)

    node_counts = _allocate_integer_counts(total_fruit_count, node_weights)

    fruits: list[FruitInstance] = []
    fruit_index = 1
    for node_key, count in node_counts.items():
        spec, position = node_info[node_key]
        for _ in range(count):
            fruits.append(
                _generate_one_fruit(
                    f"F{fruit_index:05d}",
                    spec,
                    position,
                    rng,
                    crack_probability=crack_probability,
                    long_axis_cv=long_axis_cv,
                    short_axis_cv=short_axis_cv,
                    mass_residual_cv=mass_residual_cv,
                    detachment_force_cv=detachment_force_cv,
                )
            )
            fruit_index += 1

    return fruits, _summarize_nodes(fruits)


def generate_fruit_attachments_for_model(
    model: "OrchardModel",
    policy: "FruitDistributionPolicy",
) -> list["FruitAttachment"]:
    """Convert FruitDistributionPolicy + OrchardModel into FruitAttachment list.

    Workflow:
    1. Build _BranchFruitSpec list from model branches.
       is_terminal is inferred: a branch is terminal when no other branch
       lists it as its parent_branch_id.
    2. Run the regression-based generation to get FruitNodeSummary per node.
    3. Convert each summary to one FruitAttachment (a node lumps ``fruit_count``
       fruits, so stiffness / breaking force are the parallel-spring sums):
         location_s   = POSITION_XI[position]
         mass         = total_fruit_mass_kg
         stiffness    = fruit_count * pedicel_stiffness(mass_per_fruit, ...)
         detach_force = fruit_count * mean_detachment_force_N
         damping      = 2 * zeta * sqrt(mass * stiffness)
    """
    from orchard_fem.domain.entities import FruitAttachment
    from orchard_fem.domain.pedicel import pedicel_stiffness_n_per_m

    parent_ids = {
        b.parent_branch_id
        for b in model.branches
        if b.parent_branch_id is not None
    }
    specs = [
        _BranchFruitSpec(
            branch_id=b.branch_id,
            level=b.level,
            length_m=b.path.length(),
            is_terminal=b.branch_id not in parent_ids,
        )
        for b in model.branches
    ]

    _, node_summaries = generate_fruit_parameters(
        specs,
        total_fruit_count=policy.total_fruit_count,
        seed=policy.seed,
        include_terminal_primary=policy.include_terminal_primary,
        count_weight_cv=policy.count_weight_cv,
        long_axis_cv=policy.long_axis_cv,
        short_axis_cv=policy.short_axis_cv,
        mass_residual_cv=policy.mass_residual_cv,
        detachment_force_cv=policy.detachment_force_cv,
        crack_probability=policy.crack_probability,
    )

    attachments: list[FruitAttachment] = []
    for summary in node_summaries:
        if summary.total_fruit_mass_kg <= 0.0:
            continue
        n = max(summary.fruit_count, 1)
        mass_per_fruit = summary.total_fruit_mass_kg / n
        total_stiffness = n * pedicel_stiffness_n_per_m(
            mass_per_fruit,
            policy.pedicel_length_m,
            policy.pedicel_diameter_m,
            policy.pedicel_youngs_modulus_pa,
        )
        total_detach_force = n * summary.mean_detachment_force_N
        total_damping = (
            2.0
            * policy.attachment_damping_ratio
            * math.sqrt(summary.total_fruit_mass_kg * total_stiffness)
        )
        attachments.append(
            FruitAttachment(
                fruit_id=summary.node_id,
                branch_id=summary.branch_id,
                location_s=POSITION_XI[summary.position],
                mass=summary.total_fruit_mass_kg,
                stiffness=total_stiffness,
                damping=total_damping,
                target_component=policy.attachment_component,
                detach_force=total_detach_force,
            )
        )
    return attachments
