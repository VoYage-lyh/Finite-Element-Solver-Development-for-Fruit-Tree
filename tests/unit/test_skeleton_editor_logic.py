from __future__ import annotations

from types import SimpleNamespace

from orchard_fem.actuator.skeleton_editor import (
    _available_add_levels,
    _level_option,
    _parent_ids_for_level,
)


def _branch(branch_id: str, level: int):
    return SimpleNamespace(id=branch_id, level=level)


def test_add_level_choices_include_one_new_depth() -> None:
    branches = [
        _branch("trunk", 0),
        _branch("primary_1", 1),
        _branch("secondary_1", 2),
    ]

    assert _available_add_levels(branches) == (1, 2, 3)
    assert _level_option(1) == "Primary (L1)"
    assert _level_option(3) == "Tertiary (L3)"


def test_parent_choices_are_filtered_by_requested_level() -> None:
    branches = [
        _branch("trunk", 0),
        _branch("primary_1", 1),
        _branch("primary_2", 1),
        _branch("secondary_1", 2),
    ]

    assert _parent_ids_for_level(branches, 1) == ("trunk",)
    assert _parent_ids_for_level(branches, 2) == ("primary_1", "primary_2")
    assert _parent_ids_for_level(branches, 3) == ("secondary_1",)
    assert _parent_ids_for_level(branches, 0) == ()
