"""Repository-wide pytest classification.

Directory markers keep test selection consistent without repeating ``pytestmark``
in every module.  Expensive or optional stacks receive additional markers.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.path))
        parts = path.parts
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "verification" in parts:
            item.add_marker(pytest.mark.verification)

        if "backend" in parts and "fenicsx" in parts:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.fenicsx)
        if path.name == "test_wood_seg.py":
            item.add_marker(pytest.mark.ml)
        if path.name == "test_uq_stack.py":
            item.add_marker(pytest.mark.uq)
        if item.name in {
            "test_compute_basin_ccm_runs_and_is_sane",
            "test_petsc_time_history_solver_writes_demo_csv",
        }:
            item.add_marker(pytest.mark.slow)
