"""Schedule-only Harvest Console file-format tests (no Tk window required)."""
from __future__ import annotations

import pytest

from orchard_fem.actuator.harvest_bridge import (
    HarvestSchedule,
    HarvestStage,
    plan_harvest_execution,
)
from orchard_fem.actuator.harvest_console import (
    SCHEDULE_FILE_FORMAT,
    _schedule_export_payload,
    _schedule_from_payload,
)


def _schedule() -> HarvestSchedule:
    plan = plan_harvest_execution(
        frequency_hz=5.0,
        clamp_peak_to_peak_mm=10.0,
        duration_s=6.0,
        excitation_label="primary_1@0.25",
    )
    return HarvestSchedule(
        stages=(
            HarvestStage(
                index=1,
                plan=plan,
                new_branches=("secondary_1", "secondary_2"),
                cumulative_coverage=0.5,
                trunk_stress_pa=4.2e6,
                n_detached_fruits=8,
            ),
        ),
        clamp_label="primary_1@0.25",
        target_coverage=0.95,
    )


def test_schedule_export_contains_no_single_stage_recommendation():
    payload = _schedule_export_payload(
        _schedule(),
        model_path="workspace/tree_models/tree.json",
        model_name="tree",
        calculation={"detach_cycles": 50.0},
    )

    assert payload["format"] == SCHEDULE_FILE_FORMAT
    assert "recommendation" not in payload
    assert payload["schedule"]["stages"][0]["index"] == 1

    loaded, metadata = _schedule_from_payload(payload)
    assert loaded == _schedule()
    assert metadata["model_name"] == "tree"
    assert metadata["calculation"]["detach_cycles"] == 50.0


def test_legacy_combined_file_still_loads_its_schedule():
    legacy = {
        "recommendation": {
            "model_path": "old/tree.json",
            "model_name": "old-tree",
        },
        "schedule": _schedule().to_dict(),
    }

    loaded, metadata = _schedule_from_payload(legacy)
    assert loaded == _schedule()
    assert metadata["model_path"] == "old/tree.json"
    assert metadata["model_name"] == "old-tree"


def test_single_stage_recommendation_without_schedule_is_rejected():
    with pytest.raises(ValueError, match="not a harvest schedule"):
        _schedule_from_payload({"recommended": {"frequency_hz": 5.0}})
