from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from orchard_vision.branch_ordering import order_branches
from orchard_vision.pipeline import isolate_target_mask, root_aware_probability_mask
from orchard_vision.types import GraphEdge, SkeletonGraph


def test_target_mask_defaults_to_largest_component() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[4:25, 10:14] = True
    mask[26:29, 35:38] = True  # lower false positive that used to win root selection

    selected = isolate_target_mask(mask)

    assert selected[10, 11]
    assert not selected[27, 36]
    assert selected.sum() == 84


def test_trunk_hint_selects_its_component_inside_roi() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[4:25, 10:14] = True
    mask[10:18, 30:33] = True

    selected = isolate_target_mask(
        mask,
        target_roi_xyxy=(20, 2, 38, 24),
        trunk_root_rc=(17, 31),
    )

    assert selected[12, 31]
    assert not selected[10, 11]


def test_trunk_hint_rejects_detection_too_far_from_base() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[2:5, 2:5] = True

    with pytest.raises(ValueError, match="no wood near"):
        isolate_target_mask(mask, trunk_root_rc=(45, 45), root_snap_px=5.0)


def test_tiny_speck_under_root_does_not_replace_nearby_tree() -> None:
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:55, 25:31] = True
    mask[54, 34] = True

    selected = isolate_target_mask(mask, trunk_root_rc=(54, 34), root_snap_px=8.0)

    assert selected[40, 28]
    assert not selected[54, 34]


def test_probability_threshold_relaxes_until_trunk_component_appears() -> None:
    probability = np.zeros((60, 60), dtype=np.float32)
    probability[10:55, 5:11] = 0.36
    probability[2:10, 50:58] = 0.95

    selected, threshold = root_aware_probability_mask(
        probability,
        preferred_threshold=0.70,
        target_roi_xyxy=(5, 2, 58, 58),
        trunk_root_rc=(53, 8),
    )

    assert threshold == pytest.approx(0.3)
    assert selected[40, 8]
    assert not selected[5, 54]


def test_probability_threshold_reports_true_total_miss() -> None:
    with pytest.raises(ValueError, match="even after lowering confidence"):
        root_aware_probability_mask(
            np.zeros((30, 30), dtype=np.float32),
            preferred_threshold=0.70,
            target_roi_xyxy=(0, 0, 30, 30),
            trunk_root_rc=(25, 15),
        )


def test_branch_ordering_uses_root_hint_instead_of_lowest_false_component() -> None:
    nodes = np.asarray([[20, 10], [5, 10], [27, 35], [25, 35]])
    graph = SkeletonGraph(
        nodes=nodes,
        node_kind=["endpoint"] * 4,
        edges=[
            GraphEdge(
                0,
                1,
                np.asarray([[20, 10], [12, 10], [5, 10]]),
                np.ones(3),
            ),
            GraphEdge(
                2,
                3,
                np.asarray([[27, 35], [26, 35], [25, 35]]),
                np.ones(3),
            ),
        ],
    )

    branches, root = order_branches(graph, root_hint_rc=(19, 10), max_level=0)

    assert root == (20, 10)
    assert tuple(branches[0].pixels[-1]) == (5, 10)
