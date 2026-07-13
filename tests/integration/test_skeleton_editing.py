from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("skimage")

from skimage.io import imsave

from orchard_vision.lift_3d import MonocularPlanarLift
from orchard_vision.skeleton_editing import EditableSkeleton, SkeletonEditError
from orchard_vision.types import Branch


def _line(first: tuple[int, int], second: tuple[int, int]) -> np.ndarray:
    count = max(abs(second[0] - first[0]), abs(second[1] - first[1])) + 1
    rows = np.rint(np.linspace(first[0], second[0], count)).astype(int)
    cols = np.rint(np.linspace(first[1], second[1], count)).astype(int)
    return np.column_stack([rows, cols])


def _document(*, with_visible_left_branch: bool = False) -> EditableSkeleton:
    image = np.full((80, 90, 3), 255, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[18:73, 37:44] = True  # trunk
    mask[35:42, 40:67] = True  # primary
    mask[16:39, 62:69] = True  # secondary
    image[mask] = 45
    if with_visible_left_branch:
        image[47:54, 12:41] = 35  # visible in RGB, deliberately absent from mask

    trunk = _line((70, 40), (20, 40))
    primary = _line((38, 40), (38, 65))
    secondary = _line((38, 65), (18, 65))
    branches = [
        Branch("trunk", None, 0, trunk, np.ones(len(trunk))),
        Branch("primary_1", "trunk", 1, primary, np.ones(len(primary))),
        Branch("secondary_1", "primary_1", 2, secondary, np.ones(len(secondary))),
    ]
    return EditableSkeleton(
        image=image,
        mask=mask,
        branches=branches,
        root_rc=(70, 40),
        lift=MonocularPlanarLift(0.01, 70, 40),
        model_name="synthetic_tree",
    )


def test_reparent_recomputes_level_and_snaps_root_to_new_parent() -> None:
    document = _document()
    # Promoting this branch changes its attachment from the primary tip to the
    # trunk. Supply the visible connecting limb that justifies that geometry.
    document.paint_mask(
        [tuple(point) for point in _line((38, 40), (18, 65))], radius_px=2, value=True
    )
    document.finish_mask_edit()

    document.set_parent("secondary_1", "trunk")

    branch = document.branch("secondary_1")
    assert branch.parent_id == "trunk"
    assert branch.level == 1
    assert tuple(branch.pixels[0]) in {tuple(point) for point in document.branch("trunk").pixels}
    assert document.validation_errors() == []


def test_user_branch_requires_direct_wood_mask_support() -> None:
    document = _document()

    with pytest.raises(SkeletonEditError, match="direct wood-mask support"):
        document.add_branch("trunk", np.asarray([[50, 40], [50, 82]], dtype=float))


def test_image_guided_mask_proposal_can_back_a_real_new_branch() -> None:
    document = _document(with_visible_left_branch=True)
    controls = np.asarray([[50, 40], [50, 13]], dtype=float)

    proposal = document.suggest_branch_mask(controls)

    assert proposal[50, 20]
    assert not proposal[10, 10]
    document.merge_mask(proposal)
    branch_id = document.add_branch("trunk", controls)
    assert document.sources[branch_id] == "user_added"
    assert document.branch(branch_id).level == 1
    assert document.evidence(branch_id).support_ratio >= document.minimum_support


def test_control_point_edit_changes_derived_length() -> None:
    document = _document()
    before = document.length_m("primary_1")
    controls = document.controls["primary_1"].copy()
    controls[-1] = [38, 55]

    document.update_branch("primary_1", controls)

    assert document.length_m("primary_1") < before
    assert document.sources["primary_1"] == "user_edited"


def test_project_round_trip_preserves_pixel_geometry_and_mask(tmp_path) -> None:
    document = _document(with_visible_left_branch=True)
    image_path = tmp_path / "tree.png"
    # The source photo can be larger than the pipeline's working-resolution image.
    imsave(image_path, np.repeat(np.repeat(document.image, 2, axis=0), 2, axis=1), check_contrast=False)
    document.image_path = image_path
    project_path = tmp_path / "tree.skeleton-project.json"

    document.save_project(project_path)
    restored = EditableSkeleton.load_project(project_path)

    assert np.array_equal(restored.mask, document.mask)
    assert np.array_equal(restored.image, document.image)
    assert [branch.id for branch in restored.branches] == [branch.id for branch in document.branches]
    assert np.array_equal(restored.branch("primary_1").pixels, document.branch("primary_1").pixels)
    assert restored.lift.meters_per_pixel == pytest.approx(0.01)
    assert json.loads(project_path.read_text(encoding="utf-8"))["version"] == 1


def test_export_includes_edit_provenance_and_transform() -> None:
    document = _document()

    payload = document.to_skeleton_payload()

    assert payload["metadata"]["edited"] is True
    assert payload["metadata"]["vision_transform"]["meters_per_pixel"] == pytest.approx(0.01)
    assert payload["branches"][0]["image_support_ratio"] == pytest.approx(1.0)
    assert payload["branches"][0]["edit_source"] == "automatic"
