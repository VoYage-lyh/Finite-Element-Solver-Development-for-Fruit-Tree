from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("skimage")

from skimage.io import imsave

from orchard_vision.annotate_wood import WoodAnnotator


def _open_headless(annotator: WoodAnnotator) -> None:
    annotator._reset_image_state()
    annotator.image = annotator._load_image(annotator.image_paths[annotator._index])
    annotator.wood = np.zeros(annotator.image.shape[:2], dtype=bool)


def test_field_annotation_saves_and_resumes_roi_root_and_mask(tmp_path) -> None:
    source = tmp_path / "field_tree.jpg"
    image = np.full((120, 240, 3), 210, dtype=np.uint8)
    image[20:110, 105:125] = [85, 60, 40]
    imsave(source, image, check_contrast=False)
    out_dir = tmp_path / "labels"
    annotator = WoodAnnotator([source], out_dir, work_dim=120, brush_radius=3)
    _open_headless(annotator)

    assert annotator.image.shape[:2] == (60, 120)
    annotator.set_target_roi(42, 5, 72, 58)
    annotator.set_trunk_root(row=54, col=58)
    annotator._paint(x=58, y=40, erase=False)
    annotator.save()

    metadata = json.loads((out_dir / "field_tree_wood.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "orchard_visible_wood_annotation"
    assert metadata["version"] == 2
    assert metadata["source_shape"] == [120, 240, 3]
    assert metadata["working_shape"] == [60, 120, 3]
    assert metadata["source_to_work_scale"] == pytest.approx(0.5)
    assert metadata["trunk_root_rc_work"] == [54, 58]

    resumed = WoodAnnotator([source], out_dir, work_dim=120)
    _open_headless(resumed)
    assert resumed._load_saved_annotation() is True
    assert resumed.target_roi_xyxy == annotator.target_roi_xyxy
    assert resumed.trunk_root_rc == (54, 58)
    assert np.array_equal(resumed.wood, annotator.wood)


def test_brush_and_committed_proposals_are_clipped_to_target_roi(tmp_path) -> None:
    source = tmp_path / "field.png"
    imsave(source, np.full((80, 100, 3), 128, dtype=np.uint8), check_contrast=False)
    annotator = WoodAnnotator([source], tmp_path / "labels", work_dim=100, brush_radius=5)
    _open_headless(annotator)
    annotator.set_target_roi(30, 20, 70, 70)

    annotator._paint(x=10, y=10, erase=False)
    annotator._paint(x=50, y=50, erase=False)
    assert not annotator.wood[10, 10]
    assert annotator.wood[50, 50]

    annotator.proposal = np.ones(annotator.wood.shape, dtype=bool)
    annotator.commit_proposal()
    assert not annotator.wood[5, 5]
    assert annotator.wood[30, 40]


def test_roi_and_root_edits_support_undo_and_redo(tmp_path) -> None:
    source = tmp_path / "field.png"
    imsave(source, np.full((60, 80, 3), 128, dtype=np.uint8), check_contrast=False)
    annotator = WoodAnnotator([source], tmp_path / "labels", work_dim=80)
    _open_headless(annotator)

    annotator.set_target_roi(10, 5, 70, 55)
    annotator.set_trunk_root(row=50, col=40)
    assert annotator.trunk_root_rc == (50, 40)

    annotator.undo()
    assert annotator.trunk_root_rc is None
    annotator.redo()
    assert annotator.trunk_root_rc == (50, 40)


def test_changing_roi_removes_labels_from_neighbouring_trees(tmp_path) -> None:
    source = tmp_path / "field.png"
    imsave(source, np.full((60, 100, 3), 128, dtype=np.uint8), check_contrast=False)
    annotator = WoodAnnotator([source], tmp_path / "labels", work_dim=100)
    _open_headless(annotator)
    annotator.wood[20:30, 5:15] = True
    annotator.wood[20:30, 50:60] = True

    annotator.set_target_roi(40, 10, 70, 50)

    assert not annotator.wood[25, 10]
    assert annotator.wood[25, 55]
