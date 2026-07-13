from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skimage")

from skimage.io import imsave  # noqa: E402

import orchard_vision.wood_seg as wood_seg  # noqa: E402
from orchard_vision.wood_seg import (  # noqa: E402
    WoodDataset,
    WoodSample,
    WoodSegmenter,
    discover_wood_samples,
    segmentation_metrics,
    split_samples_by_group,
)


def _sample(name: str, group: str) -> WoodSample:
    return WoodSample(
        name=name,
        image_path=Path(f"{name}.png"),
        mask_path=Path(f"{name}_wood.png"),
        metadata_path=None,
        roi_xyxy=None,
        group_id=group,
    )


def _write_annotation(
    root: Path,
    name: str,
    *,
    roi: tuple[int, int, int, int],
    group: str | None = None,
) -> None:
    image = np.full((80, 100, 3), 140, dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[25:35, 35:45] = 255  # target wood inside the ROI
    mask[2:8, 2:8] = 255  # neighbouring-tree wood outside the valid ROI
    imsave(root / f"{name}.png", image, check_contrast=False)
    imsave(root / f"{name}_wood.png", mask, check_contrast=False)
    metadata = {"target_roi_xyxy_work": list(roi)}
    if group is not None:
        metadata["group_id"] = group
    (root / f"{name}_wood.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_tree_group_split_happens_before_view_tiling() -> None:
    samples = [
        _sample("tree_a__view1", "tree_a"),
        _sample("tree_a__view2", "tree_a"),
        _sample("tree_b__view1", "tree_b"),
        _sample("tree_c__view1", "tree_c"),
        _sample("tree_d__view1", "tree_d"),
        _sample("tree_e__view1", "tree_e"),
    ]

    splits = split_samples_by_group(samples, seed=7, val_fraction=0.2, test_fraction=0.2)

    group_sets = {
        split: {sample.group_id for sample in split_samples}
        for split, split_samples in splits.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])
    assert sum("tree_a" in groups for groups in group_sets.values()) == 1


def test_metadata_roi_and_group_are_discovered_and_outside_pixels_are_ignored(
    tmp_path,
) -> None:
    _write_annotation(tmp_path, "tree01__view02", roi=(20, 10, 70, 60), group="tree01")

    samples = discover_wood_samples(tmp_path)
    dataset = WoodDataset(samples, tile_size=64, overlap=16, train=False)
    _image, target, valid = dataset[0]

    assert samples[0].roi_xyxy == (20, 10, 70, 60)
    assert samples[0].group_id == "tree01"
    assert int(valid.sum()) == 50 * 50
    assert int(target.sum()) == 10 * 10
    assert dataset.valid_pixels == 50 * 50


class _ConstantLogitModel(torch.nn.Module):
    def forward(self, batch):
        return torch.zeros(
            (batch.shape[0], 1, batch.shape[2], batch.shape[3]),
            device=batch.device,
        )


def test_overlapping_window_inference_preserves_rectangular_image_shape() -> None:
    segmenter = WoodSegmenter(
        checkpoint="unused.pt",
        device="cpu",
        size=32,
        overlap=12,
        batch_size=3,
    )
    segmenter._model = _ConstantLogitModel()
    image = np.full((47, 71, 3), 128, dtype=np.uint8)

    probability = segmenter.predict_proba(image)

    assert probability.shape == (47, 71)
    assert np.allclose(probability, 0.5, atol=1e-6)


def test_segmentation_metrics_are_aggregated_from_pixel_counts() -> None:
    metrics = segmentation_metrics({"tp": 8, "fp": 2, "fn": 4, "tn": 10})

    assert metrics["dice"] == pytest.approx(16 / 22)
    assert metrics["iou"] == pytest.approx(8 / 14)
    assert metrics["precision"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(8 / 12)


class _TinyWoodModel(torch.nn.Module):
    def __init__(self, _backbone="tiny", _mid=4, *, pretrained=True):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.lateral = torch.nn.ModuleList([torch.nn.Identity()])
        self.fuse = torch.nn.Identity()
        self.classifier = torch.nn.Conv2d(4, 1, 1)

    def forward(self, batch):
        return self.classifier(torch.relu(self.backbone(batch)))


def test_training_writes_best_checkpoint_and_reproducibility_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    for index in range(3):
        _write_annotation(
            tmp_path,
            f"tree{index}",
            roi=(20, 10, 70, 60),
            group=f"tree{index}",
        )
    out_path = tmp_path / "tiny.pt"
    monkeypatch.setattr(wood_seg, "WoodSegModel", _TinyWoodModel)

    wood_seg.train_wood_seg(
        tmp_path,
        out_path,
        backbone="tiny",
        epochs=2,
        batch_size=4,
        device="cpu",
        tile_size=32,
        overlap=8,
        freeze_epochs=0,
        patience=2,
        pretrained=False,
    )

    checkpoint = torch.load(out_path, map_location="cpu", weights_only=False)
    metrics = json.loads(out_path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert checkpoint["version"] == 2
    assert set(checkpoint["split_groups"]) == {"train", "val", "test"}
    assert "held_out_test_metrics" in checkpoint
    assert metrics["best_epoch"] in {0, 1}
    assert len(metrics["history"]) == 2
