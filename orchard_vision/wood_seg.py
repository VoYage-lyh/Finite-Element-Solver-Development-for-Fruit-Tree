"""EfficientFormer wood segmentation for raw orchard photographs.

EfficientFormer-L1 is used as an ImageNet-pretrained feature extractor and a
small FPN-like head predicts visible wood versus leaf/background.  EfficientFormer
expects a 224-pixel attention grid, so raw photographs are never squeezed into a
single square: target-tree ROIs are split into overlapping tiles for training and
full photographs use the same overlapping-window strategy at inference.

The annotation sidecar written by :mod:`orchard_vision.annotate_wood` defines the
valid target-tree ROI.  Pixels outside that ROI are ignored rather than treated as
negative wood labels.  Splits are made by tree group *before* tiling; name related
views ``tree001__view01`` and ``tree001__view02`` to keep them in the same split.

Typical commands::

    python -m orchard_vision.wood_seg train --data datasets/wood_field
    python -m orchard_vision.wood_seg evaluate --data datasets/wood_field \
        --checkpoint weights/wood_seg.pt --split test

This module remains a deliberately small-domain component.  Fully occluded wood
cannot be learned from visible-wood labels and final skeletons remain editable in
Harvest Console.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from skimage.io import imread  # noqa: E402
from skimage.transform import rotate  # noqa: E402
from torch import nn  # noqa: E402

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CHECKPOINT_VERSION = 2
_SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class WoodSegModel(nn.Module):
    """EfficientFormer multi-scale features fused into one wood logit map."""

    def __init__(
        self,
        backbone: str = "efficientformer_l1",
        mid: int = 128,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
        )
        channels = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList(nn.Conv2d(channel, mid, 1) for channel in channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(mid, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        features = self.backbone(x)
        target = features[0].shape[-2:]
        fused = sum(
            F.interpolate(lateral(feature), size=target, mode="bilinear", align_corners=False)
            for lateral, feature in zip(self.lateral, features)
        )
        logits = self.classifier(self.fuse(fused))
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------------------
# Annotation discovery and leakage-safe grouping
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WoodSample:
    """One annotated target tree at the annotator's working resolution."""

    name: str
    image_path: Path
    mask_path: Path
    metadata_path: Path | None
    roi_xyxy: tuple[int, int, int, int] | None
    group_id: str
    split_hint: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read annotation metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Annotation metadata must be a JSON object: {path}")
    return value


def discover_wood_samples(data_dir: str | Path) -> list[WoodSample]:
    """Find image/mask pairs and optional ROI/group metadata."""
    data_dir = Path(data_dir)
    samples: list[WoodSample] = []
    for mask_path in sorted(data_dir.glob("*_wood.png")):
        name = mask_path.stem.removesuffix("_wood")
        image_path = mask_path.with_name(f"{name}.png")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image paired with {mask_path.name}: {image_path}")

        metadata_path = mask_path.with_suffix(".json")
        metadata = _read_json(metadata_path) if metadata_path.exists() else {}
        roi_raw = metadata.get("target_roi_xyxy_work")
        roi = None
        if roi_raw is not None:
            if not isinstance(roi_raw, list) or len(roi_raw) != 4:
                raise ValueError(f"Invalid target ROI in {metadata_path}")
            roi = tuple(int(round(float(value))) for value in roi_raw)

        # A double underscore is an explicit, filename-only group convention:
        # tree001__view02 -> tree001. A metadata group_id overrides it.
        group_id = str(metadata.get("group_id") or name.split("__", 1)[0])
        split_hint = metadata.get("split")
        if split_hint is not None:
            split_hint = str(split_hint).lower()
            if split_hint not in _SPLIT_NAMES:
                raise ValueError(
                    f"Invalid split {split_hint!r} in {metadata_path}; expected train/val/test"
                )
        samples.append(
            WoodSample(
                name=name,
                image_path=image_path,
                mask_path=mask_path,
                metadata_path=metadata_path if metadata_path.exists() else None,
                roi_xyxy=roi,
                group_id=group_id,
                split_hint=split_hint,
            )
        )
    if not samples:
        raise FileNotFoundError(f"No *_wood.png label pairs in {data_dir}")
    return samples


def _requested_holdout_count(group_count: int, fraction: float) -> int:
    if fraction <= 0.0 or group_count < 3:
        return 0
    return max(1, int(round(group_count * fraction)))


def split_samples_by_group(
    samples: Sequence[WoodSample],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[WoodSample]]:
    """Split whole tree groups before tiles are created, preventing data leakage."""
    if val_fraction < 0.0 or test_fraction < 0.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to < 1")

    grouped: dict[str, list[WoodSample]] = {}
    hints: dict[str, str] = {}
    for sample in samples:
        grouped.setdefault(sample.group_id, []).append(sample)
        if sample.split_hint is not None:
            previous = hints.setdefault(sample.group_id, sample.split_hint)
            if previous != sample.split_hint:
                raise ValueError(f"Group {sample.group_id!r} has conflicting split hints")

    groups = sorted(grouped)
    group_count = len(groups)
    target_val = _requested_holdout_count(group_count, val_fraction)
    target_test = _requested_holdout_count(group_count, test_fraction)
    while target_val + target_test > max(0, group_count - 1):
        if target_test >= target_val and target_test:
            target_test -= 1
        elif target_val:
            target_val -= 1

    assigned = {name: {group for group, split in hints.items() if split == name} for name in _SPLIT_NAMES}
    automatic = [group for group in groups if group not in hints]
    random.Random(seed).shuffle(automatic)

    needed_test = max(0, target_test - len(assigned["test"]))
    assigned["test"].update(automatic[:needed_test])
    automatic = automatic[needed_test:]
    needed_val = max(0, target_val - len(assigned["val"]))
    assigned["val"].update(automatic[:needed_val])
    assigned["train"].update(automatic[needed_val:])

    if not assigned["train"]:
        donor = "val" if len(assigned["val"]) > 1 else "test"
        movable = [group for group in sorted(assigned[donor]) if group not in hints]
        if not movable:
            raise ValueError("Split hints leave no training group")
        assigned[donor].remove(movable[0])
        assigned["train"].add(movable[0])

    return {
        split: [sample for group in sorted(assigned[split]) for sample in grouped[group]]
        for split in _SPLIT_NAMES
    }


# --------------------------------------------------------------------------------------
# ROI-aware overlapping tile dataset
# --------------------------------------------------------------------------------------
def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        image = image * (255.0 if finite_max <= 1.0 else 1.0)
    elif image.dtype.itemsize > 1:
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _clamp_roi(
    roi: tuple[int, int, int, int] | None,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = shape
    if roi is None:
        return (0, 0, width, height)
    left, top, right, bottom = roi
    left = int(np.clip(left, 0, width - 1))
    top = int(np.clip(top, 0, height - 1))
    right = int(np.clip(right, left + 1, width))
    bottom = int(np.clip(bottom, top + 1, height))
    return (left, top, right, bottom)


def _window_origins(length: int, size: int, overlap: int) -> list[int]:
    if size <= 0:
        raise ValueError("tile size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile size")
    if length <= size:
        return [0]
    stride = size - overlap
    origins = list(range(0, length - size + 1, stride))
    if origins[-1] != length - size:
        origins.append(length - size)
    return origins


def _pad_tile(array: np.ndarray, size: int, *, image: bool) -> np.ndarray:
    height, width = array.shape[:2]
    pad = ((0, size - height), (0, size - width))
    if array.ndim == 3:
        pad += ((0, 0),)
    if height == size and width == size:
        return array.copy()
    if image:
        mode = "reflect" if min(height, width) > 1 else "edge"
        return np.pad(array, pad, mode=mode)
    return np.pad(array, pad, mode="constant", constant_values=False)


@dataclass(frozen=True)
class _TileRef:
    source_index: int
    top: int
    left: int
    positive: bool


@dataclass
class _SourceArrays:
    name: str
    image: np.ndarray
    mask: np.ndarray


def _augment(
    image: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.65:
        angle = random.uniform(-18.0, 18.0)
        image = rotate(image, angle, resize=False, mode="reflect", preserve_range=True)
        mask = rotate(mask.astype(float), angle, resize=False, order=0, mode="constant") > 0.5
        valid = rotate(valid.astype(float), angle, resize=False, order=0, mode="constant") > 0.5
    if random.random() < 0.5:
        image = image[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        valid = valid[:, ::-1].copy()
    if random.random() < 0.8:
        image = np.clip(
            image * random.uniform(0.72, 1.28) + random.uniform(-0.10, 0.10),
            0.0,
            1.0,
        )
    if random.random() < 0.5:
        channel_gain = np.asarray(
            [random.uniform(0.88, 1.12) for _ in range(3)],
            dtype=np.float32,
        )
        image = np.clip(image * channel_gain, 0.0, 1.0)
    return image, mask, valid


def _normalize(image: np.ndarray) -> torch.Tensor:
    normalized = (image.astype(np.float32) - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


class WoodDataset(torch.utils.data.Dataset):
    """In-memory source images with lazy ROI-tile extraction."""

    def __init__(
        self,
        samples: Sequence[WoodSample] | str | Path,
        *,
        tile_size: int = 224,
        overlap: int = 64,
        train: bool = True,
        max_negative_ratio: float = 1.5,
        seed: int = 42,
    ) -> None:
        if isinstance(samples, (str, Path)):
            samples = discover_wood_samples(samples)
        if not samples:
            raise ValueError("WoodDataset needs at least one annotated sample")
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.train = bool(train)
        self.sources: list[_SourceArrays] = []
        refs: list[_TileRef] = []

        for sample in samples:
            image = _as_rgb_uint8(imread(sample.image_path))
            mask = np.asarray(imread(sample.mask_path)) > 127
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Image/mask shape mismatch for {sample.name}: "
                    f"{image.shape[:2]} != {mask.shape}"
                )
            left, top, right, bottom = _clamp_roi(sample.roi_xyxy, mask.shape)
            image = image[top:bottom, left:right]
            mask = mask[top:bottom, left:right]
            source_index = len(self.sources)
            self.sources.append(_SourceArrays(sample.name, image, mask))

            for row in _window_origins(image.shape[0], self.tile_size, self.overlap):
                for col in _window_origins(image.shape[1], self.tile_size, self.overlap):
                    patch = mask[row : row + self.tile_size, col : col + self.tile_size]
                    refs.append(_TileRef(source_index, row, col, bool(patch.any())))

        positives = [ref for ref in refs if ref.positive]
        negatives = [ref for ref in refs if not ref.positive]
        if train:
            if not positives:
                raise ValueError("Training annotations contain no positive wood pixels")
            random.Random(seed).shuffle(negatives)
            keep_negative = min(
                len(negatives),
                max(1, int(math.ceil(len(positives) * max_negative_ratio))),
            )
            refs = positives + negatives[:keep_negative]
            random.Random(seed + 1).shuffle(refs)
        self.tiles = refs
        if not self.tiles:
            raise ValueError("No training/evaluation tiles were produced")

        self.positive_pixels = 0
        self.valid_pixels = 0
        for ref in self.tiles:
            source = self.sources[ref.source_index]
            patch = source.mask[
                ref.top : ref.top + self.tile_size,
                ref.left : ref.left + self.tile_size,
            ]
            self.positive_pixels += int(np.count_nonzero(patch))
            self.valid_pixels += int(patch.size)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.tiles[index]
        source = self.sources[ref.source_index]
        image_patch = source.image[
            ref.top : ref.top + self.tile_size,
            ref.left : ref.left + self.tile_size,
        ]
        mask_patch = source.mask[
            ref.top : ref.top + self.tile_size,
            ref.left : ref.left + self.tile_size,
        ]
        valid_patch = np.ones(mask_patch.shape, dtype=bool)
        image = _pad_tile(image_patch, self.tile_size, image=True).astype(np.float32) / 255.0
        mask = _pad_tile(mask_patch, self.tile_size, image=False)
        valid = _pad_tile(valid_patch, self.tile_size, image=False)
        if self.train:
            image, mask, valid = _augment(image, mask, valid)
        return (
            _normalize(image),
            torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None],
            torch.from_numpy(np.ascontiguousarray(valid, dtype=np.float32))[None],
        )

    @property
    def positive_weight(self) -> float:
        negative_pixels = max(0, self.valid_pixels - self.positive_pixels)
        if self.positive_pixels == 0:
            return 1.0
        return float(np.clip(negative_pixels / self.positive_pixels, 1.0, 20.0))


# --------------------------------------------------------------------------------------
# Loss, metrics, training and evaluation
# --------------------------------------------------------------------------------------
def _masked_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    valid_count = valid.sum().clamp_min(1.0)
    bce_loss = (bce * valid).sum() / valid_count

    probability = torch.sigmoid(logits) * valid
    target_valid = target * valid
    intersection = (probability * target_valid).sum(dim=(1, 2, 3))
    union = probability.sum(dim=(1, 2, 3)) + target_valid.sum(dim=(1, 2, 3))
    dice_loss = (1.0 - (2.0 * intersection + 1.0) / (union + 1.0)).mean()
    return bce_loss + dice_loss


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _update_counts(
    counts: dict[str, int],
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> None:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    valid_mask = np.ones(target.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    counts["tp"] += int(np.count_nonzero(prediction & target & valid_mask))
    counts["fp"] += int(np.count_nonzero(prediction & ~target & valid_mask))
    counts["fn"] += int(np.count_nonzero(~prediction & target & valid_mask))
    counts["tn"] += int(np.count_nonzero(~prediction & ~target & valid_mask))


def segmentation_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Return aggregate binary segmentation metrics from pixel counts."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "dice": (2.0 * tp) / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def _format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{name}={value:.4f}" for name, value in metrics.items())


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    pos_weight: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    model.eval()
    counts = _empty_counts()
    total_loss = 0.0
    batches = 0
    for image, target, valid in loader:
        image, target, valid = image.to(device), target.to(device), valid.to(device)
        logits = model(image)
        total_loss += float(_masked_segmentation_loss(logits, target, valid, pos_weight))
        prediction = (torch.sigmoid(logits) >= threshold).cpu().numpy()
        _update_counts(
            counts,
            prediction,
            target.cpu().numpy() > 0.5,
            valid.cpu().numpy() > 0.5,
        )
        batches += 1
    return total_loss / max(1, batches), segmentation_metrics(counts)


def _split_group_names(splits: dict[str, list[WoodSample]]) -> dict[str, list[str]]:
    return {
        split: sorted({sample.group_id for sample in split_samples})
        for split, split_samples in splits.items()
    }


def _checkpoint_payload(
    model: WoodSegModel,
    *,
    backbone: str,
    tile_size: int,
    overlap: int,
    threshold: float,
    epoch: int,
    validation_metrics: dict[str, float],
    split_groups: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "format": "orchard_efficientformer_wood_seg",
        "version": _CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "backbone": backbone,
        "tile_size": tile_size,
        "tile_overlap": overlap,
        "threshold": threshold,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "split_groups": split_groups,
    }


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":")[1]) if ":" in device else 0
        if index >= torch.cuda.device_count():
            index = 0
            device = "cuda:0"
        torch.cuda.set_per_process_memory_fraction(0.7, index)
        return device
    return "cpu"


def train_wood_seg(
    data_dir: Path,
    out_path: Path,
    *,
    backbone: str = "efficientformer_l1",
    epochs: int = 160,
    batch_size: int = 8,
    device: str = "cuda:1",
    tile_size: int = 224,
    overlap: int = 64,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    freeze_epochs: int = 12,
    patience: int = 25,
    threshold: float = 0.5,
    pretrained: bool = True,
) -> Path:
    """Train with tree-group splits, best-validation checkpointing and early stopping."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(device)

    samples = discover_wood_samples(data_dir)
    splits = split_samples_by_group(
        samples,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    split_groups = _split_group_names(splits)
    print(
        "split groups: "
        + ", ".join(f"{name}={len(split_groups[name])}" for name in _SPLIT_NAMES),
        flush=True,
    )
    train_dataset = WoodDataset(
        splits["train"],
        tile_size=tile_size,
        overlap=overlap,
        train=True,
        seed=seed,
    )
    val_dataset = (
        WoodDataset(
            splits["val"],
            tile_size=tile_size,
            overlap=overlap,
            train=False,
        )
        if splits["val"]
        else None
    )
    test_dataset = (
        WoodDataset(
            splits["test"],
            tile_size=tile_size,
            overlap=overlap,
            train=False,
        )
        if splits["test"]
        else None
    )
    generator = torch.Generator().manual_seed(seed)
    pin_memory = device.startswith("cuda")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        if test_dataset is not None
        else None
    )

    model = WoodSegModel(backbone, pretrained=pretrained).to(device)
    backbone_frozen = freeze_epochs > 0
    for parameter in model.backbone.parameters():
        parameter.requires_grad = not backbone_frozen
    head_parameters = [
        *model.lateral.parameters(),
        *model.fuse.parameters(),
        *model.classifier.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 8e-5},
            {"params": head_parameters, "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )
    pos_weight = torch.tensor([train_dataset.positive_weight], device=device)
    print(
        f"training tiles={len(train_dataset)} val_tiles={len(val_dataset) if val_dataset else 0} "
        f"test_tiles={len(test_dataset) if test_dataset else 0} "
        f"pos_weight={float(pos_weight.item()):.2f} device={device}",
        flush=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        if backbone_frozen and epoch >= freeze_epochs:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
            backbone_frozen = False
            print(f"epoch {epoch}: unfroze EfficientFormer backbone", flush=True)

        model.train()
        if backbone_frozen:
            model.backbone.eval()
        total_loss = 0.0
        for image, target, valid in train_loader:
            image, target, valid = image.to(device), target.to(device), valid.to(device)
            logits = model(image)
            loss = _masked_segmentation_loss(logits, target, valid, pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        train_loss = total_loss / max(1, len(train_loader))

        if val_loader is not None:
            val_loss, val_metrics = _evaluate_loader(
                model,
                val_loader,
                device,
                pos_weight,
                threshold=threshold,
            )
            score = val_metrics["dice"]
        else:
            val_loss, val_metrics = train_loss, {"dice": -train_loss}
            score = -train_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_metrics": val_metrics,
            }
        )

        improved = score > best_score + 1e-5
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    backbone=backbone,
                    tile_size=tile_size,
                    overlap=overlap,
                    threshold=threshold,
                    epoch=epoch,
                    validation_metrics=val_metrics,
                    split_groups=split_groups,
                ),
                out_path,
            )
        else:
            stale_epochs += 1

        print(
            f"epoch {epoch:3d}/{epochs - 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} {_format_metrics(val_metrics)}"
            f"{'  [best]' if improved else ''}",
            flush=True,
        )
        if val_loader is not None and patience > 0 and stale_epochs >= patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = None
    if test_loader is not None:
        test_loss, test_metrics = _evaluate_loader(
            model,
            test_loader,
            device,
            pos_weight,
            threshold=threshold,
        )
        print(f"held-out test: loss={test_loss:.4f} {_format_metrics(test_metrics)}", flush=True)
        checkpoint["held_out_test_metrics"] = test_metrics
        torch.save(checkpoint, out_path)

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "format": "orchard_wood_seg_training_metrics",
                "version": 1,
                "configuration": {
                    "backbone": backbone,
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "val_fraction": val_fraction,
                    "test_fraction": test_fraction,
                    "seed": seed,
                    "freeze_epochs": freeze_epochs,
                    "patience": patience,
                    "threshold": threshold,
                },
                "split_groups": split_groups,
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "held_out_test_metrics": test_metrics,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved best wood-seg model (epoch {best_epoch}) -> {out_path}", flush=True)
    print(f"saved training metrics -> {metrics_path}", flush=True)
    return out_path


# --------------------------------------------------------------------------------------
# Overlapping-window inference
# --------------------------------------------------------------------------------------
def _blend_weights(size: int) -> np.ndarray:
    if size == 1:
        return np.ones((1, 1), dtype=np.float32)
    axis = np.hanning(size).astype(np.float32)
    return np.clip(np.outer(axis, axis), 0.05, None)


@dataclass
class WoodSegmenter:
    """EfficientFormer segmenter with seam-resistant overlapping-window inference."""

    checkpoint: str = "weights/wood_seg.pt"
    device: str = "cuda:1"
    size: int | None = None
    overlap: int | None = None
    threshold: float | None = None
    batch_size: int = 8
    _model: object = field(default=None, init=False, repr=False)

    def _load(self) -> nn.Module:
        if self._model is None:
            self.device = _resolve_device(self.device)
            checkpoint = torch.load(
                self.checkpoint,
                map_location=self.device,
                weights_only=False,
            )
            model = WoodSegModel(
                checkpoint.get("backbone", "efficientformer_l1"),
                pretrained=False,
            ).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.size = int(self.size or checkpoint.get("tile_size", 224))
            self.overlap = int(
                self.overlap if self.overlap is not None else checkpoint.get("tile_overlap", 64)
            )
            self.threshold = float(
                self.threshold if self.threshold is not None else checkpoint.get("threshold", 0.5)
            )
            self._model = model
        return self._model  # type: ignore[return-value]

    @torch.no_grad()
    def predict_proba(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a full-resolution wood probability map using overlapping tiles."""
        model = self._load()
        image = _as_rgb_uint8(image_rgb)
        height, width = image.shape[:2]
        size = int(self.size or 224)
        overlap = int(self.overlap or 0)
        rows = _window_origins(height, size, overlap)
        cols = _window_origins(width, size, overlap)
        blend = _blend_weights(size)
        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        windows: list[tuple[int, int, int, int, np.ndarray]] = []
        for row in rows:
            for col in cols:
                patch = image[row : row + size, col : col + size]
                patch_height, patch_width = patch.shape[:2]
                padded = _pad_tile(patch, size, image=True).astype(np.float32) / 255.0
                windows.append((row, col, patch_height, patch_width, padded))

        for start in range(0, len(windows), self.batch_size):
            batch_windows = windows[start : start + self.batch_size]
            tensor = torch.stack([_normalize(window[4]) for window in batch_windows]).to(
                self.device
            )
            probabilities = torch.sigmoid(model(tensor))[:, 0].cpu().numpy()
            for probability, (row, col, patch_height, patch_width, _patch) in zip(
                probabilities,
                batch_windows,
            ):
                local_weight = blend[:patch_height, :patch_width]
                probability_sum[
                    row : row + patch_height,
                    col : col + patch_width,
                ] += probability[:patch_height, :patch_width] * local_weight
                weight_sum[
                    row : row + patch_height,
                    col : col + patch_width,
                ] += local_weight
        return probability_sum / np.maximum(weight_sum, 1e-6)

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        self._load()
        threshold = self.threshold if self.threshold is not None else 0.5
        return self.predict_proba(image_rgb) >= float(threshold)


def _samples_for_checkpoint_split(
    samples: Sequence[WoodSample],
    checkpoint: dict[str, Any],
    split: str,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[WoodSample]:
    split_groups = checkpoint.get("split_groups", {})
    if split in split_groups:
        wanted = set(split_groups[split])
        return [sample for sample in samples if sample.group_id in wanted]
    return split_samples_by_group(
        samples,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )[split]


def evaluate_wood_seg(
    data_dir: Path,
    checkpoint_path: Path,
    *,
    split: str = "test",
    device: str = "cuda:1",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate full target-tree ROIs using the checkpoint's original group split."""
    if split not in _SPLIT_NAMES:
        raise ValueError(f"split must be one of {_SPLIT_NAMES}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    samples = _samples_for_checkpoint_split(
        discover_wood_samples(data_dir),
        checkpoint,
        split,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if not samples:
        raise ValueError(f"No samples belong to the {split!r} split")
    segmenter = WoodSegmenter(checkpoint=str(checkpoint_path), device=device)
    counts = _empty_counts()
    for sample in samples:
        image = _as_rgb_uint8(imread(sample.image_path))
        target = np.asarray(imread(sample.mask_path)) > 127
        if target.ndim == 3:
            target = target[..., 0]
        left, top, right, bottom = _clamp_roi(sample.roi_xyxy, target.shape)
        prediction = segmenter.segment(image[top:bottom, left:right])
        _update_counts(counts, prediction, target[top:bottom, left:right])
    metrics = segmentation_metrics(counts)
    print(f"{split} groups={len({sample.group_id for sample in samples})} {_format_metrics(metrics)}")
    return metrics


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EfficientFormer visible-wood segmentation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train with group splits and ROI tiles")
    train.add_argument("--data", default="datasets/wood_field")
    train.add_argument("--out", default="weights/wood_seg.pt")
    train.add_argument("--backbone", default="efficientformer_l1")
    train.add_argument("--epochs", type=int, default=160)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--device", default="cuda:1")
    train.add_argument("--tile-size", type=int, default=224)
    train.add_argument("--overlap", type=int, default=64)
    train.add_argument("--val-fraction", type=float, default=0.15)
    train.add_argument("--test-fraction", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--freeze-epochs", type=int, default=12)
    train.add_argument("--patience", type=int, default=25)
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument(
        "--no-pretrained",
        action="store_true",
        help="do not initialise the training backbone from ImageNet weights",
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved split on full ROIs")
    evaluate.add_argument("--data", default="datasets/wood_field")
    evaluate.add_argument("--checkpoint", default="weights/wood_seg.pt")
    evaluate.add_argument("--split", choices=_SPLIT_NAMES, default="test")
    evaluate.add_argument("--device", default="cuda:1")
    evaluate.add_argument("--val-fraction", type=float, default=0.15)
    evaluate.add_argument("--test-fraction", type=float, default=0.15)
    evaluate.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        train_wood_seg(
            Path(args.data),
            Path(args.out),
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            freeze_epochs=args.freeze_epochs,
            patience=args.patience,
            threshold=args.threshold,
            pretrained=not args.no_pretrained,
        )
    else:
        evaluate_wood_seg(
            Path(args.data),
            Path(args.checkpoint),
            split=args.split,
            device=args.device,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
