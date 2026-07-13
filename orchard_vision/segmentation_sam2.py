"""SAM 2 branch segmenter — refine visible wood via skeleton-seeded prompts.

This is an optional front-end (needs ``ultralytics``); it is imported lazily so
``import orchard_vision`` never requires torch. It is **safe-by-construction for
WSL2**, where a CUDA out-of-memory does not raise cleanly but hangs the whole VM:

* runs on an explicit GPU (default ``cuda:1``, the idle card) instead of the
  busy/display GPU 0;
* caps the CUDA memory fraction so an OOM becomes a catchable Python error rather
  than a driver hang;
* uses a **handful of point prompts** — NOT automatic mask generation — so peak
  memory stays tiny.

Strategy: the classical segmenter finds the visible woody skeleton; each ordered
branch centreline seeds a positive-point prompt and SAM 2 returns a clean
full-width mask for that branch. Unioned, these give a tighter mask than the raw
ridge response and SAM's learned prior bridges small leaf gaps along a branch it
is prompted on. It does **not** discover branches fully hidden behind canopy —
that needs a trained wood-segmentation model.
"""
from __future__ import annotations

import os

# Reduce fragmentation-driven OOM; must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402
from skimage.color import rgb2gray  # noqa: E402
from skimage.transform import rescale, resize  # noqa: E402

from orchard_vision.branch_ordering import order_branches  # noqa: E402
from orchard_vision.segmentation import (  # noqa: E402
    BranchSegmenter,
    ClassicalSegmenter,
    _largest_component,
)
from orchard_vision.skeleton_graph import build_graph, skeletonize_mask  # noqa: E402
from orchard_vision.types import Branch  # noqa: E402


@dataclass
class Sam2Segmenter:
    """Refine the classical branch mask with SAM 2 point prompts (no AMG)."""

    checkpoint: str = "sam2_t.pt"
    device: str = "cuda:1"  # idle GPU; falls back to CPU if CUDA is absent
    memory_fraction: float = 0.7  # guardrail so OOM raises instead of hanging
    sam_dimension: int = 640  # run SAM at this longer-side resolution
    points_per_branch: int = 6
    max_mask_fraction: float = 0.4  # drop a prompt's mask if it floods the image
    foreground_max_luma: float = 0.90
    seed_segmenter: BranchSegmenter = field(default_factory=ClassicalSegmenter)
    _model: object = field(default=None, init=False, repr=False)

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = image_rgb[..., :3]
        height, width = image_rgb.shape[:2]

        # 1. Classical seed skeleton → ordered branches (prompt units).
        seed_mask = self.seed_segmenter.segment(image_rgb)
        skeleton, radius = skeletonize_mask(seed_mask)
        branches, _ = order_branches(build_graph(skeleton, radius))
        if not branches:
            return seed_mask

        # 2. Prompt SAM 2 once per branch at a reduced resolution.
        scale = min(1.0, self.sam_dimension / max(height, width))
        sam_image = (
            rescale(image_rgb, scale, channel_axis=-1, anti_aliasing=True, preserve_range=True)
            .astype(np.uint8)
            if scale < 1.0
            else image_rgb
        )
        model, torch = self._load()
        pixel_budget = sam_image.shape[0] * sam_image.shape[1] * self.max_mask_fraction
        refined = np.zeros(sam_image.shape[:2], dtype=bool)

        for branch in branches:
            points = self._seed_points(branch, scale)
            if len(points) == 0:
                continue
            try:
                results = model(
                    sam_image,
                    points=points.tolist(),
                    labels=[1] * len(points),
                    device=self.device,
                    verbose=False,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            masks = results[0].masks
            if masks is None:
                continue
            union = masks.data.cpu().numpy().any(axis=0)
            if 0 < union.sum() <= pixel_budget:
                refined |= union

        # 3. Back to the pipeline's working resolution; trim to foreground.
        refined = resize(refined, (height, width), order=0, preserve_range=True).astype(bool)
        foreground = rgb2gray(image_rgb) < self.foreground_max_luma
        refined &= foreground
        return _largest_component(refined) if refined.any() else seed_mask

    def _seed_points(self, branch: Branch, scale: float) -> np.ndarray:
        """Evenly spaced interior points along a branch, as SAM ``[x, y]`` prompts."""
        pixels = branch.pixels
        count = len(pixels)
        if count < 3:
            indices = np.array([count // 2])
        else:
            span = np.linspace(int(0.15 * count), int(0.85 * count), self.points_per_branch)
            indices = np.unique(np.clip(span.astype(int), 0, count - 1))
        rows_cols = pixels[indices].astype(float)
        return rows_cols[:, ::-1] * scale  # (k, 2) as [x, y]

    def _load(self):
        if self._model is None:
            from ultralytics import SAM  # lazy: keeps torch out of the base import
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                index = int(self.device.split(":")[1]) if ":" in self.device else 0
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, index)
            else:
                self.device = "cpu"
            self._model = (SAM(self.checkpoint), torch)
        return self._model
