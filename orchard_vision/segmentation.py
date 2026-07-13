"""Branch-mask front-ends: RGB photo → boolean "woody branch" mask.

The pipeline depends only on the :class:`BranchSegmenter` protocol, so the rough
classical baseline below can be replaced by a learned segmenter (an
EfficientFormer semantic head, SAM 2, …) exposing the same ``segment`` method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage.filters import sato, threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, disk, opening


@runtime_checkable
class BranchSegmenter(Protocol):
    """Anything that turns an RGB image into a boolean branch mask."""

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a boolean mask ``(H, W)`` that is ``True`` on woody branches."""
        ...


@dataclass
class ClassicalSegmenter:
    """Training-free branch segmenter for clean / studio (bright-background) shots.

    Two cues are intersected:

    1. **Foreground** — pixels darker than the near-white background.
    2. **Tubular-ness** — a multiscale Sato ridge response, which fires on the
       elongated woody trunk/branches and stays low on rounded leaf blobs.

    Only the largest connected component is kept, dropping isolated leaf clutter
    and leaving the trunk-rooted branch network. This is a deliberately simple
    baseline (speed over accuracy); swap in a learned segmenter for
    canopy-occluded field imagery.
    """

    background_is_bright: bool = True
    foreground_max_luma: float = 0.90  # keep pixels darker than this (0..1)
    # Bias the ridge scales toward branch widths, not leaf veins (drop sigma≈1).
    ridge_sigmas: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    closing_radius: int = 3  # bridge small occlusion gaps along a branch
    opening_radius: int = 2  # erase thin leaf webs (structures < ~2·radius wide)
    min_object_fraction: float = 5e-4  # smallest kept blob, as fraction of image

    def segment(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = rgb2gray(image_rgb[..., :3])  # tolerate RGBA, returns 0..1 float
        if self.background_is_bright:
            foreground = gray < self.foreground_max_luma
        else:
            foreground = gray > (1.0 - self.foreground_max_luma)

        ridges = np.nan_to_num(sato(gray, sigmas=self.ridge_sigmas, black_ridges=True))
        inside = ridges[foreground]
        positive = inside[inside > 0]
        if positive.size < 2:
            return np.zeros(gray.shape, dtype=bool)
        threshold = float(threshold_otsu(positive))
        mask = foreground & (ridges > threshold)

        mask = closing(mask, disk(self.closing_radius))
        mask = opening(mask, disk(self.opening_radius))
        mask = binary_fill_holes(mask)
        mask = _remove_small(mask, int(gray.size * self.min_object_fraction))
        return _largest_component(mask)


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop 8-connected blobs smaller than ``min_size`` pixels."""
    labels = label(mask, connectivity=2)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero(counts >= min_size)
    keep = keep[keep != 0]  # never keep the background label
    return np.isin(labels, keep)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected blob (the trunk-rooted structure)."""
    labels = label(mask, connectivity=2)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background
    return labels == int(counts.argmax())
