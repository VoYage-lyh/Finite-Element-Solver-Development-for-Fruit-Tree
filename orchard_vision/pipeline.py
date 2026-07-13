"""End-to-end orchestration: a tree photo → ordered skeleton → solver JSON.

:class:`PhotoToSkeletonPipeline` chains the stages (segment → skeletonise →
build graph → order branches → lift to 3D → export) and can render a
level-coloured overlay for eyeballing the result. Everything is configured
through :class:`PipelineConfig`; the segmenter is injectable so a learned model
can replace the classical baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from skimage.io import imread
from skimage.transform import rescale

from orchard_vision.branch_ordering import drop_low_primaries, order_branches
from orchard_vision.export_skeleton import branches_to_skeleton_payload
from orchard_vision.instances import branch_instances, extend_branches_to_mask
from orchard_vision.lift_3d import MonocularPlanarLift
from orchard_vision.segmentation import BranchSegmenter, ClassicalSegmenter
from orchard_vision.skeleton_graph import build_graph, skeletonize_mask
from orchard_vision.types import Branch, SkeletonGraph

# Overlay colours per branch order (trunk → primary → secondary → tertiary → …).
_LEVEL_COLOURS = {0: "#e6194B", 1: "#f58231", 2: "#3cb44b", 3: "#4363d8"}
_DEEP_COLOUR = "#911eb4"


@dataclass
class PipelineConfig:
    """Tunable knobs for the whole pipeline (defaults favour speed)."""

    tree_height_m: float = 3.0  # scale fallback when no trunk diameter is measured
    trunk_diameter_m: float | None = None  # measured base diameter → metric scale (preferred)
    max_dimension: int = 1024  # downscale longer image side to this before work
    max_turn_deg: float = 80.0  # junction continuation tolerance
    min_spur_px: float = 12.0  # prune leaf hairs shorter than this
    junction_merge_px: float = 8.0  # merge junction clusters into one branch point
    radius_weight: float = 1.0  # prefer thick continuation (keeps the trunk intact)
    max_level: int | None = 2  # focus on trunk → primary → secondary accuracy
    min_primary_height_m: float = 0.15  # forbid primaries attaching below this above the base
    max_points_per_branch: int = 16
    segmenter: BranchSegmenter = field(default_factory=ClassicalSegmenter)


@dataclass
class PipelineResult:
    """Everything the pipeline produced, for export and inspection."""

    name: str
    image: np.ndarray  # (possibly downscaled) RGB used for all pixel work
    mask: np.ndarray
    skeleton: np.ndarray
    radius: np.ndarray
    graph: SkeletonGraph
    branches: list[Branch]
    root_rc: tuple[int, int]
    lift: MonocularPlanarLift
    payload: dict[str, Any]
    instances: np.ndarray  # label image: 0 = bg, i = branches[i-1] (per-branch mask)

    def level_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for branch in self.branches:
            counts[branch.level] = counts.get(branch.level, 0) + 1
        return dict(sorted(counts.items()))


class PhotoToSkeletonPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(self, image_path: str | Path) -> PipelineResult:
        image_path = Path(image_path)
        image = self._load_image(image_path)

        mask = self.config.segmenter.segment(image)
        skeleton, radius = skeletonize_mask(mask)
        graph = build_graph(skeleton, radius)
        branches, root_rc = order_branches(
            graph,
            min_spur_px=self.config.min_spur_px,
            junction_merge_px=self.config.junction_merge_px,
            max_level=self.config.max_level,
        )
        if not branches:
            raise ValueError(f"No branches extracted from {image_path.name}; check segmentation.")
        # Skeletonisation stops short of the rounded mask tips — grow centrelines
        # back out to the wood-mask edge so they span their instance regions.
        extend_branches_to_mask(branches, mask)

        if self.config.trunk_diameter_m:
            lift = MonocularPlanarLift.from_trunk_diameter(
                branches, self.config.trunk_diameter_m, root_rc
            )
        else:
            lift = MonocularPlanarLift.from_tree_height(branches, self.config.tree_height_m, root_rc)

        # Structural rules: no primaries right at the base, and every child starts
        # exactly on its parent centreline (no free-floating branch roots).
        min_height_px = self.config.min_primary_height_m / lift.meters_per_pixel
        branches = drop_low_primaries(branches, root_rc[0], min_height_px)

        payload = branches_to_skeleton_payload(
            branches,
            lift,
            model_name=image_path.stem,
            max_points_per_branch=self.config.max_points_per_branch,
        )
        return PipelineResult(
            name=image_path.stem,
            image=image,
            mask=mask,
            skeleton=skeleton,
            radius=radius,
            graph=graph,
            branches=branches,
            root_rc=root_rc,
            lift=lift,
            payload=payload,
            instances=branch_instances(mask, branches),
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = imread(image_path)
        if image.ndim == 2:  # greyscale → RGB
            image = np.stack([image] * 3, axis=-1)
        image = image[..., :3]
        longest = max(image.shape[:2])
        if longest > self.config.max_dimension:
            factor = self.config.max_dimension / longest
            image = rescale(
                image, factor, channel_axis=-1, anti_aliasing=True, preserve_range=True
            ).astype(np.uint8)
        return image

    @staticmethod
    def save_overlay(result: PipelineResult, path: str | Path) -> Path:
        """Draw the ordered branches over the photo, coloured by order."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = result.image.shape[:2]
        fig, ax = plt.subplots(figsize=(9, 9 * height / width))
        ax.imshow(result.image)
        for branch in result.branches:
            colour = _LEVEL_COLOURS.get(branch.level, _DEEP_COLOUR)
            ax.plot(branch.pixels[:, 1], branch.pixels[:, 0], color=colour, linewidth=1.6)
        ax.scatter([result.root_rc[1]], [result.root_rc[0]], c="black", s=45, zorder=5, label="root")

        present = sorted({branch.level for branch in result.branches})
        legend = [
            Line2D([0], [0], color=_LEVEL_COLOURS.get(level, _DEEP_COLOUR), lw=2,
                   label=f"L{level} · {_level_label(level)}")
            for level in present
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(f"{result.name}: {len(result.branches)} branches", fontsize=10)
        ax.axis("off")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def save_instance_overlay(result: PipelineResult, path: str | Path) -> Path:
        """Fill each branch region, coloured by branch order (all primaries share a
        colour, all secondaries share a colour, …)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        from matplotlib.lines import Line2D

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = result.image.astype(float) / 255.0
        overlay = image.copy()
        level_of = {index: branch.level for index, branch in enumerate(result.branches, start=1)}
        present = sorted(set(level_of.values()))
        for level in present:
            colour = np.array(to_rgb(_LEVEL_COLOURS.get(level, _DEEP_COLOUR)))
            ids = [i for i, lvl in level_of.items() if lvl == level]
            region = np.isin(result.instances, ids)
            overlay[region] = 0.45 * image[region] + 0.55 * colour

        height, width = result.image.shape[:2]
        fig, ax = plt.subplots(figsize=(9, 9 * height / width))
        ax.imshow(overlay)
        for branch in result.branches:  # centreline, now spanning its region
            ax.plot(branch.pixels[:, 1], branch.pixels[:, 0], color="black", linewidth=0.7, alpha=0.7)
        legend = [
            Line2D([0], [0], color=_LEVEL_COLOURS.get(level, _DEEP_COLOUR), lw=6,
                   label=f"L{level} · {_level_label(level)}")
            for level in present
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(f"{result.name}: branches by order", fontsize=10)
        ax.axis("off")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path


def _level_label(level: int) -> str:
    return {0: "trunk", 1: "primary", 2: "secondary", 3: "tertiary"}.get(level, f"order{level}")
