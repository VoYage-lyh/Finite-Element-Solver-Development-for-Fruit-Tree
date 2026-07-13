"""Editable, evidence-aware branch skeletons for the Harvest Console.

The automatic vision pipeline deliberately keeps :class:`~orchard_vision.types.Branch`
small.  An interactive editor needs more state: sparse control points, provenance,
undo snapshots, the wood mask used as geometric evidence, and a pixel-to-metre
transform.  This module provides that state without importing Tk, so the geometry
operations can be tested headlessly.

User-added centrelines are routed through the current wood mask and rejected when
too little of the resulting path is supported.  If the semantic mask missed a
visible limb, :meth:`EditableSkeleton.suggest_branch_mask` builds a *proposal*
inside a narrow corridor around the user's stroke.  The GUI must show that proposal
and obtain confirmation before merging it into the evidence mask.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label
from skimage.color import rgb2gray
from skimage.draw import disk, line
from skimage.filters import threshold_otsu
from skimage.graph import route_through_array
from skimage.io import imread
from skimage.morphology import disk as morphology_disk

from orchard_vision.export_skeleton import branches_to_skeleton_payload
from orchard_vision.instances import branch_instances
from orchard_vision.lift_3d import MonocularPlanarLift
from orchard_vision.types import Branch


class SkeletonEditError(ValueError):
    """An edit would make the skeleton unsupported or topologically invalid."""


@dataclass(frozen=True)
class BranchEvidence:
    support_ratio: float
    mean_radius_px: float
    min_radius_px: float

    @property
    def supported(self) -> bool:
        return self.support_ratio >= 0.85 and self.mean_radius_px >= 0.75


@dataclass
class SkeletonSnapshot:
    mask: np.ndarray
    branches: list[Branch]
    controls: dict[str, np.ndarray]
    sources: dict[str, str]


def _raster_polyline(points_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise an ``(N, 2)`` row/column polyline, removing repeated pixels."""
    points = np.asarray(points_rc, dtype=float).reshape((-1, 2))
    if len(points) < 2:
        return np.rint(points).astype(int)
    chunks: list[np.ndarray] = []
    for first, second in zip(points, points[1:]):
        r0, c0 = np.rint(first).astype(int)
        r1, c1 = np.rint(second).astype(int)
        rr, cc = line(r0, c0, r1, c1)
        chunk = np.column_stack([rr, cc])
        if chunks and len(chunk):
            chunk = chunk[1:]
        chunks.append(chunk)
    pixels = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=int)
    if not len(pixels):
        return pixels
    pixels[:, 0] = np.clip(pixels[:, 0], 0, shape[0] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, shape[1] - 1)
    keep = np.ones(len(pixels), dtype=bool)
    keep[1:] = np.any(pixels[1:] != pixels[:-1], axis=1)
    return pixels[keep]


def _rdp(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Small Ramer-Douglas-Peucker implementation for editable control points."""
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return points.copy()
    first, last = points[0], points[-1]
    direction = last - first
    length2 = float(np.dot(direction, direction))
    if length2 <= 1.0e-12:
        distances = np.linalg.norm(points - first, axis=1)
    else:
        alpha = np.clip(((points - first) @ direction) / length2, 0.0, 1.0)
        projection = first + alpha[:, None] * direction
        distances = np.linalg.norm(points - projection, axis=1)
    index = int(np.argmax(distances))
    if float(distances[index]) <= tolerance:
        return points[[0, -1]].copy()
    left = _rdp(points[: index + 1], tolerance)
    right = _rdp(points[index:], tolerance)
    return np.vstack([left[:-1], right])


def _control_points(pixels: np.ndarray, *, tolerance: float = 2.0, maximum: int = 24) -> np.ndarray:
    controls = _rdp(np.asarray(pixels, dtype=float), tolerance)
    if len(controls) <= maximum:
        return controls
    indices = np.unique(np.rint(np.linspace(0, len(controls) - 1, maximum)).astype(int))
    return controls[indices]


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points.astype(float), axis=0), axis=1).sum())


@dataclass
class EditableSkeleton:
    """Mutable skeleton plus the image evidence needed to justify every branch."""

    image: np.ndarray
    mask: np.ndarray
    branches: list[Branch]
    root_rc: tuple[int, int]
    lift: MonocularPlanarLift
    model_name: str
    image_path: Path | None = None
    controls: dict[str, np.ndarray] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    minimum_support: float = 0.85
    radius_map: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)[..., :3]
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.shape != self.image.shape[:2]:
            raise SkeletonEditError("Wood mask and source image dimensions do not match")
        self.branches = [copy.deepcopy(branch) for branch in self.branches]
        for branch in self.branches:
            self.controls.setdefault(branch.id, _control_points(branch.pixels))
            self.sources.setdefault(branch.id, "automatic")
        self._refresh_radius_map()
        self._normalize_existing_attachments()

    @classmethod
    def from_pipeline_result(
        cls,
        result: Any,
        *,
        image_path: str | Path | None = None,
    ) -> "EditableSkeleton":
        return cls(
            image=result.image,
            mask=result.mask,
            branches=result.branches,
            root_rc=result.root_rc,
            lift=result.lift,
            model_name=result.name,
            image_path=Path(image_path).resolve() if image_path else None,
        )

    def snapshot(self) -> SkeletonSnapshot:
        return SkeletonSnapshot(
            mask=self.mask.copy(),
            branches=copy.deepcopy(self.branches),
            controls={key: value.copy() for key, value in self.controls.items()},
            sources=dict(self.sources),
        )

    def restore(self, snapshot: SkeletonSnapshot) -> None:
        self.mask = snapshot.mask.copy()
        self.branches = copy.deepcopy(snapshot.branches)
        self.controls = {key: value.copy() for key, value in snapshot.controls.items()}
        self.sources = dict(snapshot.sources)
        self._refresh_radius_map()

    def _refresh_radius_map(self) -> None:
        self.radius_map = distance_transform_edt(self.mask).astype(np.float32)
        for branch in self.branches:
            if len(branch.pixels):
                rows = np.clip(branch.pixels[:, 0], 0, self.mask.shape[0] - 1)
                cols = np.clip(branch.pixels[:, 1], 0, self.mask.shape[1] - 1)
                branch.radius_px = self.radius_map[rows, cols]

    def _normalize_existing_attachments(self) -> None:
        """Route small automatic junction gaps onto the actual parent centreline.

        Junction-cluster contraction in the automatic ordering stage can leave a
        child's first pixel several pixels from the merged parent path.  Export used
        to hide that gap by snapping only the down-sampled metric point.  The editor
        instead repairs the dense pixel geometry up front, using the same evidence
        rules as a manual node move.
        """
        for branch in self.branches:
            if branch.parent_id is None:
                continue
            controls = self.controls[branch.id].copy()
            snapped = self._nearest_parent_pixel(branch.parent_id, controls[0])
            if float(np.linalg.norm(snapped - controls[0])) <= 0.5:
                continue
            proposed = controls.copy()
            proposed[0] = snapped
            try:
                self.update_branch(branch.id, proposed, mark_edited=False)
            except SkeletonEditError:
                # Preserve the automatic result so validation can surface the
                # unresolved attachment instead of silently deleting geometry.
                continue

    def branch(self, branch_id: str) -> Branch:
        for branch in self.branches:
            if branch.id == branch_id:
                return branch
        raise SkeletonEditError(f"Unknown branch: {branch_id}")

    def child_ids(self, branch_id: str) -> list[str]:
        return [branch.id for branch in self.branches if branch.parent_id == branch_id]

    def descendant_ids(self, branch_id: str) -> set[str]:
        descendants: set[str] = set()
        pending = self.child_ids(branch_id)
        while pending:
            child = pending.pop()
            if child in descendants:
                continue
            descendants.add(child)
            pending.extend(self.child_ids(child))
        return descendants

    def evidence(self, branch_id: str) -> BranchEvidence:
        branch = self.branch(branch_id)
        if not len(branch.pixels):
            return BranchEvidence(0.0, 0.0, 0.0)
        rows = np.clip(branch.pixels[:, 0], 0, self.mask.shape[0] - 1)
        cols = np.clip(branch.pixels[:, 1], 0, self.mask.shape[1] - 1)
        supported = self.mask[rows, cols]
        radii = self.radius_map[rows, cols]
        return BranchEvidence(
            support_ratio=float(np.mean(supported)),
            mean_radius_px=float(np.mean(radii)),
            min_radius_px=float(np.min(radii)),
        )

    def length_px(self, branch_id: str) -> float:
        return _polyline_length(self.branch(branch_id).pixels)

    def length_m(self, branch_id: str) -> float:
        return self.length_px(branch_id) * self.lift.meters_per_pixel

    def _nearest_parent_pixel(self, parent_id: str, point_rc: np.ndarray) -> np.ndarray:
        parent = self.branch(parent_id)
        if not len(parent.pixels):
            raise SkeletonEditError(f"Parent branch '{parent_id}' has no centreline")
        index = int(np.argmin(np.linalg.norm(parent.pixels.astype(float) - point_rc, axis=1)))
        return parent.pixels[index].astype(float)

    def _route_controls(self, controls: np.ndarray, *, corridor_px: int = 24) -> np.ndarray:
        controls = np.asarray(controls, dtype=float).reshape((-1, 2))
        if len(controls) < 2:
            raise SkeletonEditError("A branch needs at least two control points")
        height, width = self.mask.shape
        controls[:, 0] = np.clip(controls[:, 0], 0, height - 1)
        controls[:, 1] = np.clip(controls[:, 1], 0, width - 1)

        outside_distance = distance_transform_edt(~self.mask)
        cost = 1.0 + 3.0 / (self.radius_map.astype(float) + 1.0)
        cost += np.where(self.mask, 0.0, 35.0 + 8.0 * outside_distance)

        routed: list[np.ndarray] = []
        pad = max(int(corridor_px), 4)
        for first, second in zip(controls, controls[1:]):
            start = np.rint(first).astype(int)
            end = np.rint(second).astype(int)
            r0 = max(min(start[0], end[0]) - pad, 0)
            r1 = min(max(start[0], end[0]) + pad + 1, height)
            c0 = max(min(start[1], end[1]) - pad, 0)
            c1 = min(max(start[1], end[1]) + pad + 1, width)
            local = cost[r0:r1, c0:c1].copy()

            # Keep the geodesic close to the user's segment.  This prevents a new
            # line from jumping onto an unrelated, parallel branch in the mask.
            guide = np.zeros(local.shape, dtype=bool)
            rr, cc = line(start[0] - r0, start[1] - c0, end[0] - r0, end[1] - c0)
            guide[rr, cc] = True
            guide_distance = distance_transform_edt(~guide)
            local += 0.15 * guide_distance

            try:
                path, _ = route_through_array(
                    local,
                    (int(start[0] - r0), int(start[1] - c0)),
                    (int(end[0] - r0), int(end[1] - c0)),
                    fully_connected=True,
                    geometric=True,
                )
            except Exception as error:  # noqa: BLE001 - convert skimage detail to edit error
                raise SkeletonEditError(f"Could not trace branch through the wood mask: {error}") from error
            chunk = np.asarray(path, dtype=int)
            chunk[:, 0] += r0
            chunk[:, 1] += c0
            if routed:
                chunk = chunk[1:]
            routed.append(chunk)
        pixels = np.vstack(routed) if routed else _raster_polyline(controls, self.mask.shape)
        if len(pixels) < 2:
            raise SkeletonEditError("The edited branch has zero length")
        return pixels

    def update_branch(
        self,
        branch_id: str,
        controls: np.ndarray,
        *,
        mark_edited: bool = True,
    ) -> BranchEvidence:
        branch = self.branch(branch_id)
        proposed = np.asarray(controls, dtype=float).copy().reshape((-1, 2))
        if branch.parent_id is not None:
            proposed[0] = self._nearest_parent_pixel(branch.parent_id, proposed[0])
        drawn = _raster_polyline(proposed, self.mask.shape)
        drawn_support = float(np.mean(self.mask[drawn[:, 0], drawn[:, 1]]))
        if drawn_support < self.minimum_support:
            raise SkeletonEditError(
                f"Only {drawn_support:.0%} of the edited stroke for '{branch_id}' is supported "
                f"by the wood mask; at least {self.minimum_support:.0%} is required"
            )
        pixels = self._route_controls(proposed)
        rows, cols = pixels[:, 0], pixels[:, 1]
        evidence = BranchEvidence(
            support_ratio=float(np.mean(self.mask[rows, cols])),
            mean_radius_px=float(np.mean(self.radius_map[rows, cols])),
            min_radius_px=float(np.min(self.radius_map[rows, cols])),
        )
        if evidence.support_ratio < self.minimum_support:
            raise SkeletonEditError(
                f"Only {evidence.support_ratio:.0%} of '{branch_id}' is supported by the wood mask; "
                f"at least {self.minimum_support:.0%} is required"
            )
        branch.pixels = pixels
        branch.radius_px = self.radius_map[rows, cols]
        self.controls[branch_id] = proposed
        if mark_edited and self.sources.get(branch_id, "automatic") == "automatic":
            self.sources[branch_id] = "user_edited"
        return evidence

    def _next_id(self, level: int) -> str:
        name = {0: "trunk", 1: "primary", 2: "secondary", 3: "tertiary"}.get(
            level, f"order{level}"
        )
        if level == 0 and all(branch.id != "trunk" for branch in self.branches):
            return "trunk"
        used = {branch.id for branch in self.branches}
        index = 1
        while f"{name}_{index}" in used:
            index += 1
        return f"{name}_{index}"

    def add_branch(
        self,
        parent_id: str,
        controls: np.ndarray,
        *,
        branch_id: str | None = None,
    ) -> str:
        parent = self.branch(parent_id)
        proposed = np.asarray(controls, dtype=float).copy().reshape((-1, 2))
        if len(proposed) < 2:
            raise SkeletonEditError("Draw at least two points for a new branch")
        proposed[0] = self._nearest_parent_pixel(parent_id, proposed[0])
        drawn = _raster_polyline(proposed, self.mask.shape)
        drawn_support = float(np.mean(self.mask[drawn[:, 0], drawn[:, 1]]))
        if drawn_support < self.minimum_support:
            raise SkeletonEditError(
                f"The drawn branch has only {drawn_support:.0%} direct wood-mask support; "
                "confirm or paint its branch region before adding it"
            )
        pixels = self._route_controls(proposed)
        rows, cols = pixels[:, 0], pixels[:, 1]
        support = float(np.mean(self.mask[rows, cols]))
        if support < self.minimum_support:
            raise SkeletonEditError(
                f"The proposed branch has only {support:.0%} wood-mask support; "
                "confirm or paint its branch region before adding it"
            )
        level_value = parent.level + 1
        new_id = branch_id or self._next_id(level_value)
        if any(branch.id == new_id for branch in self.branches):
            raise SkeletonEditError(f"Branch id already exists: {new_id}")
        self.branches.append(
            Branch(
                id=new_id,
                parent_id=parent_id,
                level=level_value,
                pixels=pixels,
                radius_px=self.radius_map[rows, cols],
            )
        )
        self.controls[new_id] = proposed
        self.sources[new_id] = "user_added"
        return new_id

    def delete_branch(self, branch_id: str, *, cascade: bool = True) -> list[str]:
        branch = self.branch(branch_id)
        if branch.parent_id is None:
            raise SkeletonEditError("The trunk/root branch cannot be deleted")
        descendants = self.descendant_ids(branch_id)
        if descendants and not cascade:
            raise SkeletonEditError("The branch has descendants; reparent them or use cascade deletion")
        removed = {branch_id, *descendants}
        self.branches = [candidate for candidate in self.branches if candidate.id not in removed]
        for candidate in removed:
            self.controls.pop(candidate, None)
            self.sources.pop(candidate, None)
        return sorted(removed)

    def _update_descendant_levels(self, parent_id: str) -> None:
        parent = self.branch(parent_id)
        for child_id in self.child_ids(parent_id):
            child = self.branch(child_id)
            child.level = parent.level + 1
            self._update_descendant_levels(child_id)

    def set_parent(self, branch_id: str, parent_id: str) -> None:
        branch = self.branch(branch_id)
        if branch.parent_id is None:
            raise SkeletonEditError("The trunk/root branch cannot be reparented")
        if parent_id == branch_id or parent_id in self.descendant_ids(branch_id):
            raise SkeletonEditError("Reparenting would create a topology cycle")
        self.branch(parent_id)  # existence check
        old_parent, old_level = branch.parent_id, branch.level
        old_controls = self.controls[branch_id].copy()
        branch.parent_id = parent_id
        branch.level = self.branch(parent_id).level + 1
        proposed = old_controls.copy()
        proposed[0] = self._nearest_parent_pixel(parent_id, proposed[0])
        try:
            self.update_branch(branch_id, proposed)
            self._update_descendant_levels(branch_id)
        except Exception:
            branch.parent_id, branch.level = old_parent, old_level
            self.controls[branch_id] = old_controls
            raise

    def paint_mask(self, points_rc: Iterable[tuple[float, float]], radius_px: int, value: bool) -> None:
        radius_value = max(int(radius_px), 1)
        for row, col in points_rc:
            rr, cc = disk((int(round(row)), int(round(col))), radius_value, shape=self.mask.shape)
            self.mask[rr, cc] = bool(value)

    def finish_mask_edit(self) -> None:
        self._refresh_radius_map()

    def suggest_branch_mask(self, controls: np.ndarray, *, half_width_px: int = 18) -> np.ndarray:
        """Return a line-guided visible-wood proposal; never mutates ``self.mask``."""
        guide = np.zeros(self.mask.shape, dtype=bool)
        pixels = _raster_polyline(np.asarray(controls, dtype=float), self.mask.shape)
        if len(pixels) < 2:
            return guide
        guide[pixels[:, 0], pixels[:, 1]] = True
        corridor = distance_transform_edt(~guide) <= max(int(half_width_px), 2)
        gray = rgb2gray(self.image)
        values = gray[corridor]
        if not values.size:
            return guide
        try:
            threshold = min(float(threshold_otsu(values)), 0.92)
        except ValueError:
            threshold = 0.92
        candidate = corridor & (gray <= threshold)
        candidate = binary_closing(candidate, structure=morphology_disk(2))

        # Retain only components touched by a small dilation of the user's stroke.
        components, _ = label(candidate)
        touched = np.unique(components[distance_transform_edt(~guide) <= 2.0])
        touched = touched[touched != 0]
        proposal = np.isin(components, touched) if len(touched) else np.zeros_like(candidate)
        return proposal & corridor

    def merge_mask(self, proposal: np.ndarray) -> None:
        proposal = np.asarray(proposal, dtype=bool)
        if proposal.shape != self.mask.shape:
            raise SkeletonEditError("Mask proposal dimensions do not match the image")
        self.mask |= proposal
        self._refresh_radius_map()

    def instance_labels(self) -> np.ndarray:
        return branch_instances(self.mask, self.branches)

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        by_id: dict[str, Branch] = {}
        for branch in self.branches:
            if branch.id in by_id:
                errors.append(f"Duplicate branch id: {branch.id}")
            by_id[branch.id] = branch
            if len(branch.pixels) < 2 or branch.length_px <= 0.0:
                errors.append(f"Branch '{branch.id}' has zero length")
            evidence = self.evidence(branch.id)
            if evidence.support_ratio < self.minimum_support:
                errors.append(
                    f"Branch '{branch.id}' has only {evidence.support_ratio:.0%} wood-mask support"
                )
        roots = [branch for branch in self.branches if branch.parent_id is None]
        if len(roots) != 1:
            errors.append(f"Expected exactly one trunk/root branch, found {len(roots)}")
        for branch in self.branches:
            if branch.parent_id is None:
                if branch.level != 0:
                    errors.append(f"Root branch '{branch.id}' must have level 0")
                continue
            parent = by_id.get(branch.parent_id)
            if parent is None:
                errors.append(f"Branch '{branch.id}' references missing parent '{branch.parent_id}'")
                continue
            if branch.level != parent.level + 1:
                errors.append(
                    f"Branch '{branch.id}' level {branch.level} is inconsistent with parent "
                    f"'{parent.id}' level {parent.level}"
                )
            root_distance = float(
                np.min(np.linalg.norm(parent.pixels.astype(float) - branch.pixels[0], axis=1))
            )
            if root_distance > 1.5:
                errors.append(
                    f"Branch '{branch.id}' root is {root_distance:.1f}px away from parent centreline"
                )
        for branch in self.branches:
            if branch.id in self.descendant_ids(branch.id):
                errors.append(f"Topology cycle detected at '{branch.id}'")
        return errors

    def require_valid(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise SkeletonEditError("Skeleton validation failed:\n- " + "\n- ".join(errors))

    def to_skeleton_payload(self) -> dict[str, Any]:
        self.require_valid()
        payload = branches_to_skeleton_payload(
            self.branches,
            self.lift,
            model_name=self.model_name,
        )
        payload["metadata"].update(
            {
                "source": "orchard_vision.harvest_console_editor",
                "source_image": str(self.image_path) if self.image_path else "",
                "vision_transform": {
                    "meters_per_pixel": float(self.lift.meters_per_pixel),
                    "base_pixel_rc": [int(self.lift.base_row), int(self.lift.base_col)],
                    "image_shape": [int(self.mask.shape[0]), int(self.mask.shape[1])],
                },
                "edited": True,
            }
        )
        for record, branch in zip(payload["branches"], self.branches):
            evidence = self.evidence(branch.id)
            record["edit_source"] = self.sources.get(branch.id, "automatic")
            record["image_support_ratio"] = evidence.support_ratio
        return payload

    def save_project(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        array_path = path.with_suffix(path.suffix + ".npz")
        arrays: dict[str, np.ndarray] = {
            "image": np.asarray(self.image),
            "mask": self.mask.astype(np.uint8),
        }
        branch_records: list[dict[str, Any]] = []
        for index, branch in enumerate(self.branches):
            prefix = f"branch_{index}"
            arrays[f"{prefix}_pixels"] = np.asarray(branch.pixels, dtype=np.int32)
            arrays[f"{prefix}_radius"] = np.asarray(branch.radius_px, dtype=np.float32)
            arrays[f"{prefix}_controls"] = np.asarray(self.controls[branch.id], dtype=np.float32)
            branch_records.append(
                {
                    "id": branch.id,
                    "parent_id": branch.parent_id,
                    "level": int(branch.level),
                    "source": self.sources.get(branch.id, "automatic"),
                    "array_prefix": prefix,
                }
            )
        np.savez_compressed(array_path, **arrays)
        manifest = {
            "format": "orchard_skeleton_edit_project",
            "version": 1,
            "model_name": self.model_name,
            "image_path": str(self.image_path) if self.image_path else "",
            "arrays": array_path.name,
            "root_rc": [int(self.root_rc[0]), int(self.root_rc[1])],
            "lift": {
                "meters_per_pixel": float(self.lift.meters_per_pixel),
                "base_row": int(self.lift.base_row),
                "base_col": int(self.lift.base_col),
            },
            "minimum_support": float(self.minimum_support),
            "branches": branch_records,
        }
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load_project(cls, path: str | Path) -> "EditableSkeleton":
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != "orchard_skeleton_edit_project":
            raise SkeletonEditError("Not an Orchard skeleton edit project")
        image_path = Path(manifest["image_path"])
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        array_path = path.parent / manifest["arrays"]
        with np.load(array_path, allow_pickle=False) as arrays:
            # Version-1 projects persist the *working-resolution* image.  The
            # source path may point at a larger original photo, whose dimensions
            # no longer match the edited mask after pipeline down-scaling.
            if "image" in arrays:
                image = arrays["image"].copy()
            else:  # compatibility with early development projects
                image = imread(image_path)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
            mask = arrays["mask"].astype(bool)
            branches: list[Branch] = []
            controls: dict[str, np.ndarray] = {}
            sources: dict[str, str] = {}
            for record in manifest["branches"]:
                prefix = record["array_prefix"]
                branch = Branch(
                    id=str(record["id"]),
                    parent_id=record.get("parent_id"),
                    level=int(record["level"]),
                    pixels=arrays[f"{prefix}_pixels"].astype(int),
                    radius_px=arrays[f"{prefix}_radius"].astype(np.float32),
                )
                branches.append(branch)
                controls[branch.id] = arrays[f"{prefix}_controls"].astype(float)
                sources[branch.id] = str(record.get("source", "automatic"))
        lift_data = manifest["lift"]
        return cls(
            image=image,
            mask=mask,
            branches=branches,
            root_rc=(int(manifest["root_rc"][0]), int(manifest["root_rc"][1])),
            lift=MonocularPlanarLift(
                meters_per_pixel=float(lift_data["meters_per_pixel"]),
                base_row=int(lift_data["base_row"]),
                base_col=int(lift_data["base_col"]),
            ),
            model_name=str(manifest["model_name"]),
            image_path=image_path,
            controls=controls,
            sources=sources,
            minimum_support=float(manifest.get("minimum_support", 0.85)),
        )


__all__ = [
    "BranchEvidence",
    "EditableSkeleton",
    "SkeletonEditError",
    "SkeletonSnapshot",
]
