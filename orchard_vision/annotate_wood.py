"""SAM 2-assisted field-image wood annotator.

The annotation unit is one *target tree* in an untouched field photograph:

1. draw a target-tree ROI so neighbouring trees cannot leak into the label;
2. click the target trunk root (later used to select the connected skeleton);
3. box visible limbs for SAM 2 proposals and commit the good ones;
4. use the brush to repair thin branches or erase leaves/background.

The tool resumes existing labels instead of starting from a blank mask.  It saves
the working-resolution RGB image, a binary ``*_wood.png`` mask and a
``*_wood.json`` metadata record containing the source path, scale, target ROI and
trunk-root point.  Fully occluded wood must not be invented; label visible wood
only and let the downstream editor mark uncertain gaps.

Controls (buttons provide the same operations):
    r                 target-tree ROI mode; drag one box around the target tree
    b                 branch-box mode; drag a box around one visible limb for SAM
    m                 brush mode; left paints visible wood, right erases
    t                 trunk-root mode; click the target trunk base
    c / x             commit / discard current SAM proposal
    z / y             undo / redo
    [ / ]             smaller / larger brush
    s / n / q         save / save+next / save+quit

Run from a desktop session (WSLg/X/Windows), not a headless shell.
"""
from __future__ import annotations

import os

# Must precede any CUDA initialisation performed by ultralytics/torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from skimage.draw import disk  # noqa: E402
from skimage.io import imread, imsave  # noqa: E402
from skimage.transform import rescale  # noqa: E402

from orchard_fem.workspace import workspace_paths  # noqa: E402

_MIN_BOX_PX = 6
_FORMAT = "orchard_visible_wood_annotation"
_VERSION = 2
_VALID_MODES = {"roi", "box", "brush", "root"}


class WoodAnnotator:
    """Target-tree ROI + SAM branch boxes + brush, with resumable labels."""

    def __init__(
        self,
        image_paths: list[Path],
        out_dir: Path,
        *,
        checkpoint: str | None = None,
        device: str = "cuda:1",
        memory_fraction: float = 0.7,
        work_dim: int = 1024,
        brush_radius: int = 4,
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = checkpoint or str(workspace_paths().sam_checkpoint)
        self.device = device
        self.memory_fraction = memory_fraction
        self.work_dim = work_dim
        self.brush_radius = brush_radius

        self._model = None
        self._index = 0
        self.mode = "roi"
        self._painting = False
        self._erasing = False
        self._dirty = False
        self._undo: list[dict[str, Any]] = []
        self._redo: list[dict[str, Any]] = []
        self._buttons: list[Any] = []
        self._reset_image_state()

    # ---- per-image state ------------------------------------------------------
    def _reset_image_state(self) -> None:
        self.image: np.ndarray | None = None
        self.wood: np.ndarray | None = None
        self.proposal: np.ndarray | None = None
        self.target_roi_xyxy: tuple[int, int, int, int] | None = None
        self.trunk_root_rc: tuple[int, int] | None = None
        self.source_shape: tuple[int, ...] | None = None
        self.work_scale = 1.0
        self._painting = False
        self._dirty = False
        self._undo.clear()
        self._redo.clear()

    def _annotation_paths(self) -> tuple[Path, Path, Path]:
        stem = self.image_paths[self._index].stem
        return (
            self.out_dir / f"{stem}.png",
            self.out_dir / f"{stem}_wood.png",
            self.out_dir / f"{stem}_wood.json",
        )

    @staticmethod
    def _as_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image[..., :3]
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
            image = image * scale
        return np.clip(image, 0, 255).astype(np.uint8)

    def _load_image(self, path: Path) -> np.ndarray:
        image = self._as_uint8(imread(path))
        self.source_shape = tuple(int(value) for value in image.shape)
        longest = max(image.shape[:2])
        if longest > self.work_dim:
            self.work_scale = self.work_dim / longest
            image = rescale(
                image,
                self.work_scale,
                channel_axis=-1,
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.uint8)
        else:
            self.work_scale = 1.0
        return image

    def _load_saved_annotation(self) -> bool:
        """Load a compatible mask/ROI/root for the current source image."""
        if self.image is None or self.wood is None:
            return False
        _, mask_path, metadata_path = self._annotation_paths()
        loaded = False
        if mask_path.exists():
            candidate = imread(mask_path) > 127
            if candidate.shape == self.wood.shape:
                self.wood[:] = candidate
                loaded = True
                print(f"[resume] loaded {mask_path.name} ({candidate.mean() * 100:.2f}% wood)")
            else:
                print(
                    f"[warn] cannot resume {mask_path.name}: mask shape {candidate.shape} "
                    f"!= working image {self.wood.shape}"
                )
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                roi = metadata.get("target_roi_xyxy_work")
                root = metadata.get("trunk_root_rc_work")
                if roi is not None and len(roi) == 4:
                    self.target_roi_xyxy = self._clamp_saved_roi(
                        *[float(value) for value in roi]
                    )
                if root is not None and len(root) == 2:
                    row = int(np.clip(round(float(root[0])), 0, self.wood.shape[0] - 1))
                    col = int(np.clip(round(float(root[1])), 0, self.wood.shape[1] - 1))
                    self.trunk_root_rc = (row, col)
            except Exception as error:  # noqa: BLE001 - a bad sidecar must not lose the mask
                print(f"[warn] could not read {metadata_path.name}: {error}")
        return loaded

    # ---- history --------------------------------------------------------------
    def _snapshot(self) -> dict[str, Any]:
        return {
            "wood": None if self.wood is None else self.wood.copy(),
            "proposal": None if self.proposal is None else self.proposal.copy(),
            "roi": self.target_roi_xyxy,
            "root": self.trunk_root_rc,
        }

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot["wood"] is not None:
            self.wood = snapshot["wood"].copy()
        self.proposal = (
            None if snapshot["proposal"] is None else snapshot["proposal"].copy()
        )
        self.target_roi_xyxy = snapshot["roi"]
        self.trunk_root_rc = snapshot["root"]

    def _record_edit(self) -> None:
        self._undo.append(self._snapshot())
        if len(self._undo) > 50:
            self._undo.pop(0)
        self._redo.clear()
        self._dirty = True

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._snapshot())
        self._restore_snapshot(self._undo.pop())
        self._dirty = True
        self._redraw()

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._snapshot())
        self._restore_snapshot(self._redo.pop())
        self._dirty = True
        self._redraw()

    # ---- target tree ----------------------------------------------------------
    def _clamp_roi(self, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        if self.image is None:
            raise ValueError("No image is open")
        height, width = self.image.shape[:2]
        left = int(np.clip(np.floor(min(x0, x1)), 0, width - 1))
        top = int(np.clip(np.floor(min(y0, y1)), 0, height - 1))
        right = int(np.clip(np.ceil(max(x0, x1)) + 1, left + 1, width))
        bottom = int(np.clip(np.ceil(max(y0, y1)) + 1, top + 1, height))
        return (left, top, right, bottom)

    def _clamp_saved_roi(
        self, left: float, top: float, right: float, bottom: float
    ) -> tuple[int, int, int, int]:
        """Clamp a saved half-open ROI without expanding it on every reload."""
        if self.image is None:
            raise ValueError("No image is open")
        height, width = self.image.shape[:2]
        left_i = int(np.clip(np.floor(left), 0, width - 1))
        top_i = int(np.clip(np.floor(top), 0, height - 1))
        right_i = int(np.clip(np.ceil(right), left_i + 1, width))
        bottom_i = int(np.clip(np.ceil(bottom), top_i + 1, height))
        return (left_i, top_i, right_i, bottom_i)

    def _roi_mask(self) -> np.ndarray:
        if self.image is None:
            raise ValueError("No image is open")
        mask = np.ones(self.image.shape[:2], dtype=bool)
        if self.target_roi_xyxy is None:
            return mask
        mask[:] = False
        left, top, right, bottom = self.target_roi_xyxy
        mask[top:bottom, left:right] = True
        return mask

    def set_target_roi(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if abs(x1 - x0) < _MIN_BOX_PX or abs(y1 - y0) < _MIN_BOX_PX:
            return
        self._record_edit()
        self.target_roi_xyxy = self._clamp_roi(x0, y0, x1, y1)
        roi = self._roi_mask()
        if self.wood is not None:
            removed = int(np.count_nonzero(self.wood & ~roi))
            self.wood &= roi
            if removed:
                print(f"[roi] removed {removed} labelled pixels outside the target tree")
        if self.proposal is not None:
            self.proposal &= roi
        if self.trunk_root_rc is not None and not roi[self.trunk_root_rc]:
            self.trunk_root_rc = None
        self._set_mode("box")

    def set_trunk_root(self, row: float, col: float) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        rc = (
            int(np.clip(round(row), 0, height - 1)),
            int(np.clip(round(col), 0, width - 1)),
        )
        if not self._roi_mask()[rc]:
            print("[warn] trunk root must lie inside the target-tree ROI")
            return
        self._record_edit()
        self.trunk_root_rc = rc
        self._set_mode("box")

    # ---- SAM 2 ----------------------------------------------------------------
    def _load_model(self):
        if self._model is None:
            from ultralytics import SAM
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                index = int(self.device.split(":")[1]) if ":" in self.device else 0
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, index)
            else:
                self.device = "cpu"
            self._model = SAM(self.checkpoint)
        return self._model

    def propose_from_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self.image is None:
            return
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        results = self._load_model()(self.image, bboxes=[box], device=self.device, verbose=False)
        masks = results[0].masks
        proposal = masks.data.cpu().numpy().any(axis=0) if masks is not None else None
        if proposal is not None:
            proposal &= self._roi_mask()
        self.proposal = proposal

    def commit_proposal(self) -> None:
        if self.proposal is None or self.wood is None:
            return
        self._record_edit()
        self.wood |= self.proposal & self._roi_mask()
        self.proposal = None
        self._redraw()

    def discard_proposal(self) -> None:
        self.proposal = None
        self._redraw()

    # ---- brush ----------------------------------------------------------------
    def _paint(self, x: float, y: float, erase: bool) -> None:
        if self.wood is None:
            return
        rr, cc = disk((int(y), int(x)), self.brush_radius, shape=self.wood.shape)
        inside = self._roi_mask()[rr, cc]
        rr, cc = rr[inside], cc[inside]
        self.wood[rr, cc] = not erase

    # ---- persistence ----------------------------------------------------------
    def _metadata(self) -> dict[str, Any]:
        if self.image is None:
            raise ValueError("No image is open")
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "source_image": str(self.image_paths[self._index].resolve()),
            "source_shape": list(self.source_shape or self.image.shape),
            "working_shape": list(self.image.shape),
            "source_to_work_scale": float(self.work_scale),
            "target_roi_xyxy_work": (
                list(self.target_roi_xyxy) if self.target_roi_xyxy is not None else None
            ),
            "trunk_root_rc_work": (
                list(self.trunk_root_rc) if self.trunk_root_rc is not None else None
            ),
            "label_policy": {
                "target": "visible wood of the selected target tree only",
                "occluded_wood": "do not invent fully occluded branches",
                "valid_region": "target_roi_xyxy_work only",
                "inside_roi_unlabelled": "background / negative",
                "outside_roi": "ignore during training",
            },
        }

    def save(self) -> None:
        if self.image is None or self.wood is None:
            return
        image_path, mask_path, metadata_path = self._annotation_paths()
        if not self.wood.any() and mask_path.exists() and (imread(mask_path) > 127).any():
            print(f"[skip] {mask_path.name}: refusing to replace a non-empty label with blank")
            return
        if not self.wood.any():
            print("[warn] saving a blank wood mask")
        if self.target_roi_xyxy is None:
            print("[warn] no target-tree ROI; field labels should define one with [r]")
        if self.trunk_root_rc is None:
            print("[warn] no trunk root; click it with [t] before finishing the image")
        imsave(image_path, self.image, check_contrast=False)
        imsave(mask_path, self.wood.astype(np.uint8) * 255, check_contrast=False)
        metadata_path.write_text(
            json.dumps(self._metadata(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._dirty = False
        print(
            f"saved {self.image_paths[self._index].stem}: "
            f"wood={self.wood.mean() * 100:.2f}% ROI={self.target_roi_xyxy} "
            f"root={self.trunk_root_rc}"
        )

    # ---- rendering ------------------------------------------------------------
    def _redraw(self) -> None:
        if not hasattr(self, "ax") or self.image is None or self.wood is None:
            return
        self.ax.clear()
        self.ax.imshow(self.image)
        if self.wood.any():
            layer = np.zeros((*self.wood.shape, 4))
            layer[self.wood] = (0.2, 0.9, 0.3, 0.45)
            self.ax.imshow(layer)
        if self.proposal is not None and self.proposal.any():
            layer = np.zeros((*self.proposal.shape, 4))
            layer[self.proposal] = (0.95, 0.2, 0.2, 0.40)
            self.ax.imshow(layer)
        if self.target_roi_xyxy is not None:
            from matplotlib.patches import Rectangle

            left, top, right, bottom = self.target_roi_xyxy
            self.ax.add_patch(
                Rectangle(
                    (left, top),
                    right - left,
                    bottom - top,
                    fill=False,
                    edgecolor="#ffd600",
                    linewidth=2.0,
                    linestyle="--",
                )
            )
        if self.trunk_root_rc is not None:
            row, col = self.trunk_root_rc
            self.ax.scatter([col], [row], c="#00e5ff", edgecolors="black", s=75, zorder=8)
        name = self.image_paths[self._index].name
        resumed = " resumed" if self._undo or self.wood.any() else ""
        self.ax.set_title(
            f"{name} ({self._index + 1}/{len(self.image_paths)})  mode={self.mode}{resumed}  "
            f"brush={self.brush_radius}px\n"
            "r=target ROI  b=branch box  m=brush(L paint/R erase)  t=trunk root  "
            "c/x=commit/drop  z/y=undo/redo  s/n/q=save/next/quit",
            fontsize=9,
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _open_current(self) -> None:
        self._reset_image_state()
        self.image = self._load_image(self.image_paths[self._index])
        self.wood = np.zeros(self.image.shape[:2], dtype=bool)
        self._load_saved_annotation()
        self.mode = "box" if self.target_roi_xyxy is not None else "roi"
        if hasattr(self, "_selector"):
            self._selector.set_active(self.mode in {"roi", "box"})
        self._redraw()

    # ---- events ----------------------------------------------------------------
    def _set_mode(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown annotation mode: {mode}")
        self.mode = mode
        if hasattr(self, "_selector"):
            self._selector.set_active(mode in {"roi", "box"})
        self._redraw()

    def _on_box(self, eclick, erelease) -> None:
        if self.mode not in {"roi", "box"}:
            return
        if eclick.xdata is None or erelease.xdata is None:
            return
        if abs(erelease.xdata - eclick.xdata) < _MIN_BOX_PX:
            return
        if abs(erelease.ydata - eclick.ydata) < _MIN_BOX_PX:
            return
        if self.mode == "roi":
            self.set_target_roi(eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        else:
            self.propose_from_box(eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
            self._redraw()

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if self.mode == "root":
            self.set_trunk_root(event.ydata, event.xdata)
            return
        if self.mode != "brush":
            return
        self._record_edit()
        self._painting = True
        self._erasing = event.button != 1
        self._paint(event.xdata, event.ydata, self._erasing)
        self._redraw()

    def _on_motion(self, event) -> None:
        if not self._painting or event.inaxes is not self.ax or event.xdata is None:
            return
        self._paint(event.xdata, event.ydata, self._erasing)
        self._redraw()

    def _on_release(self, _event) -> None:
        self._painting = False

    def _advance(self, *, quit_after: bool = False) -> None:
        self.save()
        if quit_after or self._index + 1 >= len(self.image_paths):
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            return
        self._index += 1
        self._open_current()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "r":
            self._set_mode("roi")
        elif key == "b":
            self._set_mode("box")
        elif key == "m":
            self._set_mode("brush")
        elif key == "t":
            self._set_mode("root")
        elif key == "c":
            self.commit_proposal()
        elif key == "x":
            self.discard_proposal()
        elif key == "z":
            self.undo()
        elif key == "y":
            self.redo()
        elif key == "]":
            self.brush_radius = min(self.brush_radius + 1, 60)
        elif key == "[":
            self.brush_radius = max(self.brush_radius - 1, 1)
        elif key == "s":
            self.save()
        elif key == "n":
            self._advance()
            return
        elif key == "q":
            self._advance(quit_after=True)
            return
        self._redraw()

    def _add_button(self, fig, label: str, x: float, callback) -> None:
        from matplotlib.widgets import Button

        axes = fig.add_axes([x, 0.025, 0.095, 0.045])
        button = Button(axes, label)
        button.on_clicked(callback)
        self._buttons.append(button)

    def run(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector

        for keymap in ("keymap.save", "keymap.quit", "keymap.quit_all"):
            plt.rcParams[keymap] = []
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.subplots_adjust(bottom=0.11)
        self._selector = RectangleSelector(
            self.ax,
            self._on_box,
            useblit=True,
            button=[1],
            minspanx=_MIN_BOX_PX,
            minspany=_MIN_BOX_PX,
        )
        buttons = [
            ("Target ROI", lambda _event: self._set_mode("roi")),
            ("Branch box", lambda _event: self._set_mode("box")),
            ("Brush", lambda _event: self._set_mode("brush")),
            ("Trunk root", lambda _event: self._set_mode("root")),
            ("Commit", lambda _event: self.commit_proposal()),
            ("Undo", lambda _event: self.undo()),
            ("Save", lambda _event: self.save()),
            ("Next", lambda _event: self._advance()),
        ]
        for index, (label, callback) in enumerate(buttons):
            self._add_button(self.fig, label, 0.02 + index * 0.12, callback)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._open_current()
        plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Annotate visible wood for one selected target tree in field photos."
    )
    paths = workspace_paths()
    parser.add_argument("inputs", nargs="+", help="Raw field photo path(s)")
    parser.add_argument(
        "--out-dir",
        default=str(paths.wood_annotations),
        help="Output dataset directory (default: workspace wood annotations)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=str(paths.sam_checkpoint),
        help="SAM 2 checkpoint",
    )
    parser.add_argument("--sam-device", default="cuda:1", help="SAM device, e.g. cuda:1 or cpu")
    parser.add_argument("--work-dim", type=int, default=1024, help="Longer working-image side")
    args = parser.parse_args(argv)

    WoodAnnotator(
        [Path(path) for path in args.inputs],
        Path(args.out_dir),
        checkpoint=args.sam_checkpoint,
        device=args.sam_device,
        work_dim=args.work_dim,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
