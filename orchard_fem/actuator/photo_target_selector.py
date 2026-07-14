"""Small target-tree prompt used before Harvest Console photo extraction."""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import numpy as np

from orchard_fem.actuator.ui_layout import fit_window_to_screen


@dataclass(frozen=True)
class TargetTreeSelection:
    """Operator prompt in pipeline working-image coordinates."""

    roi_xyxy: tuple[int, int, int, int]
    trunk_root_rc: tuple[int, int]


class TargetTreeSelectorDialog:
    """Let the operator isolate one tree and anchor its visible trunk base."""

    def __init__(self, owner: tk.Misc, image: np.ndarray) -> None:
        self.owner = owner
        image_array = np.asarray(image)[..., :3]
        if np.issubdtype(image_array.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(image_array)) <= 1.0 else 1.0
            image_array = np.clip(image_array * scale, 0, 255)
        self.image = image_array.astype(np.uint8)
        self.result: TargetTreeSelection | None = None
        self.roi_xyxy: tuple[int, int, int, int] | None = None
        self.trunk_root_rc: tuple[int, int] | None = None
        self.mode = "roi"
        self._drag_start: np.ndarray | None = None
        self._drag_current: np.ndarray | None = None
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._photo = None

        self.win = tk.Toplevel(owner)
        self.win.title("Select target tree and trunk base")
        fit_window_to_screen(self.win, (1120, 760), (700, 480))
        self.win.transient(owner)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)
        self.win.bind("<Escape>", lambda _event: self._cancel())

        header = ttk.Frame(self.win, padding=(8, 7))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="1. Drag a tight box around one target tree.  2. Click its visible trunk base.",
            anchor="w",
        ).pack(fill="x")
        toolbar = ttk.Frame(self.win, padding=(8, 0, 8, 7))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Draw target ROI", command=self._set_roi_mode).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Set trunk base", command=self._set_root_mode).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Use full image", command=self._use_full_image).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Cancel", command=self._cancel).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Continue extraction", command=self._accept).pack(
            side="right", padx=2
        )

        self.canvas = tk.Canvas(self.win, bg="#252525", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        self.status = tk.StringVar(value="Drag a rectangle around the target tree.")
        ttk.Label(self.win, textvariable=self.status, anchor="w", relief="sunken", padding=5).pack(
            fill="x", padx=8, pady=(5, 8)
        )
        self.win.after_idle(self._redraw)
        self.win.grab_set()

    def _image_point(self, x: float, y: float) -> np.ndarray | None:
        row = (y - self._offset_y) / self._scale
        col = (x - self._offset_x) / self._scale
        height, width = self.image.shape[:2]
        if not (0 <= row < height and 0 <= col < width):
            return None
        return np.asarray([row, col], dtype=float)

    def _screen_xy(self, row: float, col: float) -> tuple[float, float]:
        return self._offset_x + col * self._scale, self._offset_y + row * self._scale

    def _redraw(self) -> None:
        try:
            from PIL import Image, ImageTk
        except Exception as error:  # noqa: BLE001
            self.status.set(f"Pillow is required for target selection: {error}")
            return
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        source = Image.fromarray(self.image)
        self._scale = min(width / source.width, height / source.height)
        draw_width = max(1, int(round(source.width * self._scale)))
        draw_height = max(1, int(round(source.height * self._scale)))
        self._offset_x = (width - draw_width) / 2.0
        self._offset_y = (height - draw_height) / 2.0
        resized = source.resize((draw_width, draw_height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            self._offset_x, self._offset_y, anchor="nw", image=self._photo
        )

        roi = self.roi_xyxy
        if self._drag_start is not None and self._drag_current is not None:
            rows = sorted((self._drag_start[0], self._drag_current[0]))
            cols = sorted((self._drag_start[1], self._drag_current[1]))
            roi = (int(cols[0]), int(rows[0]), int(cols[1]), int(rows[1]))
        if roi is not None:
            left, top, right, bottom = roi
            x1, y1 = self._screen_xy(top, left)
            x2, y2 = self._screen_xy(bottom, right)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00e5ff", width=3)
        if self.trunk_root_rc is not None:
            row, col = self.trunk_root_rc
            x, y = self._screen_xy(row, col)
            radius = 7
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#ffeb3b",
                outline="#202020",
                width=2,
            )

    def _set_roi_mode(self) -> None:
        self.mode = "roi"
        self.status.set("Drag a tight rectangle around only the target tree.")

    def _set_root_mode(self) -> None:
        self.mode = "root"
        self.status.set("Click the visible base of the target trunk.")

    def _use_full_image(self) -> None:
        height, width = self.image.shape[:2]
        self.roi_xyxy = (0, 0, width, height)
        self._set_root_mode()
        self._redraw()

    def _on_press(self, event) -> None:
        point = self._image_point(event.x, event.y)
        if point is None:
            return
        if self.mode == "root":
            self.trunk_root_rc = tuple(np.rint(point).astype(int))
            row, col = self.trunk_root_rc
            self.status.set(
                f"Trunk base set at row={row}, col={col}. "
                "Press Continue extraction, or click to adjust it."
            )
            self._redraw()
            return
        self._drag_start = point
        self._drag_current = point.copy()

    def _on_motion(self, event) -> None:
        if self.mode != "roi" or self._drag_start is None:
            return
        point = self._image_point(event.x, event.y)
        if point is not None:
            self._drag_current = point
            self._redraw()

    def _on_release(self, event) -> None:
        if self.mode != "roi" or self._drag_start is None:
            return
        point = self._image_point(event.x, event.y)
        if point is not None:
            rows = sorted((self._drag_start[0], point[0]))
            cols = sorted((self._drag_start[1], point[1]))
            left, right = int(round(cols[0])), int(round(cols[1]))
            top, bottom = int(round(rows[0])), int(round(rows[1]))
            if right - left >= 10 and bottom - top >= 10:
                self.roi_xyxy = (left, top, right, bottom)
                self.mode = "root"
                self.status.set(
                    f"Target ROI=({left}, {top}, {right}, {bottom}). "
                    "Now click the visible trunk base."
                )
        self._drag_start = None
        self._drag_current = None
        self._redraw()

    def _accept(self) -> None:
        if self.roi_xyxy is None:
            messagebox.showwarning("Target tree required", "Draw the target-tree ROI first.", parent=self.win)
            return
        if self.trunk_root_rc is None:
            messagebox.showwarning("Trunk base required", "Click the visible trunk base first.", parent=self.win)
            return
        left, top, right, bottom = self.roi_xyxy
        row, col = self.trunk_root_rc
        if not (left <= col < right and top <= row < bottom):
            messagebox.showwarning(
                "Invalid trunk base",
                "The trunk base must lie inside the selected target-tree ROI.",
                parent=self.win,
            )
            return
        self.result = TargetTreeSelection(self.roi_xyxy, self.trunk_root_rc)
        self.win.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.win.destroy()


def select_target_tree(owner: tk.Misc, image: np.ndarray) -> TargetTreeSelection | None:
    """Show the modal target prompt and return its working-image coordinates."""
    dialog = TargetTreeSelectorDialog(owner, image)
    owner.wait_window(dialog.win)
    return dialog.result
