"""Tk skeleton editor embedded in :mod:`orchard_fem.actuator.harvest_console`.

The editor works in source-image pixel coordinates and delegates all topology,
evidence and persistence rules to :mod:`orchard_vision.skeleton_editing`.  Keeping
those rules outside Tk makes the safety-critical part of the workflow testable in
headless CI.
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np

from orchard_vision.skeleton_editing import EditableSkeleton, SkeletonEditError


_LEVEL_COLOURS = {
    0: "#e6194b",
    1: "#f58231",
    2: "#3cb44b",
    3: "#4363d8",
}
_DEEP_COLOUR = "#911eb4"


class SkeletonEditorWindow:
    """Interactive photo-overlay editor owned by the Harvest Console."""

    def __init__(
        self,
        owner: tk.Misc,
        document: EditableSkeleton,
        *,
        on_model_exported: Callable[[Path], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.owner = owner
        self.document = document
        self.on_model_exported = on_model_exported
        self.log = log or (lambda _message: None)
        self.win = tk.Toplevel(owner)
        self.win.title(f"Skeleton editor — {document.model_name}")
        self.win.geometry("1320x820")
        self.win.minsize(1040, 680)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        self.mode = tk.StringVar(value="select")
        self.brush_radius = tk.IntVar(value=8)
        self.parent_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.selected_branch_id: str | None = next(
            (branch.id for branch in document.branches if branch.parent_id is None),
            document.branches[0].id if document.branches else None,
        )
        self.selected_control: int | None = None
        self.new_points: list[np.ndarray] = []
        self._candidate_mask: np.ndarray | None = None
        self._undo: list = []
        self._redo: list = []
        self._drag_before = None
        self._drag_controls: np.ndarray | None = None
        self._mask_before = None
        self._mask_painting = False
        self._dirty = False
        self._background_photo = None
        self._instance_labels: np.ndarray | None = None
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

        self._build()
        self._refresh_branch_table()
        self._select_branch(self.selected_branch_id)
        self.win.after_idle(self._redraw)

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        toolbar = ttk.Frame(self.win, padding=(6, 5))
        toolbar.pack(fill="x")
        for text, value in (
            ("Select / drag", "select"),
            ("Add branch", "add"),
            ("Paint wood", "paint"),
            ("Erase wood", "erase"),
        ):
            ttk.Radiobutton(
                toolbar,
                text=text,
                value=value,
                variable=self.mode,
                command=self._mode_changed,
            ).pack(side="left", padx=2)
        ttk.Label(toolbar, text="Brush px").pack(side="left", padx=(10, 2))
        ttk.Spinbox(toolbar, from_=1, to=60, width=4, textvariable=self.brush_radius).pack(
            side="left"
        )
        ttk.Button(toolbar, text="Finish new branch", command=self._finish_new_branch).pack(
            side="left", padx=(10, 2)
        )
        ttk.Button(toolbar, text="Cancel stroke", command=self._cancel_new_branch).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Undo", command=self._undo_edit).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Redo", command=self._redo_edit).pack(side="right", padx=2)

        main = ttk.Panedwindow(self.win, orient="horizontal")
        main.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        canvas_frame = ttk.Frame(main)
        side = ttk.Frame(main, width=330)
        main.add(canvas_frame, weight=5)
        main.add(side, weight=2)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#252525",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Button-3>", lambda _event: self._finish_new_branch())
        self.win.bind("<Delete>", self._delete_control)
        self.win.bind("<Escape>", lambda _event: self._cancel_new_branch())
        self.win.bind("<Control-z>", lambda _event: self._undo_edit())
        self.win.bind("<Control-y>", lambda _event: self._redo_edit())

        branch_box = ttk.LabelFrame(side, text="Branches", padding=5)
        branch_box.pack(fill="both", expand=True, pady=(0, 5))
        columns = ("id", "parent", "level", "length", "support")
        self.branch_tree = ttk.Treeview(
            branch_box,
            columns=columns,
            show="headings",
            height=13,
            selectmode="browse",
        )
        headings = {
            "id": ("ID", 92),
            "parent": ("Parent", 82),
            "level": ("L", 30),
            "length": ("Length m", 66),
            "support": ("Evidence", 62),
        }
        for column, (label, width) in headings.items():
            self.branch_tree.heading(column, text=label)
            self.branch_tree.column(column, width=width, anchor="center")
        self.branch_tree.pack(fill="both", expand=True)
        self.branch_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        topology = ttk.LabelFrame(side, text="Hierarchy (level is derived from parent)", padding=6)
        topology.pack(fill="x", pady=4)
        row = ttk.Frame(topology)
        row.pack(fill="x")
        ttk.Label(row, text="Parent").pack(side="left")
        self.parent_combo = ttk.Combobox(
            row, textvariable=self.parent_var, state="readonly", width=20
        )
        self.parent_combo.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row, text="Apply", command=self._apply_parent).pack(side="left")
        buttons = ttk.Frame(topology)
        buttons.pack(fill="x", pady=(5, 0))
        ttk.Button(buttons, text="Promote", command=self._promote_branch).pack(
            side="left", fill="x", expand=True, padx=(0, 2)
        )
        ttk.Button(buttons, text="Delete branch", command=self._delete_branch).pack(
            side="left", fill="x", expand=True, padx=(2, 0)
        )

        help_box = ttk.LabelFrame(side, text="Editing", padding=6)
        help_box.pack(fill="x", pady=4)
        ttk.Label(
            help_box,
            justify="left",
            wraplength=305,
            text=(
                "Select: click a line; drag yellow control points. Double-click near a "
                "selected line to insert a node; Delete removes the selected node.\n"
                "Add branch: first select its parent, then click along the visible limb. "
                "Right-click or press Finish. Unsupported lines require a visible-mask proposal.\n"
                "Paint/erase changes the wood evidence used for length and diameter sampling. "
                "The selected branch's matched instance region is highlighted amber."
            ),
        ).pack(anchor="w")

        files = ttk.LabelFrame(side, text="Project / export", padding=6)
        files.pack(fill="x", pady=4)
        ttk.Button(files, text="Save editable project…", command=self._save_project).pack(
            fill="x", pady=2
        )
        ttk.Button(files, text="Validate", command=self._validate).pack(fill="x", pady=2)
        ttk.Button(
            files,
            text="Export skeleton + load FEM model…",
            command=self._export_and_load,
        ).pack(fill="x", pady=2)

        status = ttk.Label(
            self.win,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            padding=(7, 3),
        )
        status.pack(fill="x", padx=6, pady=(0, 5))

    # -------------------------------------------------------------- coordinate/render
    def _image_uint8(self) -> np.ndarray:
        image = np.asarray(self.document.image)[..., :3]
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255)
        return image.astype(np.uint8)

    def _screen_xy(self, point_rc: np.ndarray) -> tuple[float, float]:
        return (
            self._offset_x + float(point_rc[1]) * self._scale,
            self._offset_y + float(point_rc[0]) * self._scale,
        )

    def _canvas_rc(self, x: float, y: float) -> np.ndarray | None:
        if self._scale <= 0:
            return None
        row = (y - self._offset_y) / self._scale
        col = (x - self._offset_x) / self._scale
        height, width = self.document.mask.shape
        if not (0 <= row < height and 0 <= col < width):
            return None
        return np.asarray([row, col], dtype=float)

    def _redraw(self) -> None:
        try:
            from PIL import Image, ImageTk
        except Exception as error:  # noqa: BLE001
            self.status_var.set(f"Pillow is required for the skeleton editor: {error}")
            return
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        image = self._image_uint8().astype(float)
        mask = self.document.mask
        image[mask] = 0.72 * image[mask] + 0.28 * np.asarray([45.0, 180.0, 75.0])
        if self.selected_branch_id is not None:
            if self._instance_labels is None:
                self._instance_labels = self.document.instance_labels()
            branch_ids = [branch.id for branch in self.document.branches]
            if self.selected_branch_id in branch_ids:
                instance_index = branch_ids.index(self.selected_branch_id) + 1
                selected_region = self._instance_labels == instance_index
                image[selected_region] = (
                    0.48 * image[selected_region]
                    + 0.52 * np.asarray([255.0, 180.0, 35.0])
                )
        if self._candidate_mask is not None:
            candidate = self._candidate_mask
            image[candidate] = 0.55 * image[candidate] + 0.45 * np.asarray([0.0, 190.0, 220.0])
        source = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        self._scale = min(width / source.width, height / source.height)
        draw_w = max(int(round(source.width * self._scale)), 1)
        draw_h = max(int(round(source.height * self._scale)), 1)
        self._offset_x = (width - draw_w) / 2.0
        self._offset_y = (height - draw_h) / 2.0
        resized = source.resize((draw_w, draw_h), Image.Resampling.LANCZOS)
        self._background_photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            self._offset_x,
            self._offset_y,
            anchor="nw",
            image=self._background_photo,
        )

        for branch in self.document.branches:
            if not len(branch.pixels):
                continue
            coordinates: list[float] = []
            for point in branch.pixels:
                coordinates.extend(self._screen_xy(point))
            selected = branch.id == self.selected_branch_id
            colour = "#ffeb3b" if selected else _LEVEL_COLOURS.get(branch.level, _DEEP_COLOUR)
            self.canvas.create_line(
                *coordinates,
                fill=colour,
                width=4 if selected else 2,
                smooth=False,
            )
            mid = branch.pixels[len(branch.pixels) // 2]
            x, y = self._screen_xy(mid)
            self.canvas.create_text(
                x + 4,
                y - 4,
                text=branch.id,
                fill="white",
                anchor="sw",
                font=("TkDefaultFont", 8, "bold"),
            )

        if self.selected_branch_id is not None:
            controls = self._drag_controls
            if controls is None:
                controls = self.document.controls.get(self.selected_branch_id)
            if controls is not None:
                if self._drag_controls is not None:
                    preview: list[float] = []
                    for point in controls:
                        preview.extend(self._screen_xy(point))
                    self.canvas.create_line(*preview, fill="#fff176", width=2, dash=(4, 2))
                for index, point in enumerate(controls):
                    x, y = self._screen_xy(point)
                    radius = 6 if index == self.selected_control else 4
                    self.canvas.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="#ffeb3b",
                        outline="#202020",
                        width=1,
                    )

        if self.new_points:
            coordinates = []
            for point in self.new_points:
                coordinates.extend(self._screen_xy(point))
            if len(coordinates) >= 4:
                self.canvas.create_line(*coordinates, fill="#00e5ff", width=3, dash=(5, 3))
            for point in self.new_points:
                x, y = self._screen_xy(point)
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#00e5ff")
        self._update_status()

    # ---------------------------------------------------------------- selection/edit
    def _nearest_branch(self, point_rc: np.ndarray) -> str | None:
        best_id, best_distance = None, float("inf")
        for branch in self.document.branches:
            if not len(branch.pixels):
                continue
            distance = float(np.min(np.linalg.norm(branch.pixels.astype(float) - point_rc, axis=1)))
            if distance < best_distance:
                best_id, best_distance = branch.id, distance
        threshold = max(9.0 / max(self._scale, 1.0e-6), 3.0)
        return best_id if best_distance <= threshold else None

    def _nearest_control(self, point_rc: np.ndarray) -> int | None:
        if self.selected_branch_id is None:
            return None
        controls = self.document.controls.get(self.selected_branch_id)
        if controls is None or not len(controls):
            return None
        distances = np.linalg.norm(controls - point_rc, axis=1)
        index = int(np.argmin(distances))
        threshold = max(10.0 / max(self._scale, 1.0e-6), 3.0)
        return index if float(distances[index]) <= threshold else None

    def _on_press(self, event) -> None:
        point = self._canvas_rc(event.x, event.y)
        if point is None:
            return
        mode = self.mode.get()
        if mode == "add":
            self.new_points.append(point)
            self._redraw()
            return
        if mode in ("paint", "erase"):
            self._mask_before = self.document.snapshot()
            self._mask_painting = True
            self.document.paint_mask([tuple(point)], self.brush_radius.get(), mode == "paint")
            self._redraw()
            return
        control = self._nearest_control(point)
        if control is not None:
            self.selected_control = control
            self._drag_before = self.document.snapshot()
            self._drag_controls = self.document.controls[self.selected_branch_id].copy()
            self._drag_controls[control] = point
            self._redraw()
            return
        branch_id = self._nearest_branch(point)
        if branch_id is not None:
            self._select_branch(branch_id)
        else:
            self.selected_control = None
            self._redraw()

    def _on_motion(self, event) -> None:
        point = self._canvas_rc(event.x, event.y)
        if point is None:
            return
        mode = self.mode.get()
        if self._mask_painting and mode in ("paint", "erase"):
            self.document.paint_mask([tuple(point)], self.brush_radius.get(), mode == "paint")
            self._redraw()
            return
        if self._drag_controls is not None and self.selected_control is not None:
            self._drag_controls[self.selected_control] = point
            self._redraw()

    def _on_release(self, _event) -> None:
        if self._mask_painting:
            self._mask_painting = False
            self.document.finish_mask_edit()
            if self._mask_before is not None:
                self._record_edit(self._mask_before)
            self._mask_before = None
            self._refresh_after_edit()
            return
        if self._drag_controls is None or self.selected_branch_id is None:
            return
        before = self._drag_before
        controls = self._drag_controls.copy()
        self._drag_controls = None
        self._drag_before = None
        try:
            self.document.update_branch(self.selected_branch_id, controls)
        except SkeletonEditError as error:
            if before is not None:
                self.document.restore(before)
            messagebox.showwarning("Edit rejected", str(error), parent=self.win)
        else:
            if before is not None:
                self._record_edit(before)
        self._refresh_after_edit()

    def _on_double_click(self, event) -> None:
        if self.mode.get() != "select" or self.selected_branch_id is None:
            return
        point = self._canvas_rc(event.x, event.y)
        if point is None:
            return
        controls = self.document.controls[self.selected_branch_id].copy()
        if len(controls) < 2:
            return
        best_index, best_point, best_distance = 0, point, float("inf")
        for index, (first, second) in enumerate(zip(controls, controls[1:])):
            delta = second - first
            denom = float(np.dot(delta, delta))
            alpha = 0.0 if denom <= 1.0e-12 else float(
                np.clip(np.dot(point - first, delta) / denom, 0.0, 1.0)
            )
            projection = first + alpha * delta
            distance = float(np.linalg.norm(point - projection))
            if distance < best_distance:
                best_index, best_point, best_distance = index, point, distance
        before = self.document.snapshot()
        proposed = np.insert(controls, best_index + 1, best_point, axis=0)
        try:
            self.document.update_branch(self.selected_branch_id, proposed)
        except SkeletonEditError as error:
            messagebox.showwarning("Node rejected", str(error), parent=self.win)
            return
        self.selected_control = best_index + 1
        self._record_edit(before)
        self._refresh_after_edit()

    def _delete_control(self, _event=None) -> None:
        if self.mode.get() != "select" or self.selected_branch_id is None:
            return
        if self.selected_control is None:
            return
        controls = self.document.controls[self.selected_branch_id].copy()
        if len(controls) <= 2:
            messagebox.showwarning(
                "Cannot delete node", "A branch must keep at least two nodes.", parent=self.win
            )
            return
        before = self.document.snapshot()
        proposed = np.delete(controls, self.selected_control, axis=0)
        try:
            self.document.update_branch(self.selected_branch_id, proposed)
        except SkeletonEditError as error:
            messagebox.showwarning("Node rejected", str(error), parent=self.win)
            return
        self.selected_control = min(self.selected_control, len(proposed) - 1)
        self._record_edit(before)
        self._refresh_after_edit()

    # ------------------------------------------------------------- new branch/mask
    def _finish_new_branch(self) -> None:
        if self.mode.get() != "add" or len(self.new_points) < 2:
            if self.mode.get() == "add":
                messagebox.showinfo(
                    "Add branch", "Click at least two points along a visible branch.", parent=self.win
                )
            return
        if self.selected_branch_id is None:
            messagebox.showwarning("Add branch", "Select the parent branch first.", parent=self.win)
            return
        controls = np.asarray(self.new_points, dtype=float)
        before = self.document.snapshot()
        try:
            new_id = self.document.add_branch(self.selected_branch_id, controls)
        except SkeletonEditError as first_error:
            proposal = self.document.suggest_branch_mask(controls)
            direct = np.zeros(self.document.mask.shape, dtype=bool)
            if proposal.any():
                # The proposal must cover a meaningful part of the user's stroke;
                # an isolated dark leaf in the corridor is not enough.
                from skimage.draw import line

                for first, second in zip(controls, controls[1:]):
                    start = np.rint(first).astype(int)
                    end = np.rint(second).astype(int)
                    start[0] = np.clip(start[0], 0, direct.shape[0] - 1)
                    start[1] = np.clip(start[1], 0, direct.shape[1] - 1)
                    end[0] = np.clip(end[0], 0, direct.shape[0] - 1)
                    end[1] = np.clip(end[1], 0, direct.shape[1] - 1)
                    rr, cc = line(*start, *end)
                    direct[rr, cc] = True
                proposal_support = float(np.mean(proposal[direct])) if direct.any() else 0.0
            else:
                proposal_support = 0.0
            if proposal_support < 0.55:
                messagebox.showwarning(
                    "Branch has no image support",
                    f"{first_error}\n\nThe source image also provides insufficient visible-wood "
                    "evidence along this stroke. Paint/confirm the real branch region first.",
                    parent=self.win,
                )
                self.document.restore(before)
                self._redraw()
                return
            self._candidate_mask = proposal
            self._redraw()
            self.win.update_idletasks()
            accepted = messagebox.askyesno(
                "Confirm corresponding branch region",
                "The cyan region is the image-derived candidate belonging to the new line.\n\n"
                "Merge this region into the wood evidence and add the branch?",
                parent=self.win,
            )
            self._candidate_mask = None
            if not accepted:
                self.document.restore(before)
                self._redraw()
                return
            try:
                self.document.merge_mask(proposal)
                new_id = self.document.add_branch(self.selected_branch_id, controls)
            except SkeletonEditError as second_error:
                self.document.restore(before)
                messagebox.showwarning("Branch rejected", str(second_error), parent=self.win)
                self._redraw()
                return
        self._record_edit(before)
        self.new_points.clear()
        self.mode.set("select")
        self._select_branch(new_id)
        self._refresh_after_edit()
        self.log(f"Skeleton editor: added {new_id} with image evidence")

    def _cancel_new_branch(self) -> None:
        self.new_points.clear()
        self._candidate_mask = None
        self._drag_controls = None
        self._redraw()

    def _mode_changed(self) -> None:
        if self.mode.get() != "add":
            self.new_points.clear()
        self._redraw()

    # ---------------------------------------------------------------- topology
    def _on_tree_select(self, _event=None) -> None:
        selection = self.branch_tree.selection()
        if selection:
            self._select_branch(selection[0], update_tree=False)

    def _select_branch(self, branch_id: str | None, *, update_tree: bool = True) -> None:
        self.selected_branch_id = branch_id
        self.selected_control = None
        self._drag_controls = None
        if branch_id is None:
            self.parent_combo.configure(values=[])
            self.parent_var.set("")
        else:
            branch = self.document.branch(branch_id)
            excluded = {branch_id, *self.document.descendant_ids(branch_id)}
            choices = [candidate.id for candidate in self.document.branches if candidate.id not in excluded]
            self.parent_combo.configure(values=choices)
            self.parent_var.set(branch.parent_id or "")
            if update_tree and self.branch_tree.exists(branch_id):
                self.branch_tree.selection_set(branch_id)
                self.branch_tree.see(branch_id)
        self._redraw()

    def _apply_parent(self) -> None:
        if self.selected_branch_id is None:
            return
        parent_id = self.parent_var.get().strip()
        if not parent_id:
            return
        before = self.document.snapshot()
        try:
            self.document.set_parent(self.selected_branch_id, parent_id)
        except SkeletonEditError as error:
            messagebox.showwarning("Hierarchy rejected", str(error), parent=self.win)
            return
        self._record_edit(before)
        self._refresh_after_edit()

    def _promote_branch(self) -> None:
        if self.selected_branch_id is None:
            return
        branch = self.document.branch(self.selected_branch_id)
        if branch.parent_id is None:
            return
        parent = self.document.branch(branch.parent_id)
        if parent.parent_id is None:
            messagebox.showinfo("Promote", "This branch is already primary.", parent=self.win)
            return
        self.parent_var.set(parent.parent_id)
        self._apply_parent()

    def _delete_branch(self) -> None:
        if self.selected_branch_id is None:
            return
        branch = self.document.branch(self.selected_branch_id)
        if branch.parent_id is None:
            messagebox.showwarning("Delete", "The trunk cannot be deleted.", parent=self.win)
            return
        descendants = self.document.descendant_ids(branch.id)
        suffix = f" and {len(descendants)} descendant(s)" if descendants else ""
        if not messagebox.askyesno(
            "Delete branch", f"Delete '{branch.id}'{suffix}?", parent=self.win
        ):
            return
        before = self.document.snapshot()
        parent_id = branch.parent_id
        self.document.delete_branch(branch.id, cascade=True)
        self._record_edit(before)
        self._select_branch(parent_id)
        self._refresh_after_edit()

    # ---------------------------------------------------------------- undo/files
    def _record_edit(self, before) -> None:
        self._undo.append(before)
        if len(self._undo) > 50:
            self._undo.pop(0)
        self._redo.clear()
        self._dirty = True

    def _undo_edit(self) -> None:
        if not self._undo:
            return
        self._redo.append(self.document.snapshot())
        self.document.restore(self._undo.pop())
        if self.selected_branch_id not in {branch.id for branch in self.document.branches}:
            self.selected_branch_id = self.document.branches[0].id
        self._dirty = True
        self._refresh_after_edit()

    def _redo_edit(self) -> None:
        if not self._redo:
            return
        self._undo.append(self.document.snapshot())
        self.document.restore(self._redo.pop())
        self._dirty = True
        self._refresh_after_edit()

    def _refresh_branch_table(self) -> None:
        selected = self.selected_branch_id
        self.branch_tree.delete(*self.branch_tree.get_children())
        for branch in self.document.branches:
            evidence = self.document.evidence(branch.id)
            self.branch_tree.insert(
                "",
                "end",
                iid=branch.id,
                values=(
                    branch.id,
                    branch.parent_id or "—",
                    branch.level,
                    f"{self.document.length_m(branch.id):.3f}",
                    f"{evidence.support_ratio:.0%}",
                ),
            )
        if selected and self.branch_tree.exists(selected):
            self.branch_tree.selection_set(selected)

    def _refresh_after_edit(self) -> None:
        self._instance_labels = None
        self._refresh_branch_table()
        self._select_branch(self.selected_branch_id)

    def _update_status(self) -> None:
        if self.selected_branch_id is None:
            self.status_var.set("Select a branch. Green overlay = accepted wood evidence; cyan = proposal.")
            return
        branch = self.document.branch(self.selected_branch_id)
        evidence = self.document.evidence(branch.id)
        self.status_var.set(
            f"{branch.id}  parent={branch.parent_id or '—'}  level={branch.level}  "
            f"length={self.document.length_m(branch.id):.3f} m  "
            f"support={evidence.support_ratio:.0%}  mean radius={evidence.mean_radius_px:.1f}px  "
            f"mode={self.mode.get()}"
        )

    def _save_project(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".json",
            initialfile=f"{self.document.model_name}.skeleton-project.json",
            filetypes=[("Skeleton edit project", "*.json")],
        )
        if not path:
            return
        try:
            saved = self.document.save_project(path)
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Save failed", str(error), parent=self.win)
            return
        self._dirty = False
        self.log(f"Skeleton edit project saved → {saved}")

    def _validate(self) -> None:
        errors = self.document.validation_errors()
        if errors:
            messagebox.showwarning(
                "Skeleton validation",
                "The skeleton cannot be exported:\n\n- " + "\n- ".join(errors),
                parent=self.win,
            )
        else:
            messagebox.showinfo(
                "Skeleton validation",
                f"Valid: {len(self.document.branches)} branches with image evidence.",
                parent=self.win,
            )

    def _export_and_load(self) -> None:
        try:
            payload = self.document.to_skeleton_payload()
        except SkeletonEditError as error:
            messagebox.showwarning("Export blocked", str(error), parent=self.win)
            return
        skeleton_path = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".json",
            initialfile=f"{self.document.model_name}.edited.skeleton.json",
            filetypes=[("Skeleton JSON", "*.json")],
        )
        if not skeleton_path:
            return
        model_path = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".json",
            initialfile=f"{self.document.model_name}.edited.model.json",
            filetypes=[("Orchard FEM model", "*.json")],
        )
        if not model_path:
            return
        try:
            skeleton = Path(skeleton_path)
            skeleton.parent.mkdir(parents=True, exist_ok=True)
            skeleton.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            from orchard_fem.io.skeleton_import import convert_skeleton_file

            model = convert_skeleton_file(skeleton, Path(model_path))
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Export failed", str(error), parent=self.win)
            return
        self._dirty = False
        self.log(f"Edited skeleton exported → {skeleton}")
        self.log(f"Solver model exported → {model}")
        if self.on_model_exported is not None:
            self.on_model_exported(model)
        messagebox.showinfo(
            "Export complete",
            "The edited skeleton was converted and loaded into Harvest Console.",
            parent=self.win,
        )

    def _on_close(self) -> None:
        if self._dirty and not messagebox.askyesno(
            "Close skeleton editor",
            "There are unsaved edits. Close without saving an editable project?",
            parent=self.win,
        ):
            return
        self.win.destroy()


__all__ = ["SkeletonEditorWindow"]
