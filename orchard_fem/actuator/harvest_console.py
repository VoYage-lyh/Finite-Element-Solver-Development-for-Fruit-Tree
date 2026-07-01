# -*- coding: utf-8 -*-
"""Harvest Console — 整链交互前端:树模型 → 仿真 → 电动缸执行.

三步流程(Notebook 标签页),共享底部日志:

① **树模型** — 选择 tree JSON(``trees/*.json``),加载并显示主要参数
   (:func:`orchard_fem.workflows.harvest_recommendation.summarize_orchard_model`)。
② **仿真** — 后台线程一键跑完整条流水线:
   :func:`~orchard_fem.workflows.harvest_recommendation.recommend_harvest_parameters`
   (FRF 扫频 → 共振 → 夹持×(f,A) Pareto,**叠加电动缸包络硬约束**)定出最佳夹持,
   随后自动在其上构建**调参序列**
   (:func:`~orchard_fem.workflows.harvest_schedule.compute_harvest_schedule`:一次激振
   够不够、不够就分阶段;每阶段时长由疲劳模型算)。结果表 + 推荐/序列两面板并排显示。
③ **执行** — 串口连接、清报警、回中,一个 RUN 把②的序列在电动缸上逐阶段跑
   (:func:`~orchard_fem.actuator.ds5l1.run_harvest_schedule_on_rig`),STOP 立即断使能;
   每次运行归档至 ``results/harvest_runs/``。

运行:``python -m orchard_fem.actuator.harvest_console``(仿真需 dolfinx)。
"""
from __future__ import annotations

import dataclasses
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from importlib import util as _importlib_util
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from orchard_fem.actuator.ds5l1 import DS5L1, run_harvest_schedule_on_rig
from orchard_fem.actuator.harvest_bridge import DS5L1Limits
from orchard_fem.workflows.harvest_recommendation import (
    RecommendationOptions,
    RecommendationResult,
    candidate_clamp_labels,
    summarize_orchard_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TREES_DIR = REPO_ROOT / "trees"
RUNS_DIR = REPO_ROOT / "results" / "harvest_runs"

LIMITS = DS5L1Limits()

# Flat modern palette (forestry-green accent), used across the whole console.
PALETTE = {
    "bg": "#eef2f0",          # window background (cool light grey-green)
    "surface": "#ffffff",     # cards / inputs / tables
    "surface_alt": "#f3f6f4", # subtle alternate fill (tabs, headings)
    "border": "#d4ddd8",
    "text": "#1f2a24",
    "muted": "#5f6f67",
    "primary": "#2e7d32",     # primary action / accent (green)
    "primary_dark": "#1b5e20",
    "primary_hover": "#388e3c",
    "primary_soft": "#e3efe5",
    "accent": "#1565c0",      # schedule action (blue)
    "accent_hover": "#1976d2",
    "danger": "#c62828",      # stop
    "danger_hover": "#d32f2f",
    "on_dark": "#eaf3ec",
    "disabled_fg": "#b9c4be",
    "sel": "#cdeccf",
}


def _has(module: str) -> bool:
    return _importlib_util.find_spec(module) is not None


def _pick_ui_font(root: tk.Tk) -> str:
    """
    Linux/WSL 下 Tk 默认字体常缺 CJK 字形(显示为 \\uXXXX 或方框);
    WSL 上可通过 fontconfig 引入 /mnt/c/Windows/Fonts(微软雅黑)。
    """
    import tkinter.font as tkfont

    families = set(tkfont.families(root))
    for cand in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC",
                 "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei",
                 "Segoe UI"):
        if cand in families:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                         "TkHeadingFont", "TkTooltipFont", "TkCaptionFont"):
                try:
                    tkfont.nametofont(name).configure(family=cand)
                except tk.TclError:
                    pass
            return cand
    return "TkDefaultFont"


class HarvestConsole:
    """主窗口:四步标签页 + 共享日志/进度条。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.model = None
        self.model_path: Path | None = None
        self.result: RecommendationResult | None = None
        self.schedule = None                       # HarvestSchedule (the executable artifact)
        self._hlabels: dict = {}                   # branch_id → hierarchical label (T/1/1.1)
        self._clamp_raw: list = []                 # raw "branch_id@s" aligned with the clamp list
        self._fig_imgs: dict = {}                  # keep PhotoImage refs alive
        self._fig_src: dict = {}                   # label → source PIL image (for rescale)
        self._fig_fitsize: dict = {}               # label → last fitted (w, h)
        self._fig_dir = None                       # temp dir for rendered topology PNGs
        self.drv = DS5L1()
        self._q: queue.Queue = queue.Queue()
        self._sim_cancel = threading.Event()
        self._rig_stop = threading.Event()
        self._sim_running = False
        self._rig_running = False

        root.title("Orchard Harvest Console — Simulation → DS5L1 Execution")
        root.geometry("1180x720")               # landscape rectangle, compact
        root.minsize(1040, 640)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.ui_font = _pick_ui_font(root)
        self._setup_style(ttk.Style())
        p = PALETTE

        # --- header band: title + flow (left), lab identity (right) ---
        header = tk.Frame(root, bg=p["primary_dark"])
        header.pack(fill="x")
        left = tk.Frame(header, bg=p["primary_dark"])
        left.pack(side="left", anchor="w", padx=14, pady=6)
        tk.Label(left, text="Orchard Harvest Console",
                 font=(self.ui_font, 17, "bold"), bg=p["primary_dark"], fg="white",
                 ).pack(anchor="w")
        tk.Label(left, text="Tree model  →  Simulation  →  Execution (plan / schedule)",
                 font=(self.ui_font, 10), bg=p["primary_dark"], fg=p["on_dark"],
                 ).pack(anchor="w")
        tk.Label(header,
                 text="Key Laboratory of State Forestry Administration\n"
                      "on Forestry Equipment and Automation",
                 font=(self.ui_font, 10, "bold"), justify="right",
                 bg=p["primary_dark"], fg=p["on_dark"],
                 ).pack(side="right", anchor="e", padx=14, pady=6)
        env = (f"dolfinx {'✓' if _has('dolfinx') else '✗ load exported results only'}     "
               f"pyserial {'✓' if _has('serial') else '✗ actuator connection unavailable'}")
        tk.Label(root, text=env, font=(self.ui_font, 9), bg=p["surface_alt"],
                 fg=p["muted"], anchor="w", padx=14,
                 ).pack(fill="x")

        # --- shared progress + log ---
        # Reserve the bottom panel BEFORE the notebook (side="bottom"), so it can
        # never be squeezed off-screen when a tab's content is tall.
        bottom = ttk.Frame(root, padding=(8, 2, 8, 6))
        bottom.pack(side="bottom", fill="x")
        ttk.Label(bottom, text="Log / progress", style="Muted.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(bottom, maximum=1.0)
        self.progress.pack(fill="x", pady=(2, 4))
        self.log_text = tk.Text(bottom, height=7, state="disabled",
                                font=("Consolas", 10), bg=p["surface"], fg=p["text"],
                                relief="flat", borderwidth=0, highlightthickness=1,
                                highlightbackground=p["border"], padx=8, pady=5)
        self.log_text.pack(fill="both", expand=True)

        self.nb = ttk.Notebook(root)
        self.nb.pack(side="top", fill="both", expand=True, padx=6, pady=(4, 2))
        self._build_tab_model()
        self._build_tab_sim()
        self._build_tab_rig()       # ③ Execution (working point + plan + run)

        self.root.after(100, self._pump)

    # ------------------------------------------------------------------ #
    # 主题与控件样式
    # ------------------------------------------------------------------ #

    def _setup_style(self, style: ttk.Style) -> None:
        """Flat, modern restyle of the (Linux/WSL) ttk theme."""
        p, f = PALETTE, self.ui_font
        try:
            style.theme_use("clam")          # restylable on every platform
        except tk.TclError:
            pass
        self.root.configure(bg=p["bg"])
        style.configure(".", background=p["bg"], foreground=p["text"],
                        fieldbackground=p["surface"], bordercolor=p["border"],
                        focuscolor=p["primary"], font=(f, 11))
        style.configure("TFrame", background=p["bg"])
        style.configure("TLabel", background=p["bg"], foreground=p["text"], font=(f, 11))
        style.configure("Muted.TLabel", background=p["bg"], foreground=p["muted"])
        style.configure("Heading.TLabel", background=p["bg"],
                        foreground=p["primary_dark"], font=(f, 12, "bold"))
        style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"],
                        relief="solid", borderwidth=1, padding=5)
        style.configure("TLabelframe.Label", background=p["bg"],
                        foreground=p["primary_dark"], font=(f, 11, "bold"))
        # buttons — subtle default + green accent variant
        style.configure("TButton", background=p["surface"], foreground=p["text"],
                        bordercolor=p["border"], relief="flat", borderwidth=1,
                        padding=(10, 5), font=(f, 11))
        style.map("TButton",
                  background=[("pressed", p["primary_soft"]),
                              ("active", p["surface_alt"]),
                              ("disabled", p["surface_alt"])],
                  bordercolor=[("active", p["primary"])],
                  foreground=[("disabled", p["disabled_fg"])])
        style.configure("Accent.TButton", background=p["primary"],
                        foreground="white", bordercolor=p["primary"],
                        borderwidth=0, padding=(12, 6), font=(f, 11, "bold"))
        style.map("Accent.TButton",
                  background=[("pressed", p["primary_dark"]),
                              ("active", p["primary_hover"]),
                              ("disabled", p["surface_alt"])],
                  foreground=[("disabled", p["disabled_fg"])])
        # inputs
        for w in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(w, fieldbackground=p["surface"], background=p["surface"],
                            bordercolor=p["border"], borderwidth=1, relief="flat",
                            padding=3, arrowcolor=p["muted"], font=(f, 11))
        style.map("TCombobox", fieldbackground=[("readonly", p["surface"])],
                  bordercolor=[("focus", p["primary"])])
        style.configure("TCheckbutton", background=p["bg"], foreground=p["text"], font=(f, 11))
        style.configure("TRadiobutton", background=p["bg"], foreground=p["text"], font=(f, 11))
        style.map("TCheckbutton", background=[("active", p["bg"])])
        style.map("TRadiobutton", background=[("active", p["bg"])])
        # notebook tabs
        style.configure("TNotebook", background=p["bg"], borderwidth=0,
                        tabmargins=(2, 4, 2, 0))
        style.configure("TNotebook.Tab", background=p["surface_alt"],
                        foreground=p["muted"], padding=(16, 7),
                        font=(f, 11, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", p["surface"]), ("active", p["primary_soft"])],
                  foreground=[("selected", p["primary_dark"])])
        # table
        style.configure("Treeview", background=p["surface"],
                        fieldbackground=p["surface"], foreground=p["text"],
                        rowheight=28, borderwidth=0, relief="flat", font=(f, 10))
        style.configure("Treeview.Heading", background=p["surface_alt"],
                        foreground=p["muted"], font=(f, 10, "bold"),
                        relief="flat", padding=5, borderwidth=0)
        style.map("Treeview.Heading", background=[("active", p["primary_soft"])])
        style.map("Treeview", background=[("selected", p["sel"])],
                  foreground=[("selected", p["text"])])
        # progressbar + scrollbar
        style.configure("TProgressbar", troughcolor=p["border"],
                        background=p["primary"], bordercolor=p["border"],
                        lightcolor=p["primary"], darkcolor=p["primary"],
                        thickness=10, borderwidth=0)
        style.configure("Vertical.TScrollbar", background=p["surface_alt"],
                        troughcolor=p["bg"], bordercolor=p["bg"],
                        arrowcolor=p["muted"], relief="flat", borderwidth=0)
        style.map("Vertical.TScrollbar", background=[("active", p["border"])])

    def _big_button(self, parent, text, base, hover, command):
        """A flat, filled accent button (classic tk) with hover feedback."""
        btn = tk.Button(parent, text=text, font=(self.ui_font, 14, "bold"),
                        bg=base, fg="white", activebackground=hover,
                        activeforeground="white", relief="flat", bd=0,
                        highlightthickness=0, cursor="hand2",
                        disabledforeground=PALETTE["disabled_fg"],
                        state="disabled", command=command)

        def on_enter(_e):
            if str(btn["state"]) != "disabled":
                btn.configure(bg=hover)

        def on_leave(_e):
            btn.configure(bg=base)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    # ------------------------------------------------------------------ #
    # 共享:日志 / 队列泵
    # ------------------------------------------------------------------ #

    def log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')}  {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _notify(self, title: str, msg: str, *, warn: bool = False) -> None:
        """Surface a message in the always-visible log AND a fronted dialog.

        On WSL/Linux a bare modal messagebox can open *behind* the main window
        and grab input, making the app look frozen — so we also write to the log
        and parent + lift the dialog so it actually shows.
        """
        self.log(f"⚠ {msg.splitlines()[0]}")
        try:
            self.root.lift()
            self.root.focus_force()
            (messagebox.showwarning if warn else messagebox.showerror)(
                title, msg, parent=self.root)
        except Exception:  # noqa: BLE001 - dialog must never break the flow
            pass

    def _confirm(self, title: str, body: str, *, ok_text: str = "Proceed") -> bool:
        """Modal Yes/No dialog showing *body* in a monospace, content-sized box.

        Used instead of ``messagebox.askyesno`` for plan/schedule summaries, whose
        aligned tables wrap badly in the proportional-font system dialog.
        """
        p = PALETTE
        lines = body.splitlines() or [""]
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=p["bg"])
        win.transient(self.root)
        win.resizable(False, False)
        ttk.Label(win, text=title, style="Heading.TLabel").pack(
            anchor="w", padx=14, pady=(12, 4))
        txt = tk.Text(win, font=("Consolas", 10), bg=p["surface"], fg=p["text"],
                      relief="flat", borderwidth=0, highlightthickness=1,
                      highlightbackground=p["border"], padx=10, pady=8,
                      width=min(max(len(x) for x in lines) + 2, 80),
                      height=min(len(lines), 22))
        txt.insert("end", body)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True, padx=14, pady=4)
        result = {"ok": False}
        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=14, pady=(4, 12))

        def _yes() -> None:
            result["ok"] = True
            win.destroy()

        ttk.Button(btns, text=ok_text, style="Accent.TButton", command=_yes,
                   ).pack(side="right", padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=4)
        win.bind("<Escape>", lambda _e: win.destroy())
        win.update_idletasks()
        win.lift()
        win.focus_force()
        win.grab_set()
        self.root.wait_window(win)
        return result["ok"]

    def _post(self, kind: str, payload) -> None:
        self._q.put((kind, payload))

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "sim_done":
                    self._on_sim_done(payload)
                elif kind == "sched_done":
                    self._on_sched_done(payload)
                elif kind == "figs_done":
                    self._show_figures(payload)
                elif kind == "rig_done":
                    self._on_rig_done(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    # ------------------------------------------------------------------ #
    # ① 树模型
    # ------------------------------------------------------------------ #

    def _build_tab_model(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  ① Tree Model  ")
        row = ttk.Frame(tab)
        row.pack(fill="x", padx=6, pady=4)
        self.var_model_path = tk.StringVar()
        ttk.Label(row, text="Model JSON:").pack(side="left")
        ttk.Entry(row, textvariable=self.var_model_path).pack(
            side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=self.on_browse_model).pack(side="left")
        ttk.Button(row, text="Load", command=self.on_load_model).pack(side="left", padx=6)

        main = ttk.Frame(tab)
        main.pack(fill="both", expand=True, padx=6, pady=3)

        box = ttk.LabelFrame(main, text="Model summary")
        box.pack(side="left", fill="y", padx=(0, 6))
        self.model_info = tk.Text(box, state="disabled", width=46, font=(self.ui_font, 11),
                                  bg=PALETTE["surface"], fg=PALETTE["text"],
                                  relief="flat", borderwidth=0, highlightthickness=1,
                                  highlightbackground=PALETTE["border"], padx=10, pady=8)
        self.model_info.pack(fill="both", expand=True, padx=4, pady=4)

        # topology on the right; two figures side-by-side, each auto-fit to its half
        figs = ttk.LabelFrame(main, text="Topology (branch labels match ② clamp points)")
        figs.pack(side="left", fill="both", expand=True)
        self.fig2d = ttk.Label(figs, text="2D side view — load a model to render",
                               anchor="center", style="Muted.TLabel")
        self.fig2d.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self.fig3d = ttk.Label(figs, text="3D view — load a model to render",
                               anchor="center", style="Muted.TLabel")
        self.fig3d.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        self.fig2d.bind("<Configure>", lambda _e: self._fit_figure(self.fig2d))
        self.fig3d.bind("<Configure>", lambda _e: self._fit_figure(self.fig3d))

    def on_browse_model(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(TREES_DIR if TREES_DIR.exists() else REPO_ROOT),
            filetypes=[("Tree model JSON", "*.json")])
        if path:
            self.var_model_path.set(path)
            self.on_load_model()

    def on_load_model(self) -> None:
        path = self.var_model_path.get().strip()
        if not path:
            messagebox.showwarning("Notice", "Select a model JSON file first.")
            return
        try:
            from orchard_fem.io.loaders import load_orchard_model
            self.model = load_orchard_model(path)
            self.model_path = Path(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load failed", str(e))
            self.log(f"Model load failed: {e}")
            return
        summary = summarize_orchard_model(self.model, path)
        self.model_info.configure(state="normal")
        self.model_info.delete("1.0", "end")
        self.model_info.insert("end", "\n".join(summary.lines()))
        self.model_info.configure(state="disabled")
        # hierarchical branch labels (T / 1 / 1.1 …) — shared with the topology figures
        try:
            import json
            from orchard_fem.visualization.model_scene import hierarchical_labels
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            self._hlabels = hierarchical_labels(raw.get("branches", []))
        except Exception:  # noqa: BLE001
            raw, self._hlabels = None, {}
        # candidate clamps: keep raw "branch_id@s" for execution, show hierarchical label
        try:
            raw_labels = candidate_clamp_labels(self.model, RecommendationOptions())
        except Exception:  # noqa: BLE001 - no "trunk" node → fall back to each branch root
            raw_labels = [f"{b.branch_id}@0.00" for b in self.model.branches]
        self._clamp_raw = list(raw_labels)
        self.clamp_list.delete(0, "end")
        for raw_label in raw_labels:
            self.clamp_list.insert("end", self._display_clamp(raw_label))
        self.clamp_list.selection_set(0, "end")
        if raw is not None:
            self._render_topology(raw)
        self.log(f"Model loaded: {summary.name} "
                 f"({summary.n_branches} branches, {summary.n_fruits} fruits)")

    def _display_clamp(self, raw: str) -> str:
        """Raw 'branch_id@s' → 'hlabel@s' (hierarchical label matching the figures)."""
        if "@" in raw:
            bid, s = raw.split("@", 1)
            return f"{self._hlabels.get(bid, bid)}@{s}"
        return self._hlabels.get(raw, raw)

    def _render_topology(self, raw: dict) -> None:
        """Render 2D + 3D topology PNGs in the background, then show them in ①."""
        self.fig2d.configure(text="Rendering 2D…", image="")
        self.fig3d.configure(text="Rendering 3D…", image="")
        if self._fig_dir is None:
            import tempfile
            self._fig_dir = tempfile.mkdtemp(prefix="orchard_topology_")
        p2d = str(Path(self._fig_dir) / "topology_2d.png")
        p3d = str(Path(self._fig_dir) / "topology_3d.png")

        def worker() -> None:
            try:
                from orchard_fem.visualization.rendering import plot_geometry
                from orchard_fem.visualization.scene3d import plot_tree_3d
                # structure + branch labels only — no excitation/observation markers
                # (those are FRF inputs, not a chosen clamp; would mislead pre-simulation)
                plot_geometry(raw, Path(p2d), show=False, show_io_markers=False)
                plot_tree_3d(raw, show=False, output_path=p3d, show_io_markers=False)
                try:
                    import matplotlib.pyplot as plt
                    plt.close("all")
                except Exception:  # noqa: BLE001
                    pass
                self._post("figs_done", (p2d, p3d))
            except Exception as e:  # noqa: BLE001
                self._post("figs_done", e)

        threading.Thread(target=worker, daemon=True).start()

    def _show_figures(self, payload) -> None:
        if isinstance(payload, Exception):
            self.fig2d.configure(text=f"2D render failed:\n{payload}", image="")
            self.fig3d.configure(text="", image="")
            self.log(f"Topology render failed: {payload}")
            return
        p2d, p3d = payload
        try:
            from PIL import Image
        except Exception as e:  # noqa: BLE001
            self.fig2d.configure(text=f"Pillow unavailable: {e}", image="")
            return
        for label, path in ((self.fig2d, p2d), (self.fig3d, p3d)):
            try:
                self._fig_src[label] = Image.open(path).convert("RGB")
                self._fig_fitsize.pop(label, None)   # force a re-fit
                self._fit_figure(label)
            except Exception as e:  # noqa: BLE001
                label.configure(text=f"render failed: {e}", image="")
        self.log("Topology rendered (2D + 3D).")

    def _fit_figure(self, label) -> None:
        """Scale the stored source image to the label's current size (resize-aware)."""
        src = self._fig_src.get(label)
        if src is None:
            return
        w, h = label.winfo_width(), label.winfo_height()
        if w < 40 or h < 40:
            return
        if self._fig_fitsize.get(label) == (w, h):   # avoid <Configure> feedback loop
            return
        self._fig_fitsize[label] = (w, h)
        try:
            from PIL import ImageTk
            img = src.copy()
            img.thumbnail((w - 10, h - 10))
            photo = ImageTk.PhotoImage(img)
            self._fig_imgs[label] = photo            # keep a ref so Tk doesn't GC it
            label.configure(image=photo, text="")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # ② 仿真推荐
    # ------------------------------------------------------------------ #

    def _build_tab_sim(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  ② Simulation  ")

        opts = ttk.LabelFrame(tab, text="Simulation settings")
        opts.pack(fill="x", padx=6, pady=4)
        pad = dict(padx=4, pady=3)
        self.var_band_lo = tk.DoubleVar(value=3.0)
        self.var_band_hi = tk.DoubleVar(value=20.0)
        self.var_steps = tk.IntVar(value=60)
        self.var_agrid = tk.StringVar(value="")   # blank = derive from actuator envelope
        self.var_dense = tk.BooleanVar(value=True)
        self.var_spacing = tk.DoubleVar(value=0.05)
        self.var_ddet = tk.DoubleVar(value=2.0)
        self.var_ncycles = tk.DoubleVar(value=50.0)
        self.var_coverage = tk.StringVar(value="branch")

        g = ttk.Frame(opts)
        g.pack(fill="x")
        ttk.Label(g, text="Band (Hz)").grid(row=0, column=0, **pad)
        ttk.Entry(g, textvariable=self.var_band_lo, width=5).grid(row=0, column=1, **pad)
        ttk.Label(g, text="–").grid(row=0, column=2)
        ttk.Entry(g, textvariable=self.var_band_hi, width=5).grid(row=0, column=3, **pad)
        ttk.Label(g, text="Sweep steps").grid(row=0, column=4, **pad)
        ttk.Entry(g, textvariable=self.var_steps, width=5).grid(row=0, column=5, **pad)
        ttk.Label(g, text="Amplitude A (mm, blank=auto)").grid(row=0, column=6, **pad)
        ttk.Entry(g, textvariable=self.var_agrid, width=16).grid(row=0, column=7, **pad)
        ttk.Label(g, text="Coverage").grid(row=0, column=8, **pad)
        ttk.Combobox(g, textvariable=self.var_coverage, width=7, state="readonly",
                     values=["branch", "fruit"]).grid(row=0, column=9, **pad)
        ttk.Checkbutton(g, text="Dense fruit, spacing", variable=self.var_dense,
                        ).grid(row=1, column=0, columnspan=2, **pad)
        ttk.Entry(g, textvariable=self.var_spacing, width=5).grid(row=1, column=2, columnspan=2, **pad)
        ttk.Label(g, text="Detachment (mm)").grid(row=1, column=4, **pad)
        ttk.Entry(g, textvariable=self.var_ddet, width=5).grid(row=1, column=5, **pad)
        ttk.Label(g, text="Detach cycles").grid(row=1, column=6, **pad)
        ttk.Entry(g, textvariable=self.var_ncycles, width=6).grid(row=1, column=7, **pad)

        mid = ttk.Frame(tab)
        mid.pack(fill="x", padx=6)
        clamp_box = ttk.LabelFrame(mid, text="Candidate clamp points (multi-select)")
        clamp_box.pack(side="left", fill="y")
        clamp_inner = ttk.Frame(clamp_box)
        clamp_inner.pack(fill="both", expand=True, padx=4, pady=4)
        self.clamp_list = tk.Listbox(
            clamp_inner, selectmode="multiple", height=7, width=22,
            exportselection=False, activestyle="none", font=(self.ui_font, 10),
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground=PALETTE["border"], highlightcolor=PALETTE["primary"],
            bg=PALETTE["surface"], fg=PALETTE["text"],
            selectbackground=PALETTE["sel"], selectforeground=PALETTE["text"])
        clamp_sb = ttk.Scrollbar(clamp_inner, orient="vertical",
                                 command=self.clamp_list.yview)
        self.clamp_list.configure(yscrollcommand=clamp_sb.set)
        self.clamp_list.pack(side="left", fill="both", expand=True)
        clamp_sb.pack(side="left", fill="y")
        btns = ttk.Frame(mid)
        btns.pack(side="left", fill="x", padx=10)
        self.btn_sim = ttk.Button(btns, text="▶ Run simulation", style="Accent.TButton",
                                  command=self.on_run_sim)
        self.btn_sim.pack(fill="x", pady=3)
        self.btn_sim_cancel = ttk.Button(btns, text="Cancel", state="disabled",
                                         command=lambda: self._sim_cancel.set())
        self.btn_sim_cancel.pack(fill="x", pady=3)
        ttk.Button(btns, text="Export result JSON…",
                   command=self.on_export_result).pack(fill="x", pady=2)
        ttk.Button(btns, text="Load result JSON…",
                   command=self.on_import_result).pack(fill="x", pady=2)
        # One result panel: shows the recommended point, then the schedule once built.
        self.info_text = tk.Text(
            mid, font=("Consolas", 10), height=8, wrap="none",
            bg=PALETTE["surface"], fg=PALETTE["text"], relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=PALETTE["border"],
            padx=8, pady=6, state="disabled")
        self.info_text.pack(side="left", fill="both", expand=True)
        self._set_panel(self.info_text,
                        "Run a simulation to get the recommended schedule.")

        table_box = ttk.LabelFrame(
            tab, text="Candidate working points "
                      "(★ = recommended knee,  ◆ = Pareto front,  grey = outside rig envelope)")
        table_box.pack(fill="both", expand=True, padx=6, pady=4)
        cols = ("clamp", "f", "A", "stroke", "cov", "stress", "env", "mark")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=10)
        headings = [("clamp", "Clamp", 140), ("f", "f (Hz)", 70), ("A", "A (mm)", 70),
                    ("stroke", "Stroke (mm)", 90), ("cov", "Coverage", 80),
                    ("stress", "σ trunk (MPa)", 100), ("env", "In env.", 65),
                    ("mark", "Mark", 60)]
        for cid, text, width in headings:
            self.tree.heading(cid, text=text)
            self.tree.column(cid, width=width, anchor="center")
        vsb = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")
        self.tree.tag_configure("knee", background=PALETTE["sel"],
                                foreground=PALETTE["primary_dark"])
        self.tree.tag_configure("front", background=PALETTE["primary_soft"])
        self.tree.tag_configure("infeasible", foreground="#9aa8a1")

    def _sim_options(self) -> RecommendationOptions:
        # Amplitude grid is OPTIONAL: blank ⇒ derive from the actuator envelope
        # (limits.amplitude_ladder_mm), so the range follows `limits` alone.
        a_grid = tuple(float(x) for x in
                       self.var_agrid.get().replace("，", ",").split(",") if x.strip()) or None
        # The working envelope: the realistic harvester (≤15 Hz / 20 mm). The lab
        # rig DS5L1Limits() is only used for actual on-rig execution feasibility.
        limits = DS5L1Limits.realistic_harvester()
        return RecommendationOptions(
            band_hz=(float(self.var_band_lo.get()), float(self.var_band_hi.get())),
            sweep_steps=int(self.var_steps.get()),
            amplitude_grid_mm=a_grid,
            dense_fruit_spacing=(float(self.var_spacing.get())
                                 if self.var_dense.get() else None),
            detachment_displacement_m=float(self.var_ddet.get()) / 1000.0,
            coverage_mode=self.var_coverage.get(),
            limits=limits,
        )

    def on_run_sim(self) -> None:
        if self.model is None:
            self._notify("Notice", "Load a tree model in ① first.", warn=True)
            return
        if self._sim_running:
            self.log("Simulation already running — please wait or press [Cancel].")
            return
        if not _has("dolfinx"):
            self._notify(
                "Solver backend missing",
                "dolfinx (FEniCSx) is not available, so the simulation cannot run.\n"
                "Run this app in the orchard-fenicsx environment, or load an "
                "exported result JSON instead.")
            return
        try:
            options = self._sim_options()
            # use the RAW "branch_id@s" (the list shows hierarchical labels)
            selected = [self._clamp_raw[i] for i in self.clamp_list.curselection()]
            if not selected:
                raise ValueError("Select at least one clamp point in the ② candidate list.")
        except Exception as e:  # noqa: BLE001
            self._notify("Invalid parameters", str(e))
            return
        self._sim_cancel.clear()
        self._sim_running = True
        self.btn_sim.configure(state="disabled")
        self.btn_sim_cancel.configure(state="normal")
        self.log(f"Starting simulation ({len(selected)} candidate clamps)…")
        self.root.update_idletasks()        # paint the "started" state immediately

        model, path = self.model, self.model_path
        # 夹持候选由界面选择决定
        options_sel = dataclasses.replace(options, clamp_labels=tuple(selected))

        def worker() -> None:
            try:
                from orchard_fem.workflows.harvest_recommendation import (
                    recommend_harvest_parameters,
                )
                result = recommend_harvest_parameters(
                    model, model_path=str(path), options=options_sel,
                    progress_cb=lambda m, f: (self._post("log", m),
                                              self._post("progress", f)),
                    cancel_cb=self._sim_cancel.is_set,
                )
                self._post("sim_done", result)
            except Exception as e:  # noqa: BLE001
                self._post("sim_done", e)

        threading.Thread(target=worker, daemon=True).start()

    def _on_sim_done(self, payload) -> None:
        self._sim_running = False
        self.btn_sim.configure(state="normal")
        self.btn_sim_cancel.configure(state="disabled")
        self.progress["value"] = 0
        if isinstance(payload, Exception):
            msg = str(payload)
            self.log("Simulation cancelled" if msg == "cancelled"
                     else f"Simulation failed: {msg}")
            if msg != "cancelled":
                messagebox.showerror("Simulation failed", msg)
            return
        self.result = payload
        self._fill_result_table(payload)
        self.log(f"Simulation complete in {payload.elapsed_s:.0f} s")
        # 同一条流水线:仿真出最佳夹持后,自动在其上构建调参序列(无需第二个按钮)
        if self.model is not None and _has("dolfinx") and payload.recommended is not None:
            self.on_compute_schedule()

    # ---- 调参序列(多阶段),作为 run simulation 的后半段自动执行 ----
    def on_compute_schedule(self) -> None:
        if self.result is None or self.model is None or self.result.recommended is None:
            messagebox.showwarning("Notice", "Run the simulation first.")
            return
        if not _has("dolfinx"):
            messagebox.showerror("Solver backend missing",
                                 "Building the schedule needs dolfinx "
                                 "(run in the orchard-fenicsx environment).")
            return
        opt = self._sim_options()
        a_grid = list(self.result.amplitude_grid_mm)
        ncyc = float(self.var_ncycles.get())
        model = self.model
        # Multi-clamp: one grip only sheds fruit on the branches its excitation
        # energy reaches, so its coverage caps out. Cover the tree by also building
        # grids on the next-best clamps and letting the scheduler move the grip
        # between energy-reachable regions. Each clamp scans its OWN local-mode
        # frequencies (the ones the recommendation evaluated for it).
        # The multi-clamp schedule is built by the SHARED workflow builder, so it
        # is identical to scripts/generate_all_figures.py and the rig-executed run.
        max_clamps = 6
        candidates = [c for c in self.result.clamps if c.knee is not None]
        candidates.sort(key=lambda c: c.knee.coverage, reverse=True)
        candidates = candidates[:max_clamps]
        self.btn_sim.configure(state="disabled")    # keep the pipeline locked
        self.log(f"Building multi-clamp schedule over {len(candidates)} clamp(s): "
                 f"{', '.join(self._display_clamp(c.clamp_label) for c in candidates)}…")

        def worker() -> None:
            try:
                from orchard_fem.workflows.harvest_recommendation import (
                    build_scheduling_model,
                )
                from orchard_fem.workflows.harvest_schedule import (
                    StageDurationModel,
                    build_multiclamp_schedule,
                )
                # The SAME fruited + damped model the recommendation ran on
                # (shared builder), so the schedule coverage matches the Pareto.
                m = build_scheduling_model(model, opt)
                clamp_freqs = {
                    c.clamp_label: (sorted({p.frequency_hz for p in c.points})
                                    or list(self.result.frequency_grid_hz))
                    for c in candidates
                }
                sched = build_multiclamp_schedule(
                    m, clamp_freqs, a_grid, limits=opt.limits,
                    polynomial_degree=opt.polynomial_degree, max_stages=10,
                    duration_model=StageDurationModel(reference_cycles=ncyc),
                    progress_cb=lambda msg, fr: (self._post("log", msg),
                                                 self._post("progress", fr)))
                self._post("log", f"Stage durations from {ncyc:g} detach-cycles "
                                  f"(at threshold) ÷ frequency.")
                self._post("sched_done", sched)
            except Exception as e:  # noqa: BLE001
                self._post("sched_done", e)

        threading.Thread(target=worker, daemon=True).start()

    def _set_panel(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _on_sched_done(self, payload) -> None:
        self.btn_sim.configure(state="normal")     # pipeline finished → unlock
        self.progress["value"] = 0
        if isinstance(payload, Exception):
            self._set_panel(self.info_text, f"Schedule build failed:\n{payload}")
            self.log(f"Schedule build failed: {payload}")
            messagebox.showerror("Failed", str(payload))
            return
        self.schedule = payload
        self._set_panel(self.info_text, payload.summary(label_fn=self._display_clamp))
        if payload.feasible:
            self.btn_run.configure(
                state="normal" if self.drv.connected else "disabled")
            self.log(f"Schedule built: {len(payload.stages)} stages, "
                     f"{payload.total_duration_s:.1f} s — run it in ③ Execution.")
        else:
            self.log("⚠ Schedule empty or has stages outside the rig envelope — not executable")

    def _fill_result_table(self, result: RecommendationResult) -> None:
        self.tree.delete(*self.tree.get_children())
        for ci, clamp in enumerate(result.clamps):
            for pi, p in enumerate(clamp.points):
                mark = "★" if p.is_knee else ("◆" if p.on_front else "")
                tags = (("knee",) if p.is_knee
                        else ("front",) if p.on_front
                        else ("infeasible",) if not p.rig_feasible else ())
                self.tree.insert(
                    "", "end", iid=f"{ci}:{pi}", tags=tags,
                    values=(self._display_clamp(p.clamp_label), f"{p.frequency_hz:.2f}",
                            f"{p.amplitude_mm:g}", f"{p.stroke_mm:g}",
                            f"{p.coverage:.2f}", f"{p.trunk_stress_pa / 1e6:.2f}",
                            "Yes" if p.rig_feasible else "No", mark))
        rec = result.recommended
        if rec is not None:
            self._set_panel(self.info_text, (
                "Recommended working point  (building schedule…)\n"
                f"  Clamp       {self._display_clamp(rec.clamp_label)}\n"
                f"  Frequency   {rec.frequency_hz:.2f} Hz\n"
                f"  Amplitude   {rec.amplitude_mm:g} mm  (stroke {rec.stroke_mm:g} mm)\n"
                f"  Coverage    {rec.coverage:.2f}\n"
                f"  Trunk σ     {rec.trunk_stress_pa / 1e6:.2f} MPa"))
        else:
            self._set_panel(self.info_text, "No feasible working point found.")

    def on_export_result(self) -> None:
        if self.result is None:
            messagebox.showwarning("Notice", "No simulation result yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile="recommendation.json")
        if not path:
            return
        # carry the executable schedule too, so an offline rig can run it without FE
        payload = {
            "recommendation": self.result.to_json_dict(),
            "schedule": self.schedule.to_dict() if self.schedule is not None else None,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n = len(self.schedule.stages) if self.schedule is not None else 0
        self.log(f"Result exported → {path} ({n}-stage schedule included)")

    def on_import_result(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Recommendation JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            # new format: {"recommendation": {...}, "schedule": {...}|null};
            # legacy format: a bare recommendation dict.
            rec_dict = data["recommendation"] if "recommendation" in data else data
            sched_dict = data.get("schedule") if "recommendation" in data else None
            self.result = RecommendationResult.from_json_dict(rec_dict)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load failed", str(e))
            return
        self._fill_result_table(self.result)
        for step in self.result.steps:
            self.log(f"[imported] {step}")
        self.log(f"Result loaded ({self.result.model_name})")
        if sched_dict:
            from orchard_fem.actuator.harvest_bridge import HarvestSchedule
            self._on_sched_done(HarvestSchedule.from_dict(sched_dict))
            self.log("Harvest schedule restored from file — ready to run in ③.")
        else:
            self.schedule = None
            self._set_panel(self.info_text,
                            "Loaded a recommendation with no schedule.\n"
                            "Re-run the simulation (needs dolfinx) to build the staged sequence.")

    # ------------------------------------------------------------------ #
    # ③ Execution (working point + plan + run)
    # ------------------------------------------------------------------ #

    def _build_tab_rig(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  ③ Execution  ")
        pad = dict(padx=4, pady=3)

        ttk.Label(tab, text="Connect the rig and press RUN to execute the schedule "
                            "built in ② Simulation.", style="Muted.TLabel",
                  ).pack(anchor="w", padx=8, pady=(6, 0))

        conn = ttk.LabelFrame(tab, text="Serial connection (RS232 factory default 19200-8-E-1)")
        conn.pack(fill="x", padx=6, pady=4)
        default_port = "COM8" if sys.platform.startswith("win") else "/dev/ttyUSB0"
        self.var_port = tk.StringVar(value=default_port)
        self.var_baud = tk.StringVar(value="19200")
        self.var_parity = tk.StringVar(value="E")
        self.var_stop = tk.StringVar(value="1")
        ttk.Label(conn, text="Port").grid(row=0, column=0, **pad)
        ttk.Entry(conn, textvariable=self.var_port, width=12).grid(row=0, column=1, **pad)
        ttk.Label(conn, text="Baud").grid(row=0, column=2, **pad)
        ttk.Combobox(conn, textvariable=self.var_baud, width=7,
                     values=["9600", "19200", "38400"]).grid(row=0, column=3, **pad)
        ttk.Label(conn, text="Parity").grid(row=0, column=4, **pad)
        ttk.Combobox(conn, textvariable=self.var_parity, width=3,
                     values=["E", "O", "N"]).grid(row=0, column=5, **pad)
        ttk.Label(conn, text="Stop bits").grid(row=0, column=6, **pad)
        ttk.Combobox(conn, textvariable=self.var_stop, width=3,
                     values=["1", "2"]).grid(row=0, column=7, **pad)
        self.btn_conn = ttk.Button(conn, text="Connect", command=self.on_connect)
        self.btn_conn.grid(row=0, column=8, **pad)
        self.btn_alarm = ttk.Button(conn, text="Clear alarm", state="disabled",
                                    command=self.on_clear_alarm)
        self.btn_alarm.grid(row=0, column=9, **pad)

        home = ttk.LabelFrame(
            tab, text="Centering (touch-stop homing; first P9-21 enable needs a drive power cycle)")
        home.pack(fill="x", padx=6, pady=4)
        self.var_home = tk.BooleanVar(value=True)
        self.var_homeoff = tk.DoubleVar(value=25.8)
        self.var_homerev = tk.BooleanVar(value=False)
        self.var_calibrate = tk.BooleanVar(value=False)
        ttk.Checkbutton(home, text="Auto-center before run", variable=self.var_home,
                        ).pack(side="left", **pad)
        ttk.Label(home, text="Limit→center offset (mm)").pack(side="left", **pad)
        ttk.Entry(home, textvariable=self.var_homeoff, width=6).pack(side="left", **pad)
        ttk.Checkbutton(home, text="Reverse touch-stop", variable=self.var_homerev,
                        ).pack(side="left", **pad)
        ttk.Checkbutton(
            home, text="Online freq. calibration (adds 5–23 s; off = exact duration)",
            variable=self.var_calibrate).pack(side="left", **pad)

        ctl = ttk.Frame(tab)
        ctl.pack(fill="x", padx=6, pady=8)
        self.btn_run = self._big_button(
            ctl, "▶  RUN", PALETTE["primary"], PALETTE["primary_hover"], self.on_run)
        self.btn_run.pack(side="left", expand=True, fill="x", padx=6, ipady=12)
        self.btn_rig_stop = self._big_button(
            ctl, "■  STOP", PALETTE["danger"], PALETTE["danger_hover"],
            self.on_rig_stop)
        self.btn_rig_stop.pack(side="left", expand=True, fill="x", padx=6, ipady=12)

        info = ttk.LabelFrame(tab, text="Pre-run checklist")
        info.pack(fill="both", expand=True, padx=6, pady=3)
        check = tk.Text(info, height=5, font=(self.ui_font, 10), bg="#fffdf3",
                        fg=PALETTE["text"], relief="flat", borderwidth=0,
                        highlightthickness=1, highlightbackground=PALETTE["border"],
                        padx=10, pady=6)
        check.insert("end",
                     "1) Schedule built in ② and FEASIBLE; validate at low frequency first.   "
                     "2) Clamp mounted at the chosen point; rod near mid-stroke.\n"
                     "3) Alarm code 0 after connecting (clear it otherwise).   "
                     "4) First-time homing needs one drive power cycle.\n"
                     "5) Physical power switch within reach (emergency stop).   "
                     "6) Every run is archived under results/harvest_runs/.")
        check.configure(state="disabled")
        check.pack(fill="both", expand=True, padx=4, pady=4)

    def on_connect(self) -> None:
        if self.drv.connected:
            try:
                self.drv.stop()
            except Exception:  # noqa: BLE001
                pass
            self.drv.close()
            self.btn_conn.configure(text="Connect")
            self.btn_alarm.configure(state="disabled")
            self.btn_run.configure(state="disabled")
            self.log("Serial disconnected")
            return
        port = self.var_port.get().strip()
        # WSL: if the port is missing OR not openable (usbip nodes are root:root
        # 0600), auto-attach + load driver + fix perms before opening it.
        if port.startswith("/dev/tty") and not os.access(port, os.R_OK | os.W_OK):
            from orchard_fem.actuator.wsl_usb import ensure_usb_serial
            ok, msg = ensure_usb_serial(port, log=self.log)
            self.log(msg)
            if not ok:
                messagebox.showerror("Serial port unavailable", msg)
                return
        try:
            self.drv.connect(port, int(self.var_baud.get()),
                             self.var_parity.get(), int(self.var_stop.get()))
            alm = self.drv.alarm()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Connection failed", str(e))
            self.log(f"Connection failed: {e}")
            return
        self.btn_conn.configure(text="Disconnect")
        self.btn_alarm.configure(state="normal")
        if self.schedule is not None and self.schedule.feasible:
            self.btn_run.configure(state="normal")
        self.log("Serial connected"
                 + (f"; active alarm E-{alm:03d}, clear it first" if alm else "; no alarm"))

    def on_clear_alarm(self) -> None:
        try:
            alm = self.drv.clear_alarm()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Operation failed", str(e))
            return
        self.log("Alarm cleared" if alm == 0
                 else f"Alarm E-{alm:03d} persists (may need a power cycle)")

    def on_run(self) -> None:
        """RUN the schedule built in ② (1 stage if one excitation suffices, else N)."""
        if self._rig_running:
            return
        if self.schedule is not None and self.schedule.feasible:
            self.on_run_schedule()
        else:
            messagebox.showwarning(
                "Notice", "Run a simulation in ② first to build an executable schedule.")

    def on_run_schedule(self) -> None:
        if self._rig_running:
            return
        if self.schedule is None or not self.schedule.feasible:
            messagebox.showwarning("Notice", "Build an executable schedule in ② first.")
            return
        if not self.drv.connected:
            messagebox.showwarning("Notice", "Connect the serial port first.")
            return
        if not self._confirm(
                "Confirm schedule run",
                self.schedule.summary()
                + "\n\nThe actuator will run the stages above in sequence — proceed?",
                ok_text="Run schedule"):
            return
        self._rig_stop.clear()
        self._rig_running = True
        self.btn_run.configure(state="disabled")
        self.btn_rig_stop.configure(state="normal")
        schedule, drv = self.schedule, self.drv
        home, off, rev = (bool(self.var_home.get()),
                          float(self.var_homeoff.get()), bool(self.var_homerev.get()))
        calib = bool(self.var_calibrate.get())

        def worker() -> None:
            try:
                outcome = run_harvest_schedule_on_rig(
                    schedule, driver=drv,
                    home=home, home_offset_mm=off, home_reverse=rev, calibrate=calib,
                    status_cb=lambda m: self._post("log", m),
                    on_stage=lambda s: self._post(
                        "log", f"▶ Stage {s.index}: {s.plan.frequency_hz:.2f} Hz, "
                               f"stroke {s.plan.stroke_mm:.1f} mm, {s.plan.duration_s:g} s"),
                    should_stop=self._rig_stop.is_set,
                )
                self._post("rig_done", outcome)
            except Exception as e:  # noqa: BLE001
                self._post("rig_done", e)

        threading.Thread(target=worker, daemon=True).start()

    def on_rig_stop(self) -> None:
        self._rig_stop.set()
        try:
            self.drv.stop()           # disable the servo at once, don't wait for the poll
            self.log("Stop sent (servo disabled)")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Stop command failed",
                                 f"{e}\nUse the physical power switch immediately!")

    def _on_rig_done(self, payload) -> None:
        self._rig_running = False
        self.btn_rig_stop.configure(state="disabled")
        runnable = self.drv.connected and self.schedule is not None and self.schedule.feasible
        self.btn_run.configure(state="normal" if runnable else "disabled")
        if isinstance(payload, Exception):
            self.log(f"Run failed: {payload}")
            messagebox.showerror("Run failed", str(payload))
            self._save_run_record("error", str(payload))
            return
        labels = {"completed": "completed", "alarm_stop": "alarm stop",
                  "user_stop": "user stop"}
        self.log(f"Run finished: {labels.get(payload, payload)}")
        self._save_run_record(payload)

    def _save_run_record(self, outcome: str, detail: str = "") -> None:
        """运行档案:模型、调参序列、结果,便于追溯与论文数据整理。"""
        try:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_path": str(self.model_path) if self.model_path else
                              (self.result.model_path if self.result else ""),
                "schedule": (dataclasses.asdict(self.schedule)
                             if self.schedule else None),
                "outcome": outcome,
                "detail": detail,
            }
            path = RUNS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            self.log(f"Run record → {path.relative_to(REPO_ROOT)}")
        except Exception as e:  # noqa: BLE001
            self.log(f"Failed to save run record: {e}")

    # ------------------------------------------------------------------ #

    def on_close(self) -> None:
        self._sim_cancel.set()
        self._rig_stop.set()
        try:
            if self.drv.connected:
                self.drv.stop()
                self.drv.close()
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    HarvestConsole(root)
    root.mainloop()


if __name__ == "__main__":
    main()
