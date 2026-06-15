# -*- coding: utf-8 -*-
"""Harvest Console — 整链交互前端:树模型 → 仿真推荐 → 执行计划 → 电动缸执行.

四步流程(Notebook 标签页),共享底部日志:

① **树模型** — 选择 tree JSON(``trees/*.json``),加载并显示主要参数
   (:func:`orchard_fem.workflows.harvest_recommendation.summarize_orchard_model`)。
② **仿真推荐** — 后台线程运行
   :func:`~orchard_fem.workflows.harvest_recommendation.recommend_harvest_parameters`
   (FRF 扫频 → 共振 → 夹持×(f,A) Pareto,**叠加电动缸包络硬约束**),
   实时显示进度与调节步骤;结果表列出全部候选工作点(前沿/knee 标记),
   选中即可送入计划;可导出/导入 JSON(离线机仅执行)。
③ **执行计划** — 工作参数 → :class:`~orchard_fem.actuator.harvest_bridge.HarvestPlan`
   (行程/转速/周期数 + 可行性);「收紧到包络」一键自动调整越界参数。
④ **电动缸执行** — 串口连接、清报警、回中设置;执行 =
   :func:`~orchard_fem.actuator.ds5l1.run_harvest_plan_on_rig`
   (标定缓存起步 + 在线频率标定 + 报警轮询),停止按钮立即断使能;
   每次运行归档至 ``results/harvest_runs/``。

运行:``python -m orchard_fem.actuator.harvest_console``
(仿真需 dolfinx;无 dolfinx 的执行机可加载已导出的推荐 JSON 走 ③④。)
"""
from __future__ import annotations

import dataclasses
import json
import queue
import sys
import threading
import time
import tkinter as tk
from importlib import util as _importlib_util
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from orchard_fem.actuator.ds5l1 import DS5L1, run_harvest_plan_on_rig
from orchard_fem.actuator.harvest_bridge import (
    DS5L1Limits,
    HarvestPlan,
    plan_harvest_execution,
)
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
 

def _has(module: str) -> bool:
    return _importlib_util.find_spec(module) is not None


def _pick_ui_font(root: tk.Tk) -> str:
    """选一个本机可用、覆盖中文的界面字体并设为 Tk 默认。

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
        self.plan: HarvestPlan | None = None
        self.drv = DS5L1()
        self._q: queue.Queue = queue.Queue()
        self._sim_cancel = threading.Event()
        self._rig_stop = threading.Event()
        self._sim_running = False
        self._rig_running = False

        root.title("Orchard Harvest Console — 仿真 → DS5L1 执行")
        root.minsize(980, 720)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.ui_font = _pick_ui_font(root)
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("TLabelframe.Label", font=(self.ui_font, 9, "bold"),
                        foreground="#1b5e20")
        style.configure("Knee.Treeview", background="#e8f5e9")

        header = tk.Frame(root, bg="#1b5e20")
        header.pack(fill="x")
        tk.Label(header, text="Orchard Harvest Console",
                 font=(self.ui_font, 14, "bold"), bg="#1b5e20", fg="white",
                 ).pack(pady=(8, 0))
        env = (f"dolfinx {'✓' if _has('dolfinx') else '✗(仅可加载已导出结果)'}   "
               f"pyserial {'✓' if _has('serial') else '✗(无法连接电动缸)'}")
        tk.Label(header, text=env, font=(self.ui_font, 9),
                 bg="#1b5e20", fg="#c8e6c9").pack(pady=(0, 6))

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=6, pady=4)
        self._build_tab_model()
        self._build_tab_sim()
        self._build_tab_plan()
        self._build_tab_rig()

        bottom = ttk.Frame(root)
        bottom.pack(fill="x", padx=6, pady=(0, 6))
        self.progress = ttk.Progressbar(bottom, maximum=1.0)
        self.progress.pack(fill="x")
        self.log_text = tk.Text(bottom, height=9, state="disabled",
                                font=("Consolas", 9), bg="#fafafa")
        self.log_text.pack(fill="both", expand=True, pady=(3, 0))

        self.root.after(100, self._pump)

    # ------------------------------------------------------------------ #
    # 共享:日志 / 队列泵
    # ------------------------------------------------------------------ #

    def log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{time.strftime('%H:%M:%S')}  {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

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
        self.nb.add(tab, text=" ① 树模型 ")
        row = ttk.Frame(tab)
        row.pack(fill="x", padx=8, pady=8)
        self.var_model_path = tk.StringVar()
        ttk.Label(row, text="模型 JSON:").pack(side="left")
        ttk.Entry(row, textvariable=self.var_model_path).pack(
            side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="浏览…", command=self.on_browse_model).pack(side="left")
        ttk.Button(row, text="加载", command=self.on_load_model).pack(side="left", padx=6)

        box = ttk.LabelFrame(tab, text="模型主要参数")
        box.pack(fill="both", expand=True, padx=8, pady=4)
        self.model_info = tk.Text(box, state="disabled", font=(self.ui_font, 10),
                                  bg="#fcfcfc")
        self.model_info.pack(fill="both", expand=True, padx=4, pady=4)

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
            messagebox.showwarning("提示", "请先选择模型 JSON 文件")
            return
        try:
            from orchard_fem.io.loaders import load_orchard_model
            self.model = load_orchard_model(path)
            self.model_path = Path(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("加载失败", str(e))
            self.log(f"模型加载失败: {e}")
            return
        summary = summarize_orchard_model(self.model, path)
        self.model_info.configure(state="normal")
        self.model_info.delete("1.0", "end")
        self.model_info.insert("end", "\n".join(summary.lines()))
        self.model_info.configure(state="disabled")
        # 夹持候选自动填入仿真页
        try:
            labels = candidate_clamp_labels(self.model, RecommendationOptions())
        except Exception:  # noqa: BLE001 - 无 trunk 命名时回退到各枝根部
            labels = [f"{b.branch_id}@0.00" for b in self.model.branches]
        self.clamp_list.delete(0, "end")
        for label in labels:
            self.clamp_list.insert("end", label)
        self.clamp_list.selection_set(0, "end")
        self.log(f"已加载模型 {summary.name}({summary.n_branches} 枝, "
                 f"{summary.n_fruits} 果)")
        self.nb.select(1)

    # ------------------------------------------------------------------ #
    # ② 仿真推荐
    # ------------------------------------------------------------------ #

    def _build_tab_sim(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" ② 仿真推荐 ")

        opts = ttk.LabelFrame(tab, text="仿真设置")
        opts.pack(fill="x", padx=8, pady=6)
        pad = dict(padx=4, pady=3)
        self.var_band_lo = tk.DoubleVar(value=3.0)
        self.var_band_hi = tk.DoubleVar(value=20.0)
        self.var_steps = tk.IntVar(value=60)
        self.var_agrid = tk.StringVar(value="2.5, 5, 7.5, 10")
        self.var_dense = tk.BooleanVar(value=True)
        self.var_spacing = tk.DoubleVar(value=0.05)
        self.var_ddet = tk.DoubleVar(value=2.0)
        self.var_duration = tk.DoubleVar(value=10.0)
        self.var_coverage = tk.StringVar(value="branch")

        g = ttk.Frame(opts)
        g.pack(fill="x")
        ttk.Label(g, text="频带 Hz").grid(row=0, column=0, **pad)
        ttk.Entry(g, textvariable=self.var_band_lo, width=5).grid(row=0, column=1, **pad)
        ttk.Label(g, text="–").grid(row=0, column=2)
        ttk.Entry(g, textvariable=self.var_band_hi, width=5).grid(row=0, column=3, **pad)
        ttk.Label(g, text="扫频步数").grid(row=0, column=4, **pad)
        ttk.Entry(g, textvariable=self.var_steps, width=5).grid(row=0, column=5, **pad)
        ttk.Label(g, text="幅值候选 A (mm)").grid(row=0, column=6, **pad)
        ttk.Entry(g, textvariable=self.var_agrid, width=16).grid(row=0, column=7, **pad)
        ttk.Label(g, text="覆盖率口径").grid(row=0, column=8, **pad)
        ttk.Combobox(g, textvariable=self.var_coverage, width=7, state="readonly",
                     values=["branch", "fruit"]).grid(row=0, column=9, **pad)
        ttk.Checkbutton(g, text="密集布果, 间距", variable=self.var_dense,
                        ).grid(row=1, column=0, columnspan=2, **pad)
        ttk.Entry(g, textvariable=self.var_spacing, width=5).grid(row=1, column=2, columnspan=2, **pad)
        ttk.Label(g, text="脱落位移 mm").grid(row=1, column=4, **pad)
        ttk.Entry(g, textvariable=self.var_ddet, width=5).grid(row=1, column=5, **pad)
        ttk.Label(g, text="作业时长 s").grid(row=1, column=6, **pad)
        ttk.Entry(g, textvariable=self.var_duration, width=6).grid(row=1, column=7, **pad)

        mid = ttk.Frame(tab)
        mid.pack(fill="x", padx=8)
        clamp_box = ttk.LabelFrame(mid, text="候选夹持位置(多选)")
        clamp_box.pack(side="left", fill="y")
        self.clamp_list = tk.Listbox(clamp_box, selectmode="multiple", height=6,
                                     exportselection=False, width=22)
        self.clamp_list.pack(padx=4, pady=4)
        btns = ttk.Frame(mid)
        btns.pack(side="left", fill="x", padx=10)
        self.btn_sim = ttk.Button(btns, text="▶ 运行仿真", command=self.on_run_sim)
        self.btn_sim.pack(fill="x", pady=2)
        self.btn_sim_cancel = ttk.Button(btns, text="取消", state="disabled",
                                         command=lambda: self._sim_cancel.set())
        self.btn_sim_cancel.pack(fill="x", pady=2)
        ttk.Button(btns, text="导出推荐结果 JSON…",
                   command=self.on_export_result).pack(fill="x", pady=2)
        ttk.Button(btns, text="加载推荐结果 JSON…",
                   command=self.on_import_result).pack(fill="x", pady=2)
        self.lbl_recommend = ttk.Label(mid, text="—", font=(self.ui_font, 10, "bold"),
                                       foreground="#1b5e20", wraplength=420)
        self.lbl_recommend.pack(side="left", padx=10)

        table_box = ttk.LabelFrame(tab, text="候选工作点(★=推荐 knee,◆=Pareto 前沿;灰=超出电动缸包络)")
        table_box.pack(fill="both", expand=True, padx=8, pady=6)
        cols = ("clamp", "f", "A", "stroke", "cov", "stress", "env", "mark")
        self.tree = ttk.Treeview(table_box, columns=cols, show="headings", height=10)
        headings = [("clamp", "夹持", 130), ("f", "f (Hz)", 70), ("A", "A (mm)", 70),
                    ("stroke", "行程 (mm)", 80), ("cov", "覆盖率", 70),
                    ("stress", "σ主干 (MPa)", 90), ("env", "包络内", 60), ("mark", "标记", 60)]
        for cid, text, width in headings:
            self.tree.heading(cid, text=text)
            self.tree.column(cid, width=width, anchor="center")
        vsb = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")
        self.tree.tag_configure("knee", background="#c8e6c9")
        self.tree.tag_configure("front", background="#e8f5e9")
        self.tree.tag_configure("infeasible", foreground="#9e9e9e")
        self.tree.bind("<Double-1>", lambda _e: self.on_adopt_point())
        ttk.Button(tab, text="采用选中工作点 → ③ 执行计划",
                   command=self.on_adopt_point).pack(pady=(0, 6))

    def _sim_options(self) -> RecommendationOptions:
        a_grid = tuple(float(x) for x in
                       self.var_agrid.get().replace("，", ",").split(",") if x.strip())
        return RecommendationOptions(
            band_hz=(float(self.var_band_lo.get()), float(self.var_band_hi.get())),
            sweep_steps=int(self.var_steps.get()),
            amplitude_grid_mm=a_grid,
            dense_fruit_spacing=(float(self.var_spacing.get())
                                 if self.var_dense.get() else None),
            detachment_displacement_m=float(self.var_ddet.get()) / 1000.0,
            coverage_mode=self.var_coverage.get(),
            duration_s=float(self.var_duration.get()),
            limits=LIMITS,
        )

    def on_run_sim(self) -> None:
        if self.model is None:
            messagebox.showwarning("提示", "请先在 ① 加载树模型")
            return
        if self._sim_running:
            return
        if not _has("dolfinx"):
            messagebox.showerror(
                "缺少求解后端",
                "本机没有 dolfinx(FEniCSx),无法运行仿真。\n"
                "请在 orchard-fenicsx 环境运行本程序,或加载已导出的推荐结果 JSON。")
            return
        try:
            options = self._sim_options()
            selected = [self.clamp_list.get(i) for i in self.clamp_list.curselection()]
            if not selected:
                raise ValueError("请至少选择一个夹持位置")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("参数错误", str(e))
            return
        self._sim_cancel.clear()
        self._sim_running = True
        self.btn_sim.configure(state="disabled")
        self.btn_sim_cancel.configure(state="normal")
        self.log(f"开始仿真推荐({len(selected)} 个夹持候选)…")

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
            self.log(f"仿真{'已取消' if msg == 'cancelled' else f'失败: {msg}'}")
            if msg != "cancelled":
                messagebox.showerror("仿真失败", msg)
            return
        self.result = payload
        self._fill_result_table(payload)
        self.log(f"仿真完成,用时 {payload.elapsed_s:.0f} s")

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
                    values=(p.clamp_label, f"{p.frequency_hz:.2f}",
                            f"{p.amplitude_mm:g}", f"{p.stroke_mm:g}",
                            f"{p.coverage:.2f}", f"{p.trunk_stress_pa / 1e6:.2f}",
                            "是" if p.rig_feasible else "否", mark))
        rec = result.recommended
        if rec is not None:
            self.lbl_recommend.configure(text=(
                f"推荐: 夹持 {rec.clamp_label}  f={rec.frequency_hz:.2f} Hz  "
                f"A={rec.amplitude_mm:g} mm(行程 {rec.stroke_mm:g} mm)  "
                f"覆盖率 {rec.coverage:.2f}  σ={rec.trunk_stress_pa / 1e6:.2f} MPa  "
                f"时长 {result.duration_s:g} s"))

    def on_adopt_point(self) -> None:
        if self.result is None:
            messagebox.showwarning("提示", "还没有仿真结果")
            return
        sel = self.tree.selection()
        if sel:
            ci, pi = (int(x) for x in sel[0].split(":"))
            point = self.result.clamps[ci].points[pi]
        else:
            point = self.result.recommended
            if point is None:
                messagebox.showwarning("提示", "无推荐点,请在表中选择")
                return
        if not point.rig_feasible:
            if not messagebox.askyesno(
                    "超出包络", "该工作点超出电动缸包络,采用后需在 ③ 收紧。继续?"):
                return
        self.var_freq.set(point.frequency_hz)
        self.var_amp.set(point.amplitude_mm)
        self.var_plan_duration.set(self.result.duration_s)
        self.var_label.set(point.clamp_label)
        self.log(f"已采用工作点 {point.clamp_label} f={point.frequency_hz:g} Hz "
                 f"A={point.amplitude_mm:g} mm → ③")
        self.nb.select(2)
        self.on_make_plan()

    def on_export_result(self) -> None:
        if self.result is None:
            messagebox.showwarning("提示", "还没有仿真结果")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile="recommendation.json")
        if path:
            self.result.save_json(path)
            self.log(f"推荐结果已导出 → {path}")

    def on_import_result(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Recommendation JSON", "*.json")])
        if not path:
            return
        try:
            self.result = RecommendationResult.load_json(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("加载失败", str(e))
            return
        self._fill_result_table(self.result)
        for step in self.result.steps:
            self.log(f"[导入] {step}")
        self.log(f"已加载推荐结果({self.result.model_name})")

    # ------------------------------------------------------------------ #
    # ③ 执行计划
    # ------------------------------------------------------------------ #

    def _build_tab_plan(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" ③ 执行计划 ")
        pad = dict(padx=5, pady=4)
        box = ttk.LabelFrame(tab, text="工作参数(可手动修改)")
        box.pack(fill="x", padx=8, pady=8)
        self.var_freq = tk.DoubleVar(value=2.0)
        self.var_amp = tk.DoubleVar(value=2.5)
        self.var_plan_duration = tk.DoubleVar(value=10.0)
        self.var_accel = tk.IntVar(value=10)
        self.var_label = tk.StringVar(value="")
        ttk.Label(box, text="频率 f (Hz)").grid(row=0, column=0, **pad)
        ttk.Entry(box, textvariable=self.var_freq, width=8).grid(row=0, column=1, **pad)
        ttk.Label(box, text="位移幅值 A (mm, 半峰峰)").grid(row=0, column=2, **pad)
        ttk.Entry(box, textvariable=self.var_amp, width=8).grid(row=0, column=3, **pad)
        ttk.Label(box, text="→ 行程 S=2A").grid(row=0, column=4, **pad)
        self.lbl_stroke = ttk.Label(box, text="5 mm", font=(self.ui_font, 9, "bold"))
        self.lbl_stroke.grid(row=0, column=5, **pad)
        ttk.Label(box, text="时长 (s)").grid(row=1, column=0, **pad)
        ttk.Entry(box, textvariable=self.var_plan_duration, width=8).grid(row=1, column=1, **pad)
        ttk.Label(box, text="加减速 (ms)").grid(row=1, column=2, **pad)
        ttk.Entry(box, textvariable=self.var_accel, width=8).grid(row=1, column=3, **pad)
        ttk.Label(box, text="激励位置标签").grid(row=1, column=4, **pad)
        ttk.Entry(box, textvariable=self.var_label, width=16).grid(row=1, column=5, **pad)

        btns = ttk.Frame(tab)
        btns.pack(fill="x", padx=8)
        ttk.Button(btns, text="生成执行计划", command=self.on_make_plan,
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="收紧到包络", command=self.on_clamp_to_envelope,
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="导出参数 JSON…", command=self.on_export_params,
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="加载参数 JSON…", command=self.on_import_params,
                   ).pack(side="left", padx=4)

        plan_box = ttk.LabelFrame(tab, text="计划与可行性")
        plan_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.plan_info = tk.Text(plan_box, state="disabled", height=10,
                                 font=("Consolas", 10), bg="#fcfcfc")
        self.plan_info.pack(fill="both", expand=True, padx=4, pady=4)

    def _show_plan(self, text: str) -> None:
        self.plan_info.configure(state="normal")
        self.plan_info.delete("1.0", "end")
        self.plan_info.insert("end", text)
        self.plan_info.configure(state="disabled")

    def on_make_plan(self) -> None:
        try:
            amp = float(self.var_amp.get())
            self.lbl_stroke.configure(text=f"{2 * amp:g} mm")
            self.plan = plan_harvest_execution(
                frequency_hz=float(self.var_freq.get()),
                clamp_peak_to_peak_mm=2.0 * amp,
                duration_s=float(self.var_plan_duration.get()),
                accel_ms=int(self.var_accel.get()),
                excitation_label=self.var_label.get(),
                limits=LIMITS,
            )
        except Exception as e:  # noqa: BLE001
            self.plan = None
            messagebox.showerror("参数错误", str(e))
            return
        self._show_plan(self.plan.summary())
        self.log("执行计划: " + ("可行" if self.plan.feasible else "不可行(见 ③)"))
        if self.plan.feasible:
            self.btn_rig_run.configure(state="normal" if self.drv.connected else "disabled")

    def on_clamp_to_envelope(self) -> None:
        """自动调整:行程超限 → A=上限/2;频率不可达 → 降到该行程可达上限的95%。"""
        amp = float(self.var_amp.get())
        freq = float(self.var_freq.get())
        adjusted = []
        if 2.0 * amp > LIMITS.max_stroke_mm:
            amp = LIMITS.max_stroke_mm / 2.0
            adjusted.append(f"A → {amp:g} mm(行程上限 {LIMITS.max_stroke_mm:g} mm)")
        stroke = 2.0 * amp
        if LIMITS.seed_rpm(stroke, freq) is None:
            freq = round(0.95 * LIMITS.max_frequency_at_stroke(stroke), 2)
            adjusted.append(f"f → {freq:g} Hz(该行程可达上限的 95%)")
        if not adjusted:
            self.log("参数已在包络内,无需调整")
            return
        self.var_amp.set(amp)
        self.var_freq.set(freq)
        for a in adjusted:
            self.log("包络收紧: " + a)
        self.on_make_plan()

    def _params_dict(self) -> dict:
        return {
            "frequency_hz": float(self.var_freq.get()),
            "displacement_amplitude_m": float(self.var_amp.get()) / 1000.0,
            "duration_s": float(self.var_plan_duration.get()),
            "excitation_label": self.var_label.get(),
        }

    def on_export_params(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile="harvest_params.json")
        if path:
            Path(path).write_text(json.dumps(self._params_dict(), ensure_ascii=False,
                                             indent=1), encoding="utf-8")
            self.log(f"参数已导出 → {path}")

    def on_import_params(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Params JSON", "*.json")])
        if not path:
            return
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            self.var_freq.set(float(d["frequency_hz"]))
            if "displacement_amplitude_m" in d:
                self.var_amp.set(float(d["displacement_amplitude_m"]) * 1000.0)
            elif "clamp_peak_to_peak_mm" in d:
                self.var_amp.set(float(d["clamp_peak_to_peak_mm"]) / 2.0)
            self.var_plan_duration.set(float(d.get("duration_s", 10.0)))
            self.var_label.set(str(d.get("excitation_label", "")))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("加载失败", str(e))
            return
        self.on_make_plan()

    # ------------------------------------------------------------------ #
    # ④ 电动缸执行
    # ------------------------------------------------------------------ #

    def _build_tab_rig(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" ④ 电动缸执行 ")
        pad = dict(padx=5, pady=4)

        conn = ttk.LabelFrame(tab, text="串口连接(RS232 出厂 19200-8-E-1)")
        conn.pack(fill="x", padx=8, pady=6)
        default_port = "COM8" if sys.platform.startswith("win") else "/dev/ttyUSB0"
        self.var_port = tk.StringVar(value=default_port)
        self.var_baud = tk.StringVar(value="19200")
        self.var_parity = tk.StringVar(value="E")
        self.var_stop = tk.StringVar(value="1")
        ttk.Label(conn, text="端口").grid(row=0, column=0, **pad)
        ttk.Entry(conn, textvariable=self.var_port, width=12).grid(row=0, column=1, **pad)
        ttk.Label(conn, text="波特率").grid(row=0, column=2, **pad)
        ttk.Combobox(conn, textvariable=self.var_baud, width=7,
                     values=["9600", "19200", "38400"]).grid(row=0, column=3, **pad)
        ttk.Label(conn, text="校验").grid(row=0, column=4, **pad)
        ttk.Combobox(conn, textvariable=self.var_parity, width=3,
                     values=["E", "O", "N"]).grid(row=0, column=5, **pad)
        ttk.Label(conn, text="停止位").grid(row=0, column=6, **pad)
        ttk.Combobox(conn, textvariable=self.var_stop, width=3,
                     values=["1", "2"]).grid(row=0, column=7, **pad)
        self.btn_conn = ttk.Button(conn, text="连接", command=self.on_connect)
        self.btn_conn.grid(row=0, column=8, **pad)
        self.btn_alarm = ttk.Button(conn, text="清除报警", state="disabled",
                                    command=self.on_clear_alarm)
        self.btn_alarm.grid(row=0, column=9, **pad)

        home = ttk.LabelFrame(tab, text="回中(触停回零;P9-21 首次启用需驱动器断电重启)")
        home.pack(fill="x", padx=8, pady=6)
        self.var_home = tk.BooleanVar(value=True)
        self.var_homeoff = tk.DoubleVar(value=25.8)
        self.var_homerev = tk.BooleanVar(value=False)
        ttk.Checkbutton(home, text="执行前自动回中", variable=self.var_home,
                        ).pack(side="left", **pad)
        ttk.Label(home, text="限位→中点偏移 (mm)").pack(side="left", **pad)
        ttk.Entry(home, textvariable=self.var_homeoff, width=6).pack(side="left", **pad)
        ttk.Checkbutton(home, text="反向触停", variable=self.var_homerev,
                        ).pack(side="left", **pad)

        ctl = ttk.Frame(tab)
        ctl.pack(fill="x", padx=8, pady=10)
        self.btn_rig_run = tk.Button(
            ctl, text="▶  执 行 计 划", font=(self.ui_font, 13, "bold"),
            bg="#2e7d32", fg="white", relief="flat", state="disabled",
            disabledforeground="#9e9e9e", command=self.on_rig_run)
        self.btn_rig_run.pack(side="left", expand=True, fill="x", padx=8, ipady=10)
        self.btn_rig_stop = tk.Button(
            ctl, text="■  停 止", font=(self.ui_font, 13, "bold"),
            bg="#c62828", fg="white", relief="flat", state="disabled",
            disabledforeground="#9e9e9e", command=self.on_rig_stop)
        self.btn_rig_stop.pack(side="left", expand=True, fill="x", padx=8, ipady=10)

        info = ttk.LabelFrame(tab, text="执行前检查")
        info.pack(fill="both", expand=True, padx=8, pady=4)
        check = tk.Text(info, height=7, font=(self.ui_font, 10), bg="#fffde7")
        check.insert("end",
                     "1) 计划可行(③ 生成且 FEASIBLE),低频(1–2 Hz)先行验证再上推荐频率;\n"
                     "2) 夹持机构按推荐位置固定,缸杆大致位于行程中点;\n"
                     "3) 连接后报警码为 0(否则先清除);\n"
                     "4) 首次启用回零需给驱动器断电重启一次;\n"
                     "5) 桌面物理电源开关随手可及(急停);\n"
                     "6) 执行中在线标定会微调转速,结果自动存入标定表;\n"
                     "7) 每次运行记录归档于 results/harvest_runs/。")
        check.configure(state="disabled")
        check.pack(fill="both", expand=True, padx=4, pady=4)

    def on_connect(self) -> None:
        if self.drv.connected:
            try:
                self.drv.stop()
            except Exception:  # noqa: BLE001
                pass
            self.drv.close()
            self.btn_conn.configure(text="连接")
            self.btn_alarm.configure(state="disabled")
            self.btn_rig_run.configure(state="disabled")
            self.log("已断开串口")
            return
        try:
            self.drv.connect(self.var_port.get().strip(), int(self.var_baud.get()),
                             self.var_parity.get(), int(self.var_stop.get()))
            alm = self.drv.alarm()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("连接失败", str(e))
            self.log(f"连接失败: {e}")
            return
        self.btn_conn.configure(text="断开")
        self.btn_alarm.configure(state="normal")
        if self.plan is not None and self.plan.feasible:
            self.btn_rig_run.configure(state="normal")
        self.log("串口已连接"
                 + (f",当前报警 E-{alm:03d},请先清除" if alm else ",无报警"))

    def on_clear_alarm(self) -> None:
        try:
            alm = self.drv.clear_alarm()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("操作失败", str(e))
            return
        self.log("报警已清除" if alm == 0 else f"报警 E-{alm:03d} 仍在(可能需断电重启)")

    def on_rig_run(self) -> None:
        if self._rig_running:
            return
        if self.plan is None or not self.plan.feasible:
            messagebox.showwarning("提示", "请先在 ③ 生成可行的执行计划")
            return
        if not self.drv.connected:
            messagebox.showwarning("提示", "请先连接串口")
            return
        if not messagebox.askyesno(
                "确认执行",
                self.plan.summary() + "\n\n确认电动缸即将运动,夹持/场地安全?"):
            return
        self._rig_stop.clear()
        self._rig_running = True
        self.btn_rig_run.configure(state="disabled")
        self.btn_rig_stop.configure(state="normal")
        plan, drv = self.plan, self.drv
        home, off, rev = (bool(self.var_home.get()),
                          float(self.var_homeoff.get()), bool(self.var_homerev.get()))

        def worker() -> None:
            try:
                outcome = run_harvest_plan_on_rig(
                    plan, driver=drv,
                    home=home, home_offset_mm=off, home_reverse=rev,
                    status_cb=lambda m: self._post("log", m),
                    should_stop=self._rig_stop.is_set,
                )
                self._post("rig_done", outcome)
            except Exception as e:  # noqa: BLE001
                self._post("rig_done", e)

        threading.Thread(target=worker, daemon=True).start()

    def on_rig_stop(self) -> None:
        self._rig_stop.set()
        try:
            self.drv.stop()           # 立即断使能,不等轮询周期
            self.log("已发送停止(伺服断使能)")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("停止指令失败", f"{e}\n请立即使用物理电源开关!")

    def _on_rig_done(self, payload) -> None:
        self._rig_running = False
        self.btn_rig_stop.configure(state="disabled")
        self.btn_rig_run.configure(
            state="normal" if (self.drv.connected and self.plan
                               and self.plan.feasible) else "disabled")
        if isinstance(payload, Exception):
            self.log(f"执行失败: {payload}")
            messagebox.showerror("执行失败", str(payload))
            self._save_run_record("error", str(payload))
            return
        zh = {"completed": "完成", "alarm_stop": "报警停机", "user_stop": "人工停止"}
        self.log(f"执行结束: {zh.get(payload, payload)}")
        self._save_run_record(payload)

    def _save_run_record(self, outcome: str, detail: str = "") -> None:
        """运行档案:模型、计划、结果,便于追溯与论文数据整理。"""
        try:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_path": str(self.model_path) if self.model_path else
                              (self.result.model_path if self.result else ""),
                "plan": dataclasses.asdict(self.plan) if self.plan else None,
                "outcome": outcome,
                "detail": detail,
            }
            path = RUNS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            self.log(f"运行记录 → {path.relative_to(REPO_ROOT)}")
        except Exception as e:  # noqa: BLE001
            self.log(f"运行记录保存失败: {e}")

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
