# -*- coding: utf-8 -*-
"""DS5L1 伺服电动缸 往复激振控制(GUI 版).

Tk front-end over the bench-verified driver
(:class:`orchard_fem.actuator.ds5l1.DS5L1`); hardware facts and the Modbus
layer live there, the frequency model / envelope in
:class:`orchard_fem.actuator.harvest_bridge.DS5L1Limits`, and the
simulation→rig linkage in
:func:`orchard_fem.actuator.ds5l1.run_harvest_plan_on_rig`.

运行:``python -m orchard_fem.actuator.vibration_gui``(需 pyserial + tkinter)

频率:段转速决定。起步估计来自 ``1/(2f)=6S/rpm+C`` 模型,实际频率需用波形
      采集标定;界面提供在线标定(运行中实测频率,迭代修正转速,存入标定表
      ``config/ds5l1_freq_calib.json`` 复用,与联动接口共享)。
安全:① 先低频(1~2Hz)验证再逐步升频;② 启动前回中使缸位于行程中点;
      ③ 关窗/异常自动发停机;④ 桌面保留物理电源开关作急停。
"""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk, messagebox

from orchard_fem.actuator.ds5l1 import (
    DS5L1,
    PULSES_PER_MM,
    calib_key,
    load_calib,
    save_calib,
)
from orchard_fem.actuator.harvest_bridge import DS5L1Limits

# 机器包络/频率模型(与仿真联动层共用同一份定义)
LIMITS = DS5L1Limits()


# ---------------- 前端界面 ----------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.drv = DS5L1()
        self.running = False
        self.calib = load_calib()   # (S,f)→rpm 在线标定表,存 config/ds5l1_freq_calib.json
        self.mid_enc = None         # 本次会话中点的编码器值;有值→无接触静音回中
        root.title("DS5L1 Reciprocating Vibration Control")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.minsize(780, 480)
        root.configure(bg="#f4f6f4")
        pad = dict(padx=8, pady=5)

        # --- 主题样式 ---
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"),
                        foreground="#1b5e20")
        style.configure("TLabel", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("Info.TLabel", font=("Segoe UI", 9, "italic"),
                        foreground="#455a64")
        style.configure("Status.TLabel", font=("Segoe UI", 9),
                        background="#e8f0e8", foreground="#263238")

        # --- 标题栏(单位署名)---
        header = tk.Frame(root, bg="#1b5e20")
        header.pack(fill="x")
        tk.Label(header, text="DS5L1 Reciprocating Vibration Control",
                 font=("Segoe UI", 15, "bold"),
                 bg="#1b5e20", fg="white").pack(pady=(10, 1))
        tk.Label(header,
                 text="Key Laboratory of State Forestry Administration "
                      "on Forestry Equipment and Automation",
                 font=("Segoe UI", 9),
                 bg="#1b5e20", fg="#c8e6c9").pack(pady=(0, 10))

        # --- 串口区 ---
        f1 = ttk.LabelFrame(root, text="1. Serial Connection (USB-RS232)")
        f1.pack(fill="x", **pad)
        self.var_port = tk.StringVar(value="COM8")
        self.var_baud = tk.StringVar(value="19200")
        self.var_parity = tk.StringVar(value="E")
        self.var_stop = tk.StringVar(value="1")
        ttk.Label(f1, text="Port").grid(row=0, column=0, **pad)
        ttk.Entry(f1, textvariable=self.var_port, width=10).grid(row=0, column=1, **pad)
        ttk.Label(f1, text="Baud").grid(row=0, column=2, **pad)
        ttk.Combobox(f1, textvariable=self.var_baud, width=8,
                     values=["9600", "19200", "38400", "57600", "115200"]).grid(row=0, column=3, **pad)
        ttk.Label(f1, text="Parity").grid(row=0, column=4, **pad)
        ttk.Combobox(f1, textvariable=self.var_parity, width=4,
                     values=["E", "O", "N"]).grid(row=0, column=5, **pad)
        ttk.Label(f1, text="Stop bits").grid(row=0, column=6, **pad)
        ttk.Combobox(f1, textvariable=self.var_stop, width=4,
                     values=["1", "2"]).grid(row=0, column=7, **pad)
        self.btn_conn = ttk.Button(f1, text="Connect", command=self.on_connect)
        self.btn_conn.grid(row=0, column=8, **pad)

        # --- 初始化区 ---
        f2 = ttk.LabelFrame(root, text="2. Initialization (servo must be disabled)")
        f2.pack(fill="x", **pad)
        self.var_wait = tk.IntVar(value=0)
        ttk.Radiobutton(f2, text="Wait for positioning (exact amplitude)",
                        variable=self.var_wait, value=0).pack(side="left", **pad)
        ttk.Radiobutton(f2, text="No wait (amplitude derates with frequency)",
                        variable=self.var_wait, value=1).pack(side="left", **pad)
        self.btn_init = ttk.Button(f2, text="Initialize", command=self.on_init,
                                   state="disabled")
        self.btn_init.pack(side="right", **pad)

        # --- 回中区 ---
        f2b = ttk.LabelFrame(root, text="3. Centering (touch-probe homing to mid-stroke)")
        f2b.pack(fill="x", **pad)
        self.var_homedir = tk.StringVar(value="Forward")
        self.var_homeoff = tk.DoubleVar(value=25.8)   # 实测两限位间行程51.67mm的一半
        self.var_autocenter = tk.BooleanVar(value=True)
        ttk.Label(f2b, text="Probe direction").pack(side="left", **pad)
        ttk.Combobox(f2b, textvariable=self.var_homedir, width=8,
                     values=["Forward", "Reverse"], state="readonly").pack(side="left", **pad)
        ttk.Label(f2b, text="Offset to mid (mm)").pack(side="left", **pad)
        ttk.Entry(f2b, textvariable=self.var_homeoff, width=6).pack(side="left", **pad)
        ttk.Checkbutton(f2b, text="Auto-center before start",
                        variable=self.var_autocenter).pack(side="left", **pad)
        self.btn_alarm = ttk.Button(f2b, text="Clear Alarm", command=self.on_clear_alarm,
                                    state="disabled")
        self.btn_alarm.pack(side="right", **pad)
        self.btn_center = ttk.Button(f2b, text="Center", command=self.on_center,
                                     state="disabled")
        self.btn_center.pack(side="right", **pad)

        # --- 激振参数区 ---
        f3 = ttk.LabelFrame(root, text="4. Vibration Parameters")
        f3.pack(fill="x", **pad)
        self.var_stroke = tk.DoubleVar(value=3.0)
        self.var_freq = tk.DoubleVar(value=2.0)
        self.var_accel = tk.IntVar(value=10)
        ttk.Label(f3, text="Stroke S (mm)").grid(row=0, column=0, **pad)
        ttk.Spinbox(f3, textvariable=self.var_stroke, width=8,
                    from_=1.0, to=LIMITS.max_stroke_mm, increment=1.0,
                    command=self.on_live_adjust).grid(row=0, column=1, **pad)
        ttk.Label(f3, text="Frequency f (Hz)").grid(row=0, column=2, **pad)
        ttk.Spinbox(f3, textvariable=self.var_freq, width=8,
                    from_=1.0, to=LIMITS.max_freq_hz, increment=1.0,
                    command=self.on_live_adjust).grid(row=0, column=3, **pad)
        self.btn_calib = ttk.Button(f3, text="Calibrate Frequency (~30 s run)",
                                    command=self.on_calib, state="disabled")
        self.btn_calib.grid(row=1, column=0, columnspan=2, **pad)
        ttk.Label(f3, text="Accel/Decel (ms)").grid(row=1, column=2, **pad)
        ttk.Entry(f3, textvariable=self.var_accel, width=8).grid(row=1, column=3, **pad)
        self.lbl_calc = ttk.Label(f3, text="—", style="Info.TLabel")
        self.lbl_calc.grid(row=2, column=0, columnspan=4, sticky="w", **pad)
        self.btn_write = ttk.Button(f3, text="Apply Parameters",
                                    command=self.on_write, state="disabled")
        self.btn_write.grid(row=0, column=4, rowspan=2, **pad)

        # --- 启停区 ---
        f4 = tk.Frame(root, bg="#f4f6f4")
        f4.pack(fill="x", **pad)
        self.btn_start = tk.Button(f4, text="▶  START", font=("Segoe UI", 14, "bold"),
                                   bg="#2e7d32", fg="white",
                                   activebackground="#388e3c", activeforeground="white",
                                   disabledforeground="#9e9e9e", relief="flat",
                                   cursor="hand2", state="disabled",
                                   command=self.on_start)
        self.btn_start.pack(side="left", expand=True, fill="x", padx=10, pady=8, ipady=12)
        self.btn_stop = tk.Button(f4, text="■  STOP", font=("Segoe UI", 14, "bold"),
                                  bg="#c62828", fg="white",
                                  activebackground="#d32f2f", activeforeground="white",
                                  disabledforeground="#9e9e9e", relief="flat",
                                  cursor="hand2", state="disabled",
                                  command=self.on_stop)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=10, pady=8, ipady=12)

        self.var_status = tk.StringVar(value="Disconnected")
        ttk.Label(root, textvariable=self.var_status, style="Status.TLabel",
                  relief="sunken", anchor="w",
                  padding=(8, 3)).pack(fill="x", side="bottom")

    # ---- 工具 ----
    def status(self, txt):
        self.var_status.set(txt)
        self.root.update_idletasks()

    def guard(fn):
        def wrap(self, *a, **k):
            try:
                fn(self, *a, **k)
            except Exception as e:
                try:
                    if self.drv.connected:
                        self.drv.stop()
                except Exception:
                    pass
                messagebox.showerror("Error", str(e))
                self.status(f"Error: {e}")
        return wrap

    def calc(self):
        s = float(self.var_stroke.get())
        f = float(self.var_freq.get())
        if not (0 < s <= LIMITS.max_stroke_mm):
            raise ValueError(f"Stroke must be within 0–{LIMITS.max_stroke_mm:g} mm")
        if not (0 < f <= LIMITS.max_freq_hz):
            raise ValueError(f"Frequency must be within 0–{LIMITS.max_freq_hz:g} Hz")
        cached = self.calib.get(calib_key(s, f))
        if cached:
            rpm = float(cached["rpm"])
            src = f"calibrated ({cached['f_act']:.2f} Hz, {cached.get('date', '')})"
        else:
            rpm = LIMITS.seed_rpm(s, f)
            if rpm is None:
                raise ValueError(
                    f"Target {f:g} Hz exceeds the estimated limit "
                    f"(~{LIMITS.max_frequency_at_stroke(s):.1f} Hz at stroke {s:g} mm). "
                    f"Reduce frequency or stroke, or tune servo gains.")
            src = "model estimate — calibration recommended"
        pulses = round(s * PULSES_PER_MM)
        return s, f, rpm, pulses, src

    # ---- 按钮回调 ----
    @guard
    def on_connect(self):
        if self.drv.connected:
            self.drv.stop()
            self.drv.close()
            self.btn_conn.config(text="Connect")
            for b in (self.btn_init, self.btn_write, self.btn_start, self.btn_stop,
                      self.btn_center, self.btn_alarm, self.btn_calib):
                b.config(state="disabled")
            self.status("Disconnected")
            return
        self.drv.connect(self.var_port.get().strip(), int(self.var_baud.get()),
                         self.var_parity.get(), int(self.var_stop.get()))
        self.mid_enc = None   # 重连后驱动器可能断过电,编码器多圈基准不可信
        self.btn_conn.config(text="Disconnect")
        self.btn_init.config(state="normal")
        self.btn_center.config(state="normal")
        self.btn_alarm.config(state="normal")
        alm = self.drv.alarm()
        if alm:
            self.status(f"Connected. Active alarm E-{alm:03d} — use [Clear Alarm]")
        else:
            self.status("Connected. Run [Initialize] first (servo must be disabled)")

    @guard
    def on_init(self):
        msg = self.drv.init_mode(self.var_wait.get())
        self.btn_write.config(state="normal")
        self.btn_calib.config(state="normal")
        self.status("Initialized | " + msg)
        messagebox.showinfo(
            "Initialization Complete",
            msg + "\n\nNext steps:\n1) Verify the rod is near mid-stroke\n"
                  "2) Set stroke/frequency, then [Apply Parameters]\n3) [START]\n"
                  "Validate at low frequency (1–2 Hz) before increasing.")

    def on_live_adjust(self):
        """行程/频率上下箭头回调(步进±1):运行中即时写入生效,停止时仅预览。"""
        try:
            s, f, rpm, pulses, src = self.calc()
        except Exception as e:
            self.status(f"Invalid parameters: {e}")
            return
        if self.running:
            try:
                self.drv.set_vibration(s, rpm, int(self.var_accel.get()))
            except Exception as e:
                try:
                    self.drv.stop()
                except Exception:
                    pass
                messagebox.showerror("Error", f"Live adjust failed; servo stopped: {e}")
                self.running = False
                self.btn_start.config(state="normal")
                self.btn_stop.config(state="disabled")
                self.status(f"Live adjust failed: {e}")
                return
            self.lbl_calc.config(text=f"Pulses ±{pulses}  Speed {rpm:.0f} rpm  ({src})")
            self.status(f"● Adjusted: S={s:g} mm  f={f:g} Hz  {rpm:.0f} rpm ({src})")
        else:
            self.lbl_calc.config(
                text=f"Preview: pulses ±{pulses}, {rpm:.0f} rpm ({src}) — "
                     f"press [Apply Parameters]")

    @guard
    def on_write(self):
        s, f, rpm, pulses, src = self.calc()
        self.drv.set_vibration(s, rpm, int(self.var_accel.get()))
        self.lbl_calc.config(text=f"Pulses ±{pulses}  Speed {rpm:.0f} rpm  ({src})")
        self.btn_start.config(state="normal")
        self.status(f"Parameters applied: S={s:g} mm  f={f:g} Hz  {rpm:.0f} rpm ({src})")

    @guard
    def on_calib(self):
        """在线标定:运行中实测频率,迭代修正转速至命中目标,结果入表复用。"""
        if self.running:
            raise RuntimeError("Stop vibration before calibrating")
        s, f_t, rpm, _pulses, _src = self.calc()   # 起始转速:缓存或模型
        accel = int(self.var_accel.get())
        if not self._ensure_homing_ready():
            return
        self.status("Centering before calibration…")
        self._center()
        self.drv.set_vibration(s, rpm, accel)
        self.drv.start()
        capped = False
        try:
            pts = []
            for it in range(4):
                self.status(f"Calibrating {it + 1}/4: {rpm:.0f} rpm, measuring (5 s)…")
                f_act = self.drv.measure_freq(5.0)
                if f_act is None:
                    raise IOError("Frequency measurement failed: no segment cycling")
                pts.append((rpm, f_act))
                if abs(f_act - f_t) <= max(0.03, 0.02 * f_t):
                    break
                # 用本工况实测开销 C 反解新转速(运行中写入即时生效)
                C = 1.0 / (2.0 * f_act) - (LIMITS.half_period_s(s, rpm) - LIMITS.c_overhead_s)
                rpm_new = LIMITS.rpm_for(s, f_t, C)
                if rpm_new is None:
                    rpm_new = LIMITS.rpm_cap
                    capped = True
                if abs(rpm_new - rpm) < 1:
                    break
                rpm = rpm_new
                self.drv.set_vibration(s, rpm, accel)
                time.sleep(1.0)     # 新转速生效1~2个周期后再测
                if capped:
                    f_act = self.drv.measure_freq(5.0)
                    if f_act:
                        pts.append((rpm, f_act))
                    break
            rpm_fin, f_fin = pts[-1]
        finally:
            self.drv.stop()
        self.calib[calib_key(s, f_t)] = {
            "rpm": round(rpm_fin, 1), "f_act": round(f_fin, 3),
            "accel": accel, "date": time.strftime("%Y-%m-%d")}
        save_calib(self.calib)
        msg = (f"Calibrated: S={s:g} mm  f={f_t:g} Hz → {rpm_fin:.0f} rpm, "
               f"measured {f_fin:.2f} Hz")
        if abs(f_fin - f_t) > max(0.05, 0.03 * f_t):
            msg += f"  ⚠ Target not reached; ~{f_fin:.2f} Hz is the current limit"
        # 标定结果立即写入参数,可直接开始
        self.drv.set_vibration(s, rpm_fin, accel)
        self.lbl_calc.config(text=msg)
        self.btn_start.config(state="normal")
        self.status(msg)

    def _ensure_homing_ready(self) -> bool:
        """按界面设置写入回零配置;P9-21 首次启用时提示断电重启。返回是否可用。"""
        first = self.drv.setup_homing(float(self.var_homeoff.get()),
                                      self.var_homedir.get() == "Reverse")
        if first:
            messagebox.showinfo(
                "Power Cycle Required",
                "Homing (P9-21) was enabled for the first time and only takes "
                "effect after a drive power cycle.\n"
                "Power off the drive, power it on, reconnect, then retry centering.")
            self.status("Homing configured; awaiting drive power cycle")
            return False
        return True

    def _center(self):
        """回中:本会话已知中点编码器值→无接触单段移动(静音,~1秒);
        否则触停回零(每次上电后首回中会有触底接触声)并记录中点。"""
        if self.mid_enc is not None:
            try:
                delta_mm = (self.mid_enc - self.drv.enc_pos()) / 13107.2
                if abs(delta_mm) > 30:
                    raise IOError("encoder position out of range")
                self.status(f"Contactless centering: moving {delta_mm:+.2f} mm…")
                self.drv.move_relative(round(delta_mm * 1000))
                self.status("✓ Centered (contactless)")
                return
            except Exception as e:
                self.status(f"Contactless centering failed ({e}); "
                            f"falling back to touch-probe homing…")
        self.drv.home_center(status_cb=self.status)
        try:
            self.mid_enc = self.drv.enc_pos()   # 记录中点,本会话后续回中免触底
        except Exception:
            self.mid_enc = None

    def _rewrite_vibration(self):
        """回中(可能占用段1参数)后,按界面值恢复激振段参数。"""
        s, f, rpm, _pulses, _src = self.calc()
        self.drv.set_vibration(s, rpm, int(self.var_accel.get()))

    @guard
    def on_center(self):
        if self.running:
            raise RuntimeError("Stop vibration before centering")
        if not self._ensure_homing_ready():
            return
        self.status("Centering…")
        self._center()
        if str(self.btn_start["state"]) == "normal":
            self._rewrite_vibration()   # 此前写过参数,恢复之

    @guard
    def on_clear_alarm(self):
        alm = self.drv.clear_alarm()
        self.status("✓ Alarm cleared" if alm == 0
                    else f"Alarm E-{alm:03d} persists (may require a power cycle)")

    @guard
    def on_start(self):
        if self.running:
            return
        alm = self.drv.alarm()
        if alm:
            alm = self.drv.clear_alarm()
            if alm:
                raise RuntimeError(f"Alarm E-{alm:03d} cannot be auto-cleared; "
                                   f"investigate before retrying (E-161 = overload/stall)")
            self.status("Alarm auto-cleared")
        if self.var_autocenter.get():
            if not self._ensure_homing_ready():
                return
            self.status("Auto-centering before start…")
            self._center()
            self._rewrite_vibration()   # 无接触回中会占用段1参数,必须恢复
        self.drv.start()
        self.running = True
        self.btn_stop.config(state="normal")
        self.btn_start.config(state="disabled")
        self.status("● Running… (STOP or closing the window disables the servo)")

    def on_stop(self):
        try:
            self.drv.stop()
        except Exception as e:
            messagebox.showerror("Stop Command Failed",
                                 f"{e}\nUse the physical power switch immediately!")
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status("Stopped (servo off)")

    def on_close(self):
        try:
            if self.drv.connected:
                self.drv.stop()
                self.drv.close()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
