#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仿真 → DS5L1 真机联动入口:把优化得到的工作参数在电动缸上执行。

仿真侧输出 (frequency_hz, 位移幅值/行程, duration_s) → 翻译为机器参数
(行程mm, 段转速rpm, 周期数) 并做可行性/完整性因子把关 → 连接驱动器、
回中、起振、在线频率标定、报警轮询、按时长停机。标定结果存入
config/ds5l1_freq_calib.json,下次同 (S,f) 工况直接复用。

用法示例
--------
仿真给位移幅值(半峰峰,米;来自 basin.steady_amplitude 或 FRF |H|·F):
    python scripts/run_harvest_on_rig.py --port COM8 \\
        --freq 2.5 --amplitude-m 0.003 --duration 12

直接给行程(峰峰,毫米):
    python scripts/run_harvest_on_rig.py --port COM8 \\
        --freq 2 --stroke-mm 5 --duration 10

从优化器导出的 JSON(键:frequency_hz, duration_s, 以及
displacement_amplitude_m 或 clamp_peak_to_peak_mm;可选
excitation_label, integrity_factor):
    python scripts/run_harvest_on_rig.py --port COM8 --params-json plan.json

只做参数翻译与可行性检查、不连硬件:
    python scripts/run_harvest_on_rig.py --dry-run --freq 2 --stroke-mm 5 --duration 10

安全:首次启用回零(P9-21)需驱动器断电重启一次;现场保留物理电源开关作急停。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchard_fem.actuator import (  # noqa: E402
    plan_harvest_execution,
    run_harvest_plan_on_rig,
    stroke_from_amplitude_m,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Execute simulation-derived harvest parameters on the DS5L1 rig.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("working parameters (flags or --params-json)")
    src.add_argument("--params-json", type=Path,
                     help="JSON file with frequency_hz, duration_s and one of "
                          "displacement_amplitude_m / clamp_peak_to_peak_mm")
    src.add_argument("--freq", type=float, help="excitation frequency [Hz]")
    src.add_argument("--duration", type=float, help="working duration [s]")
    src.add_argument("--stroke-mm", type=float,
                     help="clamp peak-to-peak stroke S [mm]")
    src.add_argument("--amplitude-m", type=float,
                     help="displacement amplitude (half peak-to-peak) [m]; S = 2A")
    src.add_argument("--label", default="", help="excitation position label")
    src.add_argument("--integrity-factor", type=float, default=None,
                     help="basin integrity factor of this working point")
    src.add_argument("--min-integrity-factor", type=float, default=0.0,
                     help="reject the plan when IF falls below this")
    src.add_argument("--accel-ms", type=int, default=10, help="segment accel/decel [ms]")

    rig = p.add_argument_group("rig connection")
    rig.add_argument("--port", help="serial port (e.g. COM8, /dev/ttyUSB0)")
    rig.add_argument("--baud", type=int, default=19200)
    rig.add_argument("--parity", default="E", choices=["E", "O", "N"])
    rig.add_argument("--stopbits", type=int, default=1, choices=[1, 2])
    rig.add_argument("--no-home", action="store_true",
                     help="skip centring (only if the rod is already mid-stroke)")
    rig.add_argument("--home-offset-mm", type=float, default=25.8,
                     help="hard-stop → mid-stroke offset")
    rig.add_argument("--home-reverse", action="store_true",
                     help="probe the reverse hard stop when homing")
    rig.add_argument("--calibrate", action="store_true",
                     help="enable online frequency calibration (adds 5–23 s of "
                          "vibration; off by default so the run lasts exactly the "
                          "computed duration)")
    rig.add_argument("--dry-run", action="store_true",
                     help="translate + feasibility-check only; no hardware")
    return p


def resolve_parameters(args: argparse.Namespace) -> dict:
    """Merge --params-json and direct flags (flags win) into planner kwargs."""
    data: dict = {}
    if args.params_json:
        data = json.loads(args.params_json.read_text(encoding="utf-8"))

    freq = args.freq if args.freq is not None else data.get("frequency_hz")
    duration = args.duration if args.duration is not None else data.get("duration_s")
    stroke = args.stroke_mm if args.stroke_mm is not None else data.get("clamp_peak_to_peak_mm")
    amp_m = args.amplitude_m if args.amplitude_m is not None \
        else data.get("displacement_amplitude_m")
    if stroke is None and amp_m is not None:
        stroke = stroke_from_amplitude_m(float(amp_m))
    integrity = args.integrity_factor if args.integrity_factor is not None \
        else data.get("integrity_factor")

    missing = [n for n, v in
               (("frequency", freq), ("duration", duration), ("stroke/amplitude", stroke))
               if v is None]
    if missing:
        raise SystemExit(f"Missing working parameters: {', '.join(missing)} "
                         f"(use flags or --params-json).")
    return dict(
        frequency_hz=float(freq),
        clamp_peak_to_peak_mm=float(stroke),
        duration_s=float(duration),
        accel_ms=args.accel_ms,
        excitation_label=args.label or data.get("excitation_label", ""),
        integrity_factor=integrity if integrity is None else float(integrity),
        min_integrity_factor=args.min_integrity_factor,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = plan_harvest_execution(**resolve_parameters(args))
    print(plan.summary())
    if not plan.feasible:
        return 2
    if args.dry_run:
        return 0
    if not args.port:
        raise SystemExit("--port is required to execute (or use --dry-run).")

    outcome = run_harvest_plan_on_rig(
        plan,
        port=args.port,
        baud=args.baud,
        parity=args.parity,
        stopbits=args.stopbits,
        home=not args.no_home,
        home_offset_mm=args.home_offset_mm,
        home_reverse=args.home_reverse,
        calibrate=args.calibrate,
        status_cb=print,
    )
    print(f"Outcome: {outcome}")
    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
