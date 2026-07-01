#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSL ↔ DS5L1 串口连通性自检(只读,绝不让电机运动)。

用途:在接线/挂载 USB 串口后,先确认 PC 能跟驱动器正常通讯,再去跑运动。
做的事:自动找串口 → 自动探测校验/停止位(出厂 RS232 应为 19200-8-E-1,
        但交接文档常误填 N-2,故自动探测) → 读几个寄存器打印 → 关闭。
只发功能码 03(读保持寄存器),不写任何参数、不使能、不发段号。

前提(WSL):
  1) Windows 侧用 usbipd-win 把 USB 串口 attach 进 WSL(见仓库 README 或下方说明);
  2) 当前用户在 dialout 组: sudo usermod -aG dialout $USER 后重开终端;
  3) 用带 pyserial 的解释器运行,例如:
     /home/lyh/miniforge3/envs/orchard-fenicsx/bin/python scripts/ds5l1_comms_check.py

用法:
  python scripts/ds5l1_comms_check.py                 # 自动找口+自动探测串口参数
  python scripts/ds5l1_comms_check.py --port /dev/ttyUSB0
  python scripts/ds5l1_comms_check.py --port /dev/ttyUSB0 --parity E --stopbits 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchard_fem.actuator.ds5l1 import DS5L1, s16  # noqa: E402

# (校验, 停止位) 探测顺序:E-1 是真机实测值,放最前;N-2 是手册按位表(常被误用)
_PROBE = [("E", 1), ("N", 2), ("E", 2), ("O", 1), ("N", 1)]


def list_serial_ports() -> list[str]:
    """列出候选串口(优先 USB 转串口)。"""
    try:
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
    except Exception:
        ports = []
    # 兜底:直接扫 /dev
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        ports += [str(p) for p in Path("/dev").glob(pat.split("/")[-1])]
    # 去重、USB 优先
    seen, ordered = set(), []
    for p in sorted(ports, key=lambda x: ("USB" not in x and "ACM" not in x, x)):
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def try_connect(port: str, baud: int, parity: str, stopbits: int) -> DS5L1 | None:
    """尝试用给定参数连接;connect() 内部已读两帧校验链路。成功返回驱动,失败 None。"""
    drv = DS5L1()
    try:
        drv.connect(port, baud=baud, parity=parity, stopbits=stopbits)
        return drv
    except Exception as exc:
        # 区分"打不开口"(权限/不存在)和"打开了但通讯不对"(参数/接线)
        msg = str(exc)
        if "Permission denied" in msg:
            raise SystemExit(
                f"打开 {port} 被拒:把用户加入 dialout 组再重开终端\n"
                f"  sudo usermod -aG dialout $USER")
        if "could not open port" in msg or "No such file" in msg or "FileNotFound" in msg:
            return None  # 口不存在,交给上层换口
        return None      # 参数不匹配,交给上层换参数


def read_status(drv: DS5L1) -> None:
    """只读寄存器,打印驱动器状态(全程不写、不动)。"""
    ctrl = drv.read("P0_01")
    enable_mode = drv.read("P0_03")
    lo, hi = drv.read("P0_11"), drv.read("P0_12")
    ppr = s16(hi) * 10000 + s16(lo)
    alarm = drv.alarm()
    pos = drv.pos_pulses()
    seg = drv.read("U0_81")
    print("─" * 48)
    print(f"  控制方式 P0-01      = {ctrl}   ({'内部位置' if ctrl == 5 else '非内部位置模式'})")
    print(f"  使能模式 P0-03      = {enable_mode} ({'软件使能' if enable_mode == 2 else '其它'})")
    print(f"  每圈脉冲 P0-11/12   = {ppr}" + ("" if ppr == 10000 else "  ⚠ 非10000,核对缩放"))
    print(f"  当前报警 U1-00      = E-{alarm:03d}" if alarm else "  当前报警 U1-00      = 无")
    print(f"  位置反馈(指令脉冲)  = {pos}  ({pos / 1000.0:+.3f} mm)")
    print(f"  内部位置当前段 U0-81= {seg}")
    print("─" * 48)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DS5L1 串口连通性只读自检")
    ap.add_argument("--port", help="串口设备,如 /dev/ttyUSB0;不给则自动查找")
    ap.add_argument("--baud", type=int, default=19200, help="波特率(出厂 19200)")
    ap.add_argument("--parity", choices=["E", "O", "N"], help="指定校验则跳过自动探测")
    ap.add_argument("--stopbits", type=int, choices=[1, 2], help="指定停止位则跳过自动探测")
    args = ap.parse_args(argv)

    ports = [args.port] if args.port else list_serial_ports()
    if not ports:
        raise SystemExit(
            "未发现任何串口。WSL 里需先用 usbipd-win 把 USB 串口 attach 进来:\n"
            "  (Windows 管理员 PowerShell)\n"
            "    usbipd list\n"
            "    usbipd bind   --busid <BUSID>\n"
            "    usbipd attach --wsl --busid <BUSID>\n"
            "之后 WSL 里应出现 /dev/ttyUSB0")

    combos = ([(args.parity, args.stopbits)]
              if args.parity and args.stopbits else _PROBE)

    for port in ports:
        print(f"尝试串口 {port} …")
        for parity, stop in combos:
            drv = try_connect(port, args.baud, parity, stop)
            if drv is not None:
                print(f"✓ 连接成功:{port}  {args.baud}-8-{parity}-{stop}")
                try:
                    read_status(drv)
                finally:
                    drv.close()
                print("通讯正常(只读,未发任何运动指令)。")
                return 0
        print(f"  {port} 上所有参数组合均失败。")

    raise SystemExit(
        "连接失败。逐项排查:\n"
        "  • 适配器是否 attach 进 WSL(ls /dev/ttyUSB*)\n"
        "  • 用户是否在 dialout 组\n"
        "  • RS232 接线(驱动器是 RS232 口,非 RS485;TX/RX 是否对调)\n"
        "  • 站号是否为 1、驱动器是否上电")


if __name__ == "__main__":
    raise SystemExit(main())
