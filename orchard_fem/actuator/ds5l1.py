# -*- coding: utf-8 -*-
"""Bench-verified Xinje DS5L1 servo driver (Modbus RTU) + simulation linkage.

硬件:信捷 DS5L1S-20P4-PTA + MS6H-60CM30B3(17位) + 滚珠丝杠电动缸(导程10mm,直连)
      缸行程:两端硬限位间实测 51.67mm(2026-06-11 限矩150%双向触底+绝对编码器测得)
连接:PC --USB转串口(RS232)-- 驱动器 RS232 口,Modbus RTU
原理:内部位置模式(P0-01=5) + 换步模式0 + /CHGSTP 参数强制常ON(P5-35=0x0010)
      → 两段相对定位(+S / -S)无限循环 = 往复振动
      ★ 实测:段循环需要使能后的/CHGSTP上升沿才启动,故开始时序为
        P5-35=0 → F1-05=1 → P5-35=0x0010(软件造边沿)
启停:软件使能 F1-05(0x2105) 写1=开始(RUN) / 写0=立即停止(bb)
回中:触停式回原点(P9-21=1+模式6/7,手册5.3.1.9):低速顶到机械限位,
      按转矩+转速+持续时间判定到底,再自动按偏移量走到行程中点。
      ★ P9-21 首次置1后必须给驱动器断电重启一次才生效。
振幅:段脉冲数 = 行程S(mm) × 1000   (每圈10000脉冲 ÷ 导程10mm)
频率:段转速决定,模型与标定见 :mod:`orchard_fem.actuator.harvest_bridge`
      (``DS5L1Limits``,半周期 = 6S/rpm + C)。
串口:RS232 出厂实际 19200-8-E-1(真机读回 P7-11=0x2206;手册按位拆解表有误)

This module is the *driver + rig* layer: raw Modbus framing, the verified
``DS5L1`` driver class, the frequency-calibration table store, and
:func:`run_harvest_plan_on_rig` — the linkage that executes a
:class:`~orchard_fem.actuator.harvest_bridge.HarvestPlan` (translated from the
simulation's working parameters) on the physical rig.  Parameter translation
and feasibility live in :mod:`orchard_fem.actuator.harvest_bridge`; the Tk
front-end lives in :mod:`orchard_fem.actuator.vibration_gui`.

``pyserial`` is imported lazily in :meth:`DS5L1.connect`, so importing this
module (and the package) needs no hardware dependencies.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

from orchard_fem.actuator.harvest_bridge import (
    DS5L1Limits,
    HarvestPlan,
    HarvestSchedule,
    execute_harvest_plan,
    execute_harvest_schedule,
)

# ---------------- 常量(来自 DS5L1 手册附录4 Modbus 地址表) ----------------
STATION = 1                 # Modbus 站号(P7-10 / P7-00,默认 1)
PULSES_PER_MM = 1000        # 每圈10000脉冲 / 导程10mm(初始化时会读回 P0-11/12 核对)
RATED_RPM = 3000

ADDR = {
    "P0_01": 0x0001,   # 控制方式: 5=内部位置
    "P0_03": 0x0003,   # 使能模式: 2=软件使能
    "P0_04": 0x0004,   # 刚性等级(写入即时映射P1组增益;本机实测最优=19)
    "P0_11": 0x000B,   # 每圈脉冲数 低位
    "P0_12": 0x000C,   # 每圈脉冲数 高位
    "P0_13": 0x000D,   # 电子齿轮分子
    "P0_14": 0x000E,   # 电子齿轮分母
    "P4_03": 0x0403,   # 内部位置模式设置 n.[无][等待][换步][定位]
    "P4_04": 0x0404,   # 有效段数
    "P4_08": 0x0408,   # 起始段号
    # 段1
    "P4_10": 0x040A, "P4_11": 0x040B, "P4_12": 0x040C,
    "P4_13": 0x040D, "P4_14": 0x040E, "P4_16": 0x0410,
    # 段2
    "P4_17": 0x0411, "P4_18": 0x0412, "P4_19": 0x0413,
    "P4_20": 0x0414, "P4_21": 0x0415, "P4_23": 0x0417,
    "P5_35": 0x0523,   # /CHGSTP 端子分配; 0x0010=参数强制常ON(零接线)
    "F1_05": 0x2105,   # 软件使能: 1=使能RUN, 0=取消使能bb
    # ---- 触停式回原点(新回原点功能,手册5.3.1.9;模式6/7无需外接开关) ----
    "P9_11": 0x090B,   # n.[减速方式][回零模式][触发方式][Z相个数]
    "P9_12": 0x090C,   # 回原点高速 rpm(执行机械偏移量用)
    "P9_13": 0x090D,   # 回原点低速 rpm(触停寻底用)
    "P9_14": 0x090E,   # 回零加减速时间 ms(0→1000rpm)
    "P9_15": 0x090F,   # 回零超时 单位10ms(0=不限)
    "P9_16": 0x0910,   # 触停判定:转速阈值 rpm
    "P9_17": 0x0911,   # 触停判定:转矩阈值 %(过程中限矩为其1.1倍)
    "P9_18": 0x0912,   # 触停判定:持续时间 ms
    "P9_19": 0x0913,   # 机械偏移量低位(=回零后的绝对位置)
    "P9_20": 0x0914,   # 机械偏移量高位 ×10000
    "P9_21": 0x0915,   # 1=启用新回原点功能(改后需断电重启生效!)
    "P5_28": 0x051C,   # SI端子启动回原点;0→0x0010 软件造上升沿触发
    "F0_00": 0x2000,   # 写1=清除可清除类报警(如E-161过载)
    "U0_14": 0x100E,   # 位置反馈低位(带符号)
    "U0_15": 0x100F,   # 位置反馈高位 ×10000(带符号)
    "U0_57": 0x1039,   # 绝对编码器位置反馈 低16位(U0-57/58共32位,131072/圈)
    "U0_81": 0x1051,   # 内部位置模式当前段号(1/2循环;0=未执行段)
    "U1_00": 0x1100,   # 当前报警码(0=无)
    "P5_00": 0x0500,   # 定位完成(COIN)窗口,指令脉冲。50=0.05mm,
                       # 出厂11时定位整定极慢(实测拖累频率数倍)
}


def crc16(data: bytes) -> bytes:
    """Modbus RTU CRC16(低字节在前)。"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes([crc & 0xFF, crc >> 8])


def u16(v: int) -> int:
    """有符号 → 16位补码"""
    return int(v) & 0xFFFF


def s16(v: int) -> int:
    """16位补码 → 有符号"""
    return v - 0x10000 if v >= 0x8000 else v


def split_pulses(total: int):
    """总脉冲 = 高位×10000 + 低位(十进制拆分,向零截断保号)。返回 (低位, 高位)"""
    high = int(total / 10000)
    low = total - high * 10000
    return low, high


# ---------------- 频率标定表(在线标定结果,(S,f)→rpm 复用) ----------------


def default_calib_path() -> Path:
    """Calibration-table location: ``$ORCHARD_DS5L1_CALIB`` or ``<repo>/config/``."""
    env = os.environ.get("ORCHARD_DS5L1_CALIB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config" / "ds5l1_freq_calib.json"


def calib_key(stroke_mm: float, freq_hz: float) -> str:
    return f"S={stroke_mm:g},f={freq_hz:g}"


def load_calib(path: Path | str | None = None) -> dict:
    try:
        with open(path or default_calib_path(), encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def save_calib(table: dict, path: Path | str | None = None) -> None:
    p = Path(path or default_calib_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(table, fp, ensure_ascii=False, indent=1)


# ---------------- 驱动器通讯层 ----------------
class DS5L1:
    """自带的最小 Modbus RTU 实现(功能码 03/06),严格 CRC + 结构校验。

    严格校验很重要:校验位/停止位不匹配时(如对 E-1 的驱动器用 N-2),多数帧
    看似正常、个别帧被确定性毁坏(数据字节也可能错)。任何 CRC/结构不符都按
    异常抛出并附收发帧十六进制,绝不静默接受可疑数据。
    """

    TIMEOUT = 0.5

    def __init__(self):
        self.ser = None
        self.lock = threading.Lock()

    def connect(self, port: str, baud: int = 19200, parity: str = "E",
                stopbits: int = 1) -> None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "The DS5L1 driver requires pyserial (`pip install pyserial`)."
            ) from exc
        try:
            self.ser = serial.Serial(
                port=port, baudrate=baud, bytesize=8,
                parity=parity, stopbits=stopbits, timeout=0.05)
        except Exception as e:
            self.ser = None
            raise ConnectionError(f"Cannot open port {port}: {e}") from e
        # 多帧验证链路(P0-01 + 每圈脉冲低位):参数不匹配时多数帧会被毁,
        # 多读几帧可避免"碰巧能读一帧"误判连接成功。
        try:
            self.read("P0_01")
            self.read("P0_11")
        except Exception as e:
            self.close()
            raise ConnectionError(
                f"Port opened but communication check failed. Verify station ID, "
                f"baud, parity, stop bits and wiring (drive RS232 factory default: "
                f"19200-8-E-1). Detail: {e}") from e

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    @property
    def connected(self) -> bool:
        return self.ser is not None

    RETRY_BACKOFF = (0.05, 0.15, 0.4)   # 递增退避,躲开与PWM同步的干扰窗口

    def _transact(self, body: bytes, attempts: int = 4) -> bytes:
        """发送 body+CRC,接收并校验应答,坏帧自动重发(读写均幂等)。
        须持有 lock。实测伺服使能后功率级斩波会干扰 RS232(丢字节/毁CRC);
        重发可恢复,校验不通过的读应答绝不采用。"""
        last = None
        for n in range(attempts):
            if n:
                time.sleep(self.RETRY_BACKOFF[min(n - 1, len(self.RETRY_BACKOFF) - 1)])
            try:
                return self._transact_once(body)
            except IOError as e:
                last = e
        raise IOError(f"failed after {attempts} retries: {last}") from last

    def _transact_once(self, body: bytes) -> bytes:
        func = body[1]
        exp_len = (5 + 2 * ((body[4] << 8) | body[5])) if func == 0x03 else 8
        self.ser.reset_input_buffer()
        self.ser.write(body + crc16(body))
        self.ser.flush()
        deadline = time.time() + self.TIMEOUT
        buf = b""
        while time.time() < deadline and len(buf) < exp_len:
            chunk = self.ser.read(exp_len - len(buf))
            if chunk:
                buf += chunk
                if len(buf) >= 2 and buf[1] == (func | 0x80):
                    exp_len = 5  # Modbus 异常应答帧
        if not buf:
            raise IOError(f"no response within {self.TIMEOUT:.1f}s")
        if len(buf) >= 3 and buf[1] == (func | 0x80):
            raise IOError(f"Modbus exception code {buf[2]} (frame {buf.hex(' ')})")
        if len(buf) < exp_len:
            raise IOError(f"incomplete response: {len(buf)}/{exp_len} bytes {buf.hex(' ')}")
        resp = buf[:-2]
        if resp[0] != STATION or resp[1] != func:
            raise IOError(f"response header mismatch: {buf.hex(' ')}")
        if func == 0x06 and resp[2:6] != body[2:6]:
            raise IOError(f"write echo mismatch: sent {body.hex(' ')} received {buf.hex(' ')}")
        if func == 0x03 and resp[2] != exp_len - 5:
            raise IOError(f"read byte-count mismatch: {buf.hex(' ')}")
        if crc16(resp) != buf[-2:]:
            # 写应答=逐字节回显请求,回显已完全匹配则数据正确性强于CRC
            # (驱动器若收到坏请求会自己丢弃,不会回显)。使能后PWM干扰
            # 常只毁应答CRC,此情况对写指令安全放行;读指令必须重试。
            if func != 0x06:
                raise IOError(f"read response CRC error (interference, or parity/"
                              f"stop-bit mismatch; drive default 19200-8-E-1): {buf.hex(' ')}")
        time.sleep(0.01)  # RTU 帧间静默 ≥10ms
        return resp

    def read(self, name: str, count: int = 1):
        addr = ADDR[name]
        body = bytes([STATION, 0x03, addr >> 8, addr & 0xFF,
                      count >> 8, count & 0xFF])
        try:
            with self.lock:
                resp = self._transact(body)
        except Exception as e:
            raise IOError(f"Read {name}@{addr:#06x} failed: {e}") from e
        regs = [(resp[3 + 2 * i] << 8) | resp[4 + 2 * i] for i in range(count)]
        return regs if count > 1 else regs[0]

    def write(self, name: str, value: int, verify: bool = True) -> None:
        addr = ADDR[name]
        v = u16(value)
        body = bytes([STATION, 0x06, addr >> 8, addr & 0xFF, v >> 8, v & 0xFF])
        try:
            with self.lock:
                self._transact(body)
        except Exception as e:
            raise IOError(f"Write {name}@{addr:#06x}={value}({v:#06x}) failed: {e}") from e
        if verify:
            rb = self.read(name)
            if rb != v:
                raise IOError(f"{name} verify mismatch: wrote {v:#06x}, read back {rb:#06x}")

    # ---- 业务操作 ----
    def init_mode(self, wait_mode: int) -> str:
        """初始化内部位置模式(必须在 bb 未使能状态下执行)。返回核对信息。"""
        self.write("F1_05", 0, verify=False)          # 确保未使能
        self.write("P0_01", 5)                        # 内部位置模式
        self.write("P0_03", 2)                        # 软件使能
        self.write("P0_13", 1)                        # 电子齿轮 1:1
        self.write("P0_14", 1)
        self.write("P4_03", (wait_mode & 1) << 8)     # 相对定位+换步0+等待模式
        self.write("P4_04", 2)                        # 有效段数 2
        self.write("P4_08", 1)                        # 起始段 1
        self.write("P5_35", 0x0010)                   # /CHGSTP 参数强制常ON(零接线)
        self.write("P5_00", 50)                       # 到位窗口0.05mm(出厂11太严,定位整定极慢)
        self.write("P0_04", 19)                       # 刚性等级19:本机实测最优(13→2.7Hz,
                                                      # 16→5.7,19→6.5,22→5.4@S=2mm/300rpm)
        # 核对每圈脉冲数(决定 1mm=?脉冲)
        lo, hi = self.read("P0_11"), self.read("P0_12")
        ppr = s16(hi) * 10000 + s16(lo)  # 若高低位约定不同,此处读数会异常,需人工核
        msg = f"P0-11/12 readback: low={lo} high={hi} → {ppr} pulses/rev"
        if ppr != 10000:
            msg += "  ⚠ Not 10000 — verify scaling before running!"
        return msg

    def set_vibration(self, stroke_mm: float, rpm: float, accel_ms: int) -> None:
        """写两段参数(可在运行中调用,即时生效)"""
        total = round(stroke_mm * PULSES_PER_MM)
        lo1, hi1 = split_pulses(total)
        lo2, hi2 = split_pulses(-total)
        v = int(round(rpm * 10))                      # 单位 0.1rpm
        for name, val in (("P4_10", lo1), ("P4_11", hi1), ("P4_12", v),
                          ("P4_13", accel_ms), ("P4_14", accel_ms), ("P4_16", 0),
                          ("P4_17", lo2), ("P4_18", hi2), ("P4_19", v),
                          ("P4_20", accel_ms), ("P4_21", accel_ms), ("P4_23", 0)):
            self.write(name, val)

    def start(self):
        # 换步模式0需要在使能之后出现/CHGSTP上升沿才会启动段循环
        # (实测:0x0010恒ON时伺服RUN但段号恒0不动)。故先OFF→使能→再ON。
        self.write("P5_35", 0x0000, verify=False)
        self.write("F1_05", 1, verify=False)
        time.sleep(0.05)
        self.write("P5_35", 0x0010, verify=False)

    def measure_freq(self, dur: float = 5.0):
        """运行中实测往复频率:统计 U0-81 段号 1↔2 跳变。须已 start()。"""
        t0 = time.time()
        last = None
        t_first = t_last = None
        trans = 0
        while time.time() - t0 < dur:
            try:
                s = self.read("U0_81")
            except Exception:
                continue
            if s in (1, 2):
                if last is not None and s != last:
                    trans += 1
                    t = time.time()
                    if t_first is None:
                        t_first = t
                    t_last = t
                last = s
            time.sleep(0.04)
        if trans >= 3 and t_last > t_first:
            return (trans - 1) / (2.0 * (t_last - t_first))
        return None

    # ---- 报警 / 位置 ----
    def alarm(self) -> int:
        return self.read("U1_00")

    def clear_alarm(self) -> int:
        """F0-00 写1清除可清除类报警(如E-161堵转过载)。返回清除后的报警码。"""
        self.write("F0_00", 1, verify=False)
        time.sleep(0.5)
        self.write("F0_00", 0, verify=False)
        return self.alarm()

    def pos_pulses(self) -> int:
        lo, hi = self.read("U0_14", count=2)
        return s16(hi) * 10000 + s16(lo)

    def enc_pos(self) -> int:
        """绝对编码器位置(32位,131072计数/圈)。跟踪真实运动,
        与指令计数不同:触停回零的寻底阶段也能看到位移。"""
        lo, hi = self.read("U0_57", count=2)
        v = (hi << 16) | lo
        return v - 0x100000000 if v >= 0x80000000 else v

    # ---- 触停式回原点 → 推杆回行程中点 ----
    def setup_homing(self, offset_mm: float, reverse: bool = True,
                     torque_pct: int = 130, timeout_s: int = 15) -> bool:
        """配置触停式回零(模式6/7,不需要外接开关;在bb态调用)。

        reverse=True 用模式7(反向触停),False 用模式6(正向触停)。
        offset_mm = 限位到行程中点的距离;触停后驱动器自动按该偏移走到中点。
        返回 True 表示本次首次把 P9-21 置 1,需驱动器断电重启后回零才可用。

        ★ 转矩阈值实测定标:本机竖直安装带负载,常规运动转矩即达±85%、
          破静摩擦~117%,阈值≤100%时电机推不动会误判触停/卡死报E-250。
          130%(过程限矩143%)既高于运动需求,又远低于楔死水平(300%+)。
        """
        first_time = (self.read("P9_21") != 1)
        if first_time:
            self.write("P9_21", 1)
        mode = 7 if reverse else 6
        self.write("P9_11", 0x0010 | (mode << 8))   # Z相=0,触发=SI信号,模式6/7
        self.write("P9_12", 200)                    # 偏移段速度 rpm(≈33mm/s)
        self.write("P9_13", 40)                     # 触停寻底速度 rpm(≈6.7mm/s)
        self.write("P9_14", 1000)
        self.write("P9_15", int(timeout_s * 100))   # 超时,单位10ms
        self.write("P9_16", 2)                      # 触停:转速低于2rpm
        self.write("P9_17", int(torque_pct))        # 触停:转矩达阈值%(限矩1.1倍)
        self.write("P9_18", 300)                    # 触停:保持300ms判定到底(减少顶压时间)
        pulses = round(offset_mm * PULSES_PER_MM)
        if not reverse:
            pulses = -pulses    # 偏移量=回零后的绝对位置,须指向行程内侧
        lo, hi = split_pulses(pulses)
        self.write("P9_19", lo)
        self.write("P9_20", hi)
        return first_time

    def home_center(self, status_cb=None, timeout_s: int = 25) -> None:
        """执行回中:触停机械限位 → 自动走偏移量到行程中点。结束于bb。

        运动监控用绝对编码器(enc_pos):指令计数器在寻底阶段不更新,
        会把长距离寻底误判成"推不动",编码器反映真实位移无此问题。
        """
        def report(txt):
            if status_cb:
                status_cb(txt)
        alm = self.alarm()
        if alm:
            raise IOError(f"Active alarm E-{alm:03d} — clear it before centering")
        self.write("F1_05", 0, verify=False)
        # 回中期间临时降刚性:高增益(19)顶硬限位会高频啸叫+急速发热E-161;
        # 回中是低速运动,刚性13足够,结束后恢复原值(激振仍用高刚性)。
        rigidity = None
        try:
            rigidity = self.read("P0_04")
        except Exception:
            pass
        if rigidity and rigidity > 13:
            self.write("P0_04", 13, verify=False)
            # 注:13是下限。刚性9虽更安静,但实测摩擦憋停会被误判触底
            # (假零点早5.3mm),触底瞬间的轻啸只在每次上电首回中出现一次。
        self.write("P5_28", 0x0000, verify=False)
        # 使能。此时 P5-35 恒ON无边沿,激振段不会误启动(实测特性)。
        self.write("F1_05", 1, verify=False)
        time.sleep(0.3)
        try:
            self.write("P5_28", 0x0010, verify=False)   # 上升沿触发回零
            t0 = time.time()
            e_start = None
            last = None
            moved = False
            stable = 0
            bad = 0
            while True:
                time.sleep(0.25)
                try:
                    alm = self.alarm()
                    e = self.enc_pos()
                    bad = 0
                except Exception as exc:
                    bad += 1    # 使能后功率级EMI可能连续毁帧,跳过本轮采样
                    if bad >= 6:
                        raise IOError(f"Repeated comm failures while centering: {exc}") from exc
                    continue
                if alm:
                    raise IOError(f"Alarm E-{alm:03d} during centering "
                                  f"(path blocked, or torque/timeout threshold unsuitable)")
                if e_start is None:
                    e_start = e
                report(f"Centering… moved {(e - e_start) / 13107.2:+.2f} mm")
                if last is not None:
                    if abs(e - last) > 600:      # >0.05mm 视为在动
                        moved = True
                        stable = 0
                    else:
                        stable += 1
                last = e
                if moved and stable >= 3:    # 有过运动且静止≥0.75s → 回零完成,尽快断使能
                    break
                if not moved and time.time() - t0 > 6:
                    raise IOError("No motion detected: torque threshold (P9-17) "
                                  "too low or mechanism jammed")
                if time.time() - t0 > timeout_s:
                    raise IOError("Centering timeout (check that P9-21 was "
                                  "activated by a drive power cycle)")
        finally:
            try:
                self.write("P5_28", 0x0000, verify=False)
            finally:
                self.write("F1_05", 0, verify=False)    # 回到bb
                if rigidity and rigidity > 13:
                    try:
                        self.write("P0_04", rigidity, verify=False)  # 恢复激振用刚性
                    except Exception:
                        pass
        report("✓ Centering complete: rod at mid-stroke (servo off)")

    ENC_PER_PULSE = 131072 / 10000.0   # 编码器计数/指令脉冲 = 13.1072

    def move_relative(self, delta_pulses: int, rpm: float = 200,
                      accel_ms: int = 30, timeout_s: int = 10) -> None:
        """单段相对移动(只执行一次,绝对编码器闭环验证)。

        用于无接触快速回中:不碰限位、无触底/定住啸叫。原理:临时切到
        换步模式2(上升沿启动、顺序执行一遍、不循环),天然一次性,
        不依赖软件脉冲时序。会占用段1参数(P4-10~14)和P4-03/04,
        结束时恢复;调用方负责之后重写激振段参数。结束于bb。
        """
        if abs(delta_pulses) < 100:    # <0.1mm 不动
            return
        alm = self.alarm()
        if alm:
            raise IOError(f"Active alarm E-{alm:03d} — clear it first")
        self.write("F1_05", 0, verify=False)
        p403 = self.read("P4_03")                     # 记录原换步配置(含等待模式位)
        self.write("P4_03", 0x0020)                   # 换步模式2:上升沿单次执行,不循环
        self.write("P4_04", 1)                        # 只用段1
        lo, hi = split_pulses(delta_pulses)
        self.write("P4_10", lo, verify=False)
        self.write("P4_11", hi, verify=False)
        self.write("P4_12", int(rpm * 10), verify=False)
        self.write("P4_13", accel_ms, verify=False)
        self.write("P4_14", accel_ms, verify=False)
        target = self.enc_pos() + round(delta_pulses * self.ENC_PER_PULSE)
        self.write("P5_35", 0x0000, verify=False)
        self.write("F1_05", 1, verify=False)
        time.sleep(0.3)
        try:
            self.write("P5_35", 0x0010, verify=False)   # 上升沿触发,模式2只执行一遍
            t0 = time.time()
            last = None
            stable = 0
            bad = 0
            while True:
                time.sleep(0.2)
                try:
                    e = self.enc_pos()
                    alm = self.alarm()
                    bad = 0
                except Exception as exc:
                    bad += 1
                    if bad >= 6:
                        raise IOError(f"Repeated comm failures during move: {exc}") from exc
                    continue
                if alm:
                    raise IOError(f"Alarm E-{alm:03d} during move")
                if last is not None and abs(e - last) <= 600:
                    stable += 1
                else:
                    stable = 0
                last = e
                if stable >= 3:
                    if abs(e - target) <= 2600:          # ±0.2mm
                        break
                    raise IOError(f"Move target missed by {(e - target) / 13107.2:+.2f} mm")
                if time.time() - t0 > timeout_s:
                    raise IOError("Relative move timeout")
        finally:
            try:
                self.write("F1_05", 0, verify=False)       # 先断使能
            finally:
                self.write("P5_35", 0x0010, verify=False)  # bb下恢复常ON约定(无边沿风险)
                self.write("P4_03", p403)                  # 恢复原换步配置(模式0循环)
                self.write("P4_04", 2)                     # 恢复两段

    def stop(self):
        self.write("F1_05", 0, verify=False)


# ---------------- 仿真系统联动 ----------------


def run_harvest_plan_on_rig(
    plan: HarvestPlan,
    *,
    port: str | None = None,
    driver: DS5L1 | None = None,
    baud: int = 19200,
    parity: str = "E",
    stopbits: int = 1,
    home: bool = True,
    home_offset_mm: float = 25.8,
    home_reverse: bool = False,
    calibrate: bool = False,
    limits: DS5L1Limits | None = None,
    calib_path: Path | str | None = None,
    status_cb: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Execute a simulation-derived :class:`HarvestPlan` on the physical rig.

    ``calibrate`` is **off by default**: the rig vibrates for exactly
    ``plan.duration_s``.  Online frequency calibration (``measure_freq``) would
    add 5–23 s of un-counted vibration, so it is opt-in — enable it only to
    populate the calibration table for a new ``(S, f)`` point.  The segment rpm
    is always seeded from the table when a matching entry exists.

    完整联动链:仿真优化器 → :func:`~orchard_fem.actuator.harvest_bridge.
    plan_harvest_execution` 翻译为 ``HarvestPlan`` → 本函数在真机上执行。

    Flow: connect (unless *driver* given) → pre-clear alarms → init internal
    position mode → touch-probe homing to mid-stroke → seed segment rpm from
    the calibration table when a matching ``(S, f)`` entry exists → run via
    :func:`~orchard_fem.actuator.harvest_bridge.execute_harvest_plan` (online
    frequency calibration + alarm polling) → persist the converged calibration
    point back to the table.

    Parameters
    ----------
    plan:
        Feasible plan from the harvest bridge (refused otherwise).
    port / baud / parity / stopbits:
        Serial settings used when no *driver* is supplied (RS232 factory
        default 19200-8-E-1).
    driver:
        Already-connected :class:`DS5L1`; the caller keeps ownership (not
        closed here).  Exactly one of *port* / *driver* must be given.
    home / home_offset_mm / home_reverse:
        Centre the rod to mid-stroke before vibrating (recommended: the
        ``0 → +S`` relative loop assumes a centred start).  ``RuntimeError`` is
        raised if homing (P9-21) was enabled for the first time — the drive
        must then be power-cycled and the run retried.
    calib_path:
        Frequency-calibration table; defaults to :func:`default_calib_path`.

    should_stop:
        Polled during the run; returning ``True`` stops early
        (outcome ``"user_stop"``).

    Returns
    -------
    str
        ``"completed"``, ``"alarm_stop"``, or ``"user_stop"``.
    """
    if (driver is None) == (port is None):
        raise ValueError("Supply exactly one of `port` or `driver`.")

    def status(msg: str) -> None:
        if status_cb is not None:
            status_cb(msg)

    own = driver is None
    drv = driver or DS5L1()
    if own:
        status(f"Connecting to {port} ({baud}-8-{parity}-{stopbits})…")
        drv.connect(port, baud, parity, stopbits)
    try:
        alm = drv.alarm()
        if alm:
            status(f"Active alarm E-{alm:03d}; attempting clear…")
            alm = drv.clear_alarm()
            if alm:
                raise IOError(f"Alarm E-{alm:03d} cannot be auto-cleared; "
                              f"investigate before running (E-161 = overload/stall).")

        # (S, f) 标定缓存命中则用实测转速起步,在线标定只需微调。
        calib = load_calib(calib_path)
        key = calib_key(plan.stroke_mm, plan.frequency_hz)
        cached = calib.get(key)
        if cached:
            plan = dataclasses.replace(plan, seed_rpm=float(cached["rpm"]))
            status(f"Calibration cache hit {key}: seeding {cached['rpm']:.0f} rpm "
                   f"(measured {cached['f_act']:.2f} Hz on {cached.get('date', '?')}).")

        home_cb: Callable[[], None] | None = None
        if home:
            if drv.setup_homing(home_offset_mm, home_reverse):
                raise RuntimeError(
                    "Homing (P9-21) was enabled for the first time and only takes "
                    "effect after a drive power cycle. Power-cycle the drive, then rerun.")

            def home_cb() -> None:
                drv.home_center(status_cb=status_cb)

        def remember_calibration(cal) -> None:
            if cal.converged and cal.measured_hz is not None:
                calib[key] = {"rpm": round(cal.rpm, 1),
                              "f_act": round(cal.measured_hz, 3),
                              "accel": plan.accel_ms,
                              "date": time.strftime("%Y-%m-%d")}
                save_calib(calib, calib_path)
                status(f"Calibration point saved: {key} → {cal.rpm:.0f} rpm.")

        if not calibrate:
            status(f"Online calibration off — running exactly {plan.duration_s:g} s "
                   f"at {plan.seed_rpm:.0f} rpm.")
        return execute_harvest_plan(
            plan, drv,
            home=home_cb,
            calibrate=calibrate,
            limits=limits,
            on_status=status_cb,
            on_calibrated=remember_calibration,
            should_stop=should_stop,
        )
    finally:
        if own:
            drv.close()


def run_harvest_schedule_on_rig(
    schedule: HarvestSchedule,
    *,
    port: str | None = None,
    driver: DS5L1 | None = None,
    baud: int = 19200,
    parity: str = "E",
    stopbits: int = 1,
    home: bool = True,
    home_offset_mm: float = 25.8,
    home_reverse: bool = False,
    calibrate: bool = False,
    limits: DS5L1Limits | None = None,
    calib_path: Path | str | None = None,
    status_cb: Callable[[str], None] | None = None,
    on_stage: Callable | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Execute a multi-stage :class:`HarvestSchedule` on the physical rig.

    ``calibrate`` is **off by default** (each stage runs exactly its own
    ``duration_s``); enable it only to populate the calibration table.

    The staged-adjustment-sequence counterpart of :func:`run_harvest_plan_on_rig`:
    connect → pre-clear alarms → init → home once → run every stage in order via
    :func:`~orchard_fem.actuator.harvest_bridge.execute_harvest_schedule`
    (drive started once, segments rewritten live between stages).  Each stage is
    seeded from its own ``(S, f)`` calibration-table entry when present, and its
    converged calibration point is saved back.

    Returns ``"completed"``, ``"alarm_stop"``, or ``"user_stop"``.
    """
    if (driver is None) == (port is None):
        raise ValueError("Supply exactly one of `port` or `driver`.")
    if not schedule.feasible:
        raise ValueError("Refusing an infeasible/empty schedule:\n" + schedule.summary())

    def status(msg: str) -> None:
        if status_cb is not None:
            status_cb(msg)

    own = driver is None
    drv = driver or DS5L1()
    if own:
        status(f"Connecting to {port} ({baud}-8-{parity}-{stopbits})…")
        drv.connect(port, baud, parity, stopbits)
    try:
        alm = drv.alarm()
        if alm:
            status(f"Active alarm E-{alm:03d}; attempting clear…")
            alm = drv.clear_alarm()
            if alm:
                raise IOError(f"Alarm E-{alm:03d} cannot be auto-cleared; "
                              f"investigate before running (E-161 = overload/stall).")

        # Seed each stage from its own (S, f) calibration-table entry.
        calib = load_calib(calib_path)
        seeded: list = []
        for stage in schedule.stages:
            key = calib_key(stage.plan.stroke_mm, stage.plan.frequency_hz)
            cached = calib.get(key)
            plan = (dataclasses.replace(stage.plan, seed_rpm=float(cached["rpm"]))
                    if cached else stage.plan)
            seeded.append(dataclasses.replace(stage, plan=plan))
        schedule = dataclasses.replace(schedule, stages=tuple(seeded))

        home_cb: Callable[[], None] | None = None
        if home:
            if drv.setup_homing(home_offset_mm, home_reverse):
                raise RuntimeError(
                    "Homing (P9-21) was enabled for the first time and only takes "
                    "effect after a drive power cycle. Power-cycle the drive, then rerun.")

            def home_cb() -> None:
                drv.home_center(status_cb=status_cb)

        # Track the active stage so calibration is saved under the right (S, f).
        active: dict = {"key": None, "accel": 10}

        def stage_cb(stage) -> None:
            active["key"] = calib_key(stage.plan.stroke_mm, stage.plan.frequency_hz)
            active["accel"] = stage.plan.accel_ms
            if on_stage is not None:
                on_stage(stage)

        def remember_calibration(cal) -> None:
            if cal.converged and cal.measured_hz is not None and active["key"]:
                calib[active["key"]] = {"rpm": round(cal.rpm, 1),
                                        "f_act": round(cal.measured_hz, 3),
                                        "accel": active["accel"],
                                        "date": time.strftime("%Y-%m-%d")}
                save_calib(calib, calib_path)
                status(f"Calibration point saved: {active['key']} → {cal.rpm:.0f} rpm.")

        if not calibrate:
            status(f"Online calibration off — stages run exactly their durations "
                   f"({schedule.total_duration_s:.1f} s total).")
        return execute_harvest_schedule(
            schedule, drv,
            home=home_cb,
            calibrate=calibrate,
            limits=limits,
            on_status=status_cb,
            on_stage=stage_cb,
            on_calibrated=remember_calibration,
            should_stop=should_stop,
        )
    finally:
        if own:
            drv.close()
