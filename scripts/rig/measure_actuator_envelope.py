#!/usr/bin/env python3
"""Empirically map the DS5L1 electric-cylinder (f, A) envelope.

For each stroke (peak-to-peak) the script steps the motor rpm up and uses the
servo's own segment-counter (:meth:`DS5L1.measure_freq`, U0-81 1↔2 transitions)
to measure the ACTUAL reciprocation frequency. From the measured f-vs-rpm curve
it fits the half-cycle reversal overhead ``C`` (half_period ≈ 6·S/rpm + C) and
reports the highest stable rpm before an alarm — i.e. the real ``rpm_cap`` and
``c_overhead_s`` behind :class:`~orchard_fem.actuator.harvest_bridge.DS5L1Limits`.

⚠ THIS MOVES THE PHYSICAL ROD. Run it yourself, watching the rig, hand on the
e-stop / power cutoff. Home leaves the rod mid-travel; ensure clearance for the
FULL stroke above centre. Start with the defaults (≤1500 rpm); only raise
``--max-rpm`` once you have seen how the rig behaves.

Examples
--------
    # preview the plan, NO motion:
    python scripts/measure_actuator_envelope.py --dry-run

    # real sweep, conservative (≤1500 rpm), default strokes:
    python scripts/measure_actuator_envelope.py --port /dev/ttyUSB0

    # push toward the 3000 rpm rating (only after a clean conservative run):
    python scripts/measure_actuator_envelope.py --max-rpm 3000 --i-understand
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _connect(drv, port: str, parity: str, stopbits: int):
    """Open *port*, auto-trying serial framings if the explicit one fails.

    DS5L1 RS232 ships 19200-8-N-2 but RS485 is 19200-8-E-1; try the common
    combos so the user doesn't have to know which bus the rig is wired on.
    """
    combos = [(parity, stopbits)] if parity != "auto" else [
        ("N", 2), ("E", 1), ("N", 1), ("E", 2)]
    last = None
    for par, stop in combos:
        try:
            drv.connect(port, 19200, par, stop)
            print(f"  connected: 19200-8-{par}-{stop}")
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  framing 19200-8-{par}-{stop} failed: {e}")
    raise SystemExit(f"[abort] could not connect on {port}: {last}")


def _predicted_freq(limits, stroke_mm: float, rpm: float) -> float | None:
    """Model's predicted reciprocation frequency at (stroke, rpm), or None."""
    half = stroke_mm / (rpm * limits._mm_per_rpm_s) + limits.c_overhead_s
    return 1.0 / (2.0 * half) if half > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--parity", default="auto", choices=["auto", "N", "E", "O"])
    ap.add_argument("--stopbits", type=int, default=2, choices=[1, 2])
    ap.add_argument("--strokes", default="5,10,15,20",
                    help="peak-to-peak strokes mm (amplitude A = stroke/2)")
    ap.add_argument("--rpm-start", type=float, default=150.0)
    ap.add_argument("--rpm-step", type=float, default=150.0)
    ap.add_argument("--max-rpm", type=float, default=1500.0,
                    help="rpm ceiling; >1500 needs --i-understand (servo rated 3000)")
    ap.add_argument("--measure-s", type=float, default=4.0)
    ap.add_argument("--accel", type=int, default=10, help="ramp time ms (0→rated)")
    ap.add_argument("--no-home", action="store_true", help="skip the centre-homing step")
    ap.add_argument("--i-understand", action="store_true",
                    help="acknowledge motion / allow --max-rpm > 1500")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no motion")
    ap.add_argument("--out", default="results/calibration/actuator_envelope.csv")
    args = ap.parse_args()

    from orchard_fem.actuator.ds5l1 import DS5L1
    from orchard_fem.actuator.harvest_bridge import DS5L1Limits

    limits = DS5L1Limits()
    strokes = [float(x) for x in args.strokes.replace("，", ",").split(",") if x.strip()]
    if any(s > limits.max_stroke_mm for s in strokes):
        raise SystemExit(f"[abort] stroke exceeds mechanical max {limits.max_stroke_mm} mm")
    if args.max_rpm > 1500.0 and not args.i_understand:
        raise SystemExit("[abort] --max-rpm > 1500 requires --i-understand")

    rpms = []
    r = args.rpm_start
    while r <= args.max_rpm + 1e-6:
        rpms.append(round(r, 1))
        r += args.rpm_step

    print("PLAN — measured actuator envelope")
    print(f"  port {args.port}   strokes {strokes} mm   rpm {rpms[0]}…{rpms[-1]} "
          f"step {args.rpm_step}   {args.measure_s}s/point")
    print(f"  model says rpm_cap={limits.rpm_cap}, c_overhead_s={limits.c_overhead_s}")
    print(f"  → {len(strokes) * len(rpms)} points, ~{len(strokes)*len(rpms)*(args.measure_s+1.5):.0f}s")
    if args.dry_run:
        print("[dry-run] no motion. Re-run without --dry-run to drive the rig.")
        return 0
    if not args.i_understand:
        print("\n⚠ This MOVES THE ROD. Ensure clearance above centre, hand on e-stop.")
        if input("  type 'go' to start: ").strip().lower() != "go":
            print("[abort] not confirmed.")
            return 1

    drv = DS5L1()
    rows: list[dict] = []

    def _safe_stop(*_a):
        try:
            drv.set_vibration(strokes[0], 0, args.accel)
        except Exception:  # noqa: BLE001
            pass
        try:
            drv.stop()
        except Exception:  # noqa: BLE001
            pass
        print("\n[stopped] servo disabled.")

    signal.signal(signal.SIGINT, lambda *_: (_safe_stop(), sys.exit(130)))

    try:
        print(f"\nConnecting {args.port}…")
        _connect(drv, args.port, args.parity, args.stopbits)
        drv.clear_alarm()
        drv.init_mode(0)                       # internal-position reciprocation, wait=0
        if not args.no_home:
            print("Homing to centre…")
            drv.home_center(status_cb=lambda m: print(f"   {m}"))
        print(f"\n{'stroke':>6} {'A':>5} {'rpm':>6} {'f_meas':>7} {'f_model':>7} {'alarm':>6}")
        for s in strokes:
            started = False
            f_max = 0.0
            for rpm in rpms:
                drv.set_vibration(s, rpm, args.accel)
                if not started:
                    drv.start()
                    started = True
                    time.sleep(0.3)
                f = drv.measure_freq(args.measure_s)
                al = drv.alarm()
                fp = _predicted_freq(limits, s, rpm)
                print(f"{s:6.1f} {s/2:5.1f} {rpm:6.0f} "
                      f"{(f if f else float('nan')):7.2f} {(fp if fp else float('nan')):7.2f} "
                      f"{al:6d}" + ("  <ALARM>" if al else ""))
                rows.append(dict(stroke_mm=s, amplitude_mm=s/2, rpm=rpm,
                                 f_meas_hz=(f or ""), f_model_hz=(round(fp, 3) if fp else ""),
                                 alarm=al))
                if f:
                    f_max = max(f_max, f)
                if al:
                    drv.set_vibration(s, 0, args.accel)
                    drv.stop()
                    drv.clear_alarm()
                    started = False
                    print(f"   alarm at {rpm:.0f} rpm → stop this stroke; f_max≈{f_max:.2f} Hz")
                    time.sleep(0.5)
                    break
            drv.set_vibration(s, 0, args.accel)
            drv.stop()
            started = False
            print(f"   stroke {s:.0f}mm: max measured f ≈ {f_max:.2f} Hz")
            time.sleep(0.8)
    finally:
        _safe_stop()
        try:
            drv.close()
        except Exception:  # noqa: BLE001
            pass

    # ---- save + summarise ----
    if rows:
        out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[saved] {out}")
        # crude per-stroke summary: highest rpm reached + max f
        print("\nSUMMARY (measured):")
        for s in strokes:
            sr = [r for r in rows if r["stroke_mm"] == s and r["f_meas_hz"] != ""]
            if sr:
                fmax = max(float(r["f_meas_hz"]) for r in sr)
                rmax = max(float(r["rpm"]) for r in sr if not r["alarm"])
                print(f"  stroke {s:5.1f}mm (A={s/2:.1f}): max f≈{fmax:.2f} Hz, "
                      f"highest clean rpm≈{rmax:.0f}")
        print("\nNext: paste this output back and I'll fit the real rpm_cap + "
              "c_overhead_s and update DS5L1Limits, then re-run the recommendation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
