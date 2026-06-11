"""Hardware actuation: simulation working parameters → DS5L1 vibration rig.

Three layers, top-down:

* :mod:`orchard_fem.actuator.harvest_bridge` — pure translation + execution
  logic: harvest-optimiser parameters (frequency, displacement amplitude,
  duration) → machine settings (stroke, rpm, cycles) with feasibility /
  integrity-factor gating.  No hardware dependencies.
* :mod:`orchard_fem.actuator.ds5l1` — the bench-verified DS5L1 driver
  (Modbus RTU, lazy ``pyserial``), frequency-calibration table, and
  :func:`~orchard_fem.actuator.ds5l1.run_harvest_plan_on_rig` — the one-call
  simulation→rig linkage.
* :mod:`orchard_fem.actuator.vibration_gui` — Tk bench console
  (``python -m orchard_fem.actuator.vibration_gui``); not imported here to
  keep ``tkinter`` optional.
"""
from orchard_fem.actuator.ds5l1 import (
    DS5L1,
    PULSES_PER_MM,
    calib_key,
    default_calib_path,
    load_calib,
    run_harvest_plan_on_rig,
    save_calib,
)
from orchard_fem.actuator.harvest_bridge import (
    CalibrationOutcome,
    DS5L1Limits,
    HarvestPlan,
    VibrationDriver,
    calibrate_frequency,
    execute_harvest_plan,
    plan_from_harvest_parameters,
    plan_harvest_execution,
    stroke_from_amplitude_m,
)

__all__ = [
    # translation + execution (hardware-free)
    "DS5L1Limits",
    "HarvestPlan",
    "VibrationDriver",
    "CalibrationOutcome",
    "stroke_from_amplitude_m",
    "plan_harvest_execution",
    "plan_from_harvest_parameters",
    "calibrate_frequency",
    "execute_harvest_plan",
    # verified driver + rig linkage
    "DS5L1",
    "PULSES_PER_MM",
    "run_harvest_plan_on_rig",
    "calib_key",
    "load_calib",
    "save_calib",
    "default_calib_path",
]
