# Scripts

Command-line tools for the orchard-FEM harvest pipeline. Run from the repo root
in the project environment (`orchard-fenicsx`):

```bash
python scripts/<name>.py --help
```

## Harvest pipeline & rig
| Script | Purpose |
|---|---|
| `verify_pareto_end_to_end.py` | **Main driver.** End-to-end harvest recommendation + multi-clamp schedule figures for the 5 sample trees. `--ideal-actuator` runs the unconstrained "what actuator does the job need" study. |
| `run_harvest_on_rig.py` | Execute the recommended working parameters on the physical DS5L1 reciprocating rig. |
| `measure_actuator_envelope.py` | Empirically map the DS5L1 (frequency, amplitude) envelope by stepping rpm and measuring the achieved reciprocation frequency. |
| `ds5l1_comms_check.py` | Read-only WSL ↔ DS5L1 serial connectivity check (never moves the motor). |

## Calibration & measured-data validation
| Script | Purpose |
|---|---|
| `calibrate_from_measured_frf.py` | Inverse-fit Rayleigh damping + nonlinear (k₃, c₂) from a measured FRF. |
| `process_hammer_test.py` | Process hammer-impact test data into FRFs comparable to FE output. |
| `process_fixed_frequency.py` | Process variable-frequency (shaker) excitation data into per-frequency steady-state response. |
| `compute_validation_frf.py` | Single-branch, single-point FRF for hammer-test validation. |
| `compare_measured_vs_sim.py` | Overlay a measured FRF against the linear / nonlinear simulation FRFs. |

## Accuracy & meshing diagnostics
| Script | Purpose |
|---|---|
| `convergence_study.py` | Mesh (h) and time-step (Δt) convergence study. |
| `remesh_adaptive.py` | Per-branch adaptive re-meshing to a target element count. |
| `clamp_energy_reach.py` | Map how far each clamp's excitation energy actually reaches. |

## Package entry points (thin CLI wrappers)
`check_python_env.py`, `plot_frequency_response.py`, `visualize_analysis.py`,
`run_python_demo_suite.py` — launchers that delegate to the corresponding
`orchard_fem` package modules.

## `paper/`
One-off scripts that reproduce specific figures / tables / model validations for
the Biosystems Engineering 2026 paper (`render_*`, `draw_*`, Sobol sensitivity,
sensor-location scan, Duffing / quadratic-damping validation). Kept for paper
revisions; not part of the day-to-day pipeline.
