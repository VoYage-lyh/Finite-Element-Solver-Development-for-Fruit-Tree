# Scripts

Command-line tools for the orchard-FEM harvest pipeline. Run from the repo root
in the project environment (`orchard-fenicsx`):

```bash
python scripts/<path>.py --help
```

Scripts are grouped by purpose. Single-analysis commands (run / modal /
visualize / plot-frequency-response / doctor / verify / full-validate) live in
the package CLI instead — `python -m orchard_fem --help`.

## Top level

| Script | Purpose |
|---|---|
| `generate_all_figures.py` | **Main driver.** Runs the harvest recommendation + multi-clamp schedule pipeline for the five tracked example trees and refreshes the standalone verification figures under `workspace/outputs/`. No flag = render from `workspace/cache/` (fast); `--force` = recompute FE results; `--only-figures` = strict cache render. |
| `run_harvest_console.sh` | Launch Harvest Console with FEniCSx while reusing ML packages from `orchard-ml`. |

## `rig/` — physical DS5L1 electric cylinder

| Script | Purpose |
|---|---|
| `run_harvest_on_rig.py` | Execute recommended working parameters on the DS5L1 reciprocating rig. |
| `measure_actuator_envelope.py` | Empirically map the DS5L1 (frequency, amplitude) envelope by stepping rpm. |
| `ds5l1_comms_check.py` | Read-only WSL ↔ DS5L1 serial connectivity check (never moves the motor). |

## `validation/` — simulation vs. measured experiments

| Script | Purpose |
|---|---|
| `calibrate_from_measured_frf.py` | Inverse-fit Rayleigh damping + nonlinear (k₃, c₂) from a measured FRF. |
| `process_hammer_test.py` | Process hammer-impact test data into FRFs comparable to FE output. |
| `process_fixed_frequency.py` | Process variable-frequency (shaker) excitation data into per-frequency steady-state response. |
| `compute_validation_frf.py` | Single-branch, single-point FRF for hammer-test validation. |
| `compare_measured_vs_sim.py` | Overlay a measured FRF against the linear / nonlinear simulation FRFs (reads `generate_all_figures.py`'s cache). |

## `studies/` — numerical accuracy & meshing

| Script | Purpose |
|---|---|
| `convergence_study.py` | Mesh (h) and time-step (Δt) convergence study. |
| `remesh_adaptive.py` | Per-branch adaptive re-meshing to a target element size. |
| `clamp_energy_reach.py` | Map how far each clamp's excitation energy actually reaches. |

## `paper/`

One-off scripts that reproduce specific figures / tables / model validations for
the Biosystems Engineering 2026 paper (`render_*`, `draw_*`, Sobol sensitivity,
sensor-location scan, Duffing / quadratic-damping validation). Kept for paper
revisions; not part of the day-to-day pipeline.
