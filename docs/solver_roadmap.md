# Solver Roadmap

This document tracks the current Orchard FEM implementation against the active solver roadmap.

Status labels:

- Implemented
- Partial / reduced implementation
- Planned
- Reserved

## 1. Input Layer

Status: Implemented

- JSON model loading is active in `orchard_fem.io.loaders`.
- Branches, materials, joints, fruits, clamps, excitation, analysis settings, and observations are all typed on load.

## 2. Domain Object Layer

Status: Implemented

- Topology, branch paths, section series, materials, fruits, clamps, excitation, and analysis objects are active in `orchard_fem.domain` and `orchard_fem.topology`.

## 3. Geometry And Section Processing

Status: Mixed

Implemented:
- `BranchPath` geometry, direction, and inclination angle.
- Tissue-region section integration through `SectionIntegrator`.

Partial / reduced:
- Assembly still evaluates section state by station interpolation and then averages adjacent node states per beam element.
- This is practical and stable, but it is still a reduced beam-property pipeline rather than a higher-order tissue-aware element integration path.

Planned:
- Explicit branch-angle-driven assembly features beyond the current geometry helper.

## 4. Discretization Layer

Status: Mixed

Implemented:
- Manual 3D Euler-Bernoulli beam element with closed-form 12x12 local operators.
- Global coordinate transforms and beam-element scatter logic.

Planned:
- FEniCSx / DOLFINx 1D mesh branch.
- Mixed displacement-rotation field formulation in UFL.
- Timoshenko weak-form implementation.
- Automatic tangent generation through `ufl.derivative`.

## 5. Assembly Layer

Status: Mixed

Implemented:
- Global `K`, `M`, `C`, gravity load, excitation DOF, and observation DOF assembly.
- Fruit point mass plus spring-damper attachment.
- Penalty-style clamp and branch-connection constraints.
- Localized nonlinear links:
  - clamp cubic links
  - explicit joint polynomial links
  - explicit joint gap links
  - automatic nonlinear injection by branch level
- Gravity prestress with geometric stiffness contribution.

Partial / reduced:
- The working system is assembled in Python matrix form and converted to PETSc at solve time, not assembled PETSc-native from the start.
- Parent-child continuity is still enforced by pairwise penalty links rather than a stronger local continuity treatment.
- Tissue stiffness is still represented through effective section properties rather than a direct per-tissue `sum(E_i I_i)` beam operator.

## 6. Solver Layer

Status: Mixed

Implemented:
- SLEPc modal solve with shift-and-invert.
- Linear PETSc frequency response for systems without localized nonlinear links.
- Newmark time integration with manual Newton-style inner iteration.
- Nonlinear frequency-response fallback via warm-started steady-state time-domain sweep when localized nonlinear links are active.

Planned:
- Describing-function or equivalent-linear nonlinear frequency iteration.
- PETSc SNES-driven nonlinear time solve.
- Full continuation / harmonic-balance nonlinear frequency workflow.

## 7. Output And Verification

Status: Implemented

- CSV output for modal summaries, frequency response, and time history.
- Geometry, response, spectrogram, and trajectory visualization.
- `verify`, `full-validate`, and demo-suite workflows.
- Analytical and engineering verification cases for beam, Duffing, hinged two-bar, and gravity-prestress behavior.

## Reserved Extension Surfaces

Status: Reserved

- `orchard_fem/model_reduction/`
- `orchard_pinn/`

These are intentionally kept for future reduction, surrogate, and inversion work, but they are not part of the active solver core today.
