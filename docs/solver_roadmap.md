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
- Embedded 1D line-mesh specification builder for the future FEniCSx branch.
- Initial DOLFINx embedded-line mesh creation entry point for interval cells embedded in 3D.
- Initial mixed displacement-rotation field definitions and FunctionSpace construction on top of the embedded DOLFINx mesh surface.
- Initial experimental Timoshenko-style UFL beam forms driven by orchard cellwise section coefficients.
- FEniCSx cell coefficients now use direct layered-section rigidities (`EA`, `GA`, `GJ`, `EI`) instead of only area-averaged Young's modulus times total section moments.
- Embedded-line UFL forms now use the physical branch-tangent derivative instead of a global-axis derivative.
- Elastic residual forms and automatic Jacobian generation through `ufl.derivative`.
- PETSc operator assembly entry point for the default FEniCSx beam branch.

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
- The native fallback and FEniCSx branch both consume direct layered-section rigidities for stiffness. The native branch remains a simpler Euler-Bernoulli fallback, while the default FEniCSx branch carries the active Timoshenko/UFL workflow.

## 6. Solver Layer

Status: Mixed

Implemented:
- SLEPc modal solve with shift-and-invert.
- Linear PETSc frequency response for systems without localized nonlinear links.
- Newmark time integration with manual Newton-style inner iteration.
- Adaptive first-harmonic balance frequency-continuation sweep for localized nonlinear links in the native solver branch.
- Experimental SLEPc modal solve on the assembled FEniCSx embedded-beam operator branch.
- Experimental root-clamp Dirichlet boundary conditions for the FEniCSx embedded-beam modal branch.
- Experimental linear PETSc frequency-response solve on the assembled FEniCSx embedded-beam operator branch.
- Experimental adaptive first-harmonic balance frequency-continuation sweep on the FEniCSx embedded-beam operator branch.
- Experimental linear PETSc/Newmark time-history solve on the assembled FEniCSx embedded-beam operator branch.
- Experimental linear joint-constraint augmentation on the FEniCSx embedded-beam modal, linear frequency-response, and linear time-history branch.
- Experimental nonlinear joint-law links in the FEniCSx embedded-beam time-history branch.
- Experimental automatic nonlinear injection and cubic clamp links in the FEniCSx embedded-beam time-history branch.
- Experimental PETSc SNES nonlinear Newmark solve for localized nonlinear links in the FEniCSx embedded-beam time-history branch, with beam elastic residual and Jacobian assembled from UFL forms.
- Experimental fruit-attachment augmentation on the FEniCSx embedded-beam modal, linear frequency-response, and linear time-history branch.
- Experimental gravity-prestress load solve plus geometric-stiffness augmentation on the FEniCSx embedded-beam modal, linear frequency-response, and linear time-history branch.
- Main-workflow backend routing with `fenicsx` as the default solver backend and `native` available only through explicit opt-in.
- Shared staged FEniCSx system-assembly facade used by modal, frequency-response, and time-history entry points. Current stages are mesh, function space, cell data, coefficients, UFL forms, boundary conditions, base operators, branch joints, automatic nonlinear links, nonlinear clamps, fruit attachments, and gravity prestress.

Planned:
- Higher-order harmonic balance for strongly nonsinusoidal localized nonlinear responses.

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

## Next Push

Near-term implementation order:

1. Push the default `orchard_fem.fenicsx` branch from modal and frequency-response solves toward tighter benchmark-backed validation.
2. Extend the first-harmonic nonlinear frequency workflow toward higher-order harmonics only after the main FEniCSx branch is stable.
3. Move selected assembly paths closer to PETSc-native data structures once the FEniCSx branch can represent orchard topology cleanly.
