# OrchardVibrationSolver FEniCSx + PINN Refactor Design

> Historical note
>
> This document records an earlier migration plan that kept C++ as a first-class project core.
> The active repository direction has since changed to a Python-first workflow centered on
> `orchard_fem`, `python -m orchard_fem verify`, and Python-side verification benchmarks.
> Read [python_first_architecture.md](python_first_architecture.md) first for the current plan.

## 1. Scope and Non-Negotiables

This document defines the Phase P0 refactor plan for evolving the current OrchardVibrationSolver repository into a dedicated orchard-vibration framework with:

- a FEniCSx / PETSc / SLEPc based auxiliary FEM backbone,
- a stable C++ orchard-specific domain and reference-solver layer,
- PINN / neural surrogate modules for inversion and acceleration,
- preserved input/output compatibility with the current repository.

This is not a plan to build a general-purpose FEM package. The orchard-specific modeling layer remains the project center of gravity.

The user has explicitly required that C++ remain a first-class part of the project. That constraint is adopted here as an architectural rule:

- C++ remains the stable domain kernel and reference implementation.
- Python/FEniCSx is introduced as a collaborative backend and research orchestration layer, not as a wholesale replacement of the C++ codebase.
- Existing C++ CLI, examples, CSV semantics, and verification meaning remain part of the supported project surface.

The following repository invariants are treated as fixed:

- `examples/*.json` schema remains backward compatible.
- CSV output column names and order remain backward compatible.
- existing verification cases keep the same physical meaning.
- orchard-specific names stay stable where possible:
  - `BranchPath`
  - `CrossSectionProfile`
  - `TissueRegion`
  - `FruitAttachment`
  - `ClampBoundaryCondition`
  - `HarmonicExcitation`

## 2. Current Repository Audit

### 2.1 Current top-level structure

The repository is currently organized as a single C++ library plus CLI and Python scripts:

- `include/orchard_solver/`
- `src/`
- `apps/`
- `examples/`
- `tests/`
- `scripts/`
- `docs/`
- `config/`

### 2.2 Current domain model status

The current code already contains a useful orchard-specific domain layer:

- topology:
  - `TreeTopology`
  - `BranchPath`
  - `ObservationPoint`
- branch and section modeling:
  - `BranchComponent`
  - `CrossSectionProfile`
  - `ParameterizedSectionProfile`
  - `ContourSectionProfile`
  - `MeasuredSectionSeries`
  - `SectionIntegrator`
- materials:
  - `MaterialLibrary`
  - `MaterialProperties`
  - linear / nonlinear / orthotropic-placeholder materials
- orchard-specific attachments and BCs:
  - `FruitAttachment`
  - `ClampBoundaryCondition`
  - `HarmonicExcitation`
- local nonlinear constructs:
  - `JointComponent`
  - `JointConstitutiveLaw`
  - `NonlinearLink`

This layer already captures most of the orchard-specific semantics that must survive the refactor.

Just as important: this domain layer already exists in C++, and that is an asset rather than a migration obstacle. The refactor should preserve C++ ownership of:

- stable data-model definitions,
- compatibility behavior,
- baseline solver behavior,
- verification references.

### 2.3 Current numerical pipeline

The current computational path is:

`ModelIO -> OrchardModel -> StructuralAssembler -> DynamicSystem -> FrequencyResponseAnalyzer / NewmarkIntegrator -> CSV`

Important observations:

- the current assembler is beam-oriented and section-aware;
- branch section interpolation already computes effective:
  - area,
  - bending inertias,
  - polar moment placeholder,
  - mass per unit length,
  - effective Young's modulus,
  - effective shear modulus,
  - damping ratio;
- current damping uses Rayleigh damping or a fallback damping-ratio-derived approximation;
- fruits and clamps are already represented as localized couplings;
- local nonlinear terms are already handled in time history through `NonlinearLink`.

### 2.4 Current limitations that matter for refactor

The current repository still relies on handwritten generic numerical infrastructure:

- `DenseMatrix`
- dense Gaussian elimination
- manual frequency-response solve
- manual Newmark nonlinear iteration loop
- manual matrix assembly and direct scattering

This is precisely the part that should be replaced by PETSc / SLEPc and, where appropriate, DOLFINx / UFL forms.

There are also domain-level limitations that should be kept visible during migration:

- joint behavior is still represented by penalty-like coupling plus simplified constitutive laws;
- torsion uses a placeholder `Ix + Iy` style approximation;
- frequency response is currently linearized;
- the repository does not yet have a true Python-side orchard FEM package;
- `SimpleJson` is still in use.

### 2.5 Current verification and regression assets

The repository already contains useful regression assets:

- unit-style smoke tests in `tests/orchard_tests.cpp`
- verification cases in `tests/verification/`
  - cantilever first mode
  - cantilever multi-mode convergence
  - simply supported beam static deflection
  - Duffing response trend
  - hinged two-bar benchmark
- example models:
  - `examples/demo_orchard.json`
  - `examples/demo_orchard_time_history.json`
- visualization scripts for geometry and response output

These assets are strong enough to serve as migration gates in P1.

## 3. Keep / Port / Add / Legacy Classification

### 3.1 Modules to keep as stable domain interfaces

These concepts should remain stable and largely backend-agnostic:

- `TreeTopology`
- `BranchPath`
- `ObservationPoint`
- `CrossSectionProfile`
- `ParameterizedSectionProfile`
- `ContourSectionProfile`
- `MeasuredSectionSeries`
- `SectionIntegrator`
- `MaterialProperties`
- `MaterialLibrary`
- `BranchComponent`
- `FruitAttachment`
- `ClampBoundaryCondition`
- `HarmonicExcitation`
- JSON schema semantics
- CSV schema semantics

These modules represent orchard problem definition, not numerical backend policy.

### 3.2 Modules to migrate away from handwritten solver infrastructure

These are the primary migration targets:

- `solver_core/LinearAlgebra.*`
- most of `solver_core/DynamicSystem.*` numerical internals
- most of `discretization/Assembler.*`
- beam operator assembly responsibilities currently living in:
  - `discretization/BeamElement.*`
  - `discretization/Assembler.*`

The target is:

- PETSc matrices/vectors for algebra,
- SLEPc eigensolvers for modal analysis,
- DOLFINx/UFL forms where beam weak-form assembly is practical,
- PETSc SNES/KSP where nonlinear and linear solves belong.

### 3.3 Modules to add as Python coordination and auxiliary solver layers

The following should be added in Python first, because FEniCSx/PETSc/SLEPc/PyTorch live there naturally:

- orchard-side FEM orchestration package:
  - `orchard_fem/`
- PINN / surrogate package:
  - `orchard_pinn/`
- benchmark and comparison scripts
- data-generation scripts for surrogate training and inversion

Reason:

- FEniCSx and SLEPc are most productive from Python for a research codebase;
- PyTorch-based inversion and surrogate workflows belong in Python;
- JSON/HDF5/CSV I/O interoperability is simpler there;
- Python is the natural place for orchestration, data generation, inversion, surrogate training, and benchmarking.

This does **not** imply that C++ domain objects are discarded. The intended relationship is:

- C++: stable domain kernel and baseline/reference backend
- Python: orchestration, alternative backend, ML workflows, and comparison tooling

### 3.4 Modules to keep as first-class C++ implementation

The current C++ implementation should be preserved as a first-class backend during migration:

- `src/discretization/BeamElement.cpp`
- `src/discretization/Assembler.cpp`
- `src/solver_core/DynamicSystem.cpp`
- `src/solver_core/LinearAlgebra.cpp`
- `apps/orchard_cli.cpp`
- current verification tests

This C++ path is not just temporary scaffolding. It serves four continuing roles:

- orchard-domain source of truth during the refactor,
- reference backend for result comparison,
- production-capable fallback when Python/FEniCSx environments are unavailable,
- regression guardrail for migration tolerances.

The repository should not delete or demote this backend during P1-P2.

## 4. Recommended Target Structure in This Repository

The recommended logical target structure is:

```text
orchard_fem/
  topology/
  cross_section/
  materials/
  elements/
  attachments/
  bc/
  solvers/
  io/

orchard_pinn/
  inverse/
  surrogate/
  data/
  utils/

tests/
  unit/
  verification/
  integration/
```

### 4.1 Practical migration rule

This restructure should not happen in one step.

For P1-P2:

- keep existing `include/`, `src/`, and `apps/` in place as stable C++ project roots;
- add `orchard_fem/` and `orchard_pinn/` at repository root;
- add Python-side tests under:
  - `tests/unit/`
  - `tests/integration/`
- keep current `tests/verification/` as the migration reference suite.

No rename to `legacy_cpp/` is planned in the current refactor window. That would create churn without technical value.

## 5. FEniCSx Port Strategy

## 5.1 Primary decision

The project should stop maintaining handwritten generic linear algebra and generic solve loops.

Backend responsibilities move to:

- PETSc:
  - sparse matrices,
  - linear solves,
  - nonlinear solves,
  - time-step linear systems
- SLEPc:
  - generalized eigenproblems for modal analysis
- DOLFINx / UFL:
  - form definition and assembly where a weak-form representation is natural

## 5.2 Important technical caveat

A direct one-to-one replacement of the current beam element implementation with a pure stock DOLFINx beam formulation is not trivial.

Reasons:

- FEniCSx does not provide a ready-made orchard-specific Euler-Bernoulli beam stack;
- Euler-Bernoulli beam formulations with rotational DOFs and high continuity are more specialized than standard Lagrange scalar/vector PDEs;
- point attachments, branch joints, fruit couplings, and localized nonlinear springs are not a simple textbook DOLFINx demo problem.

Therefore the practical P1 strategy is:

1. make PETSc/SLEPc the new sparse algebra/eigensolver path for the Python backend;
2. keep the current C++ backend alive and comparable throughout migration;
3. build a Python-side orchard FEM layer that owns model assembly and solver orchestration;
4. use UFL/DOLFINx for form-based operators where it is productive;
5. allow transitional PETSc-side assembled operators for beam/joint/attachment couplings where a custom beam formulation is still maturing.

This means the repository will temporarily support two solver backends:

- C++ backend: stable baseline and fallback
- Python/FEniCSx/PETSc backend: migration target and ML-facing orchestration path

This dual-backend period is intentional.
This still satisfies the architectural goal: new numerical infrastructure moves toward PETSc/SLEPc/FEniCSx, while C++ remains a supported project core rather than a discarded prototype.

## 5.3 Recommended first backend split

- orchard domain representation:
  - Python, backend-agnostic
- equivalent section integration:
  - Python, pure numerical geometry utilities
- global algebra:
  - PETSc
- eigen solve:
  - SLEPc
- time integration:
  - Python orchestration + PETSc linear/nonlinear solves
- DOLFINx/UFL:
  - introduced first for form-managed operators and later expanded where it pays off

## 5.4 Solver mapping

### Modal analysis

- current target: `K phi = lambda M phi`
- future backend: SLEPc generalized Hermitian eigensolver

### Frequency response

- current target: `Z(w) = K - w^2 M + i w C`
- future backend: PETSc complex sparse solve
- scan orchestration remains orchard-specific

### Time history

- keep Newmark-family method initially for compatibility
- use PETSc linear solves per time step
- use SNES when nonlinear local laws are active
- HHT-alpha can be added after parity

## 6. Orchard-Specific Cross-Section and Tissue Partition Plan

This is the main domain-value module and should not be diluted by backend migration.

The target Python-side cross-section subsystem must provide:

- parameterized non-circular sections,
- polygon/contour-defined sections,
- eccentric and irregular regions,
- axial section interpolation,
- explicit xylem / pith / phloem partitioning,
- equivalent section integrals:
  - `EA`
  - `EI_y`
  - `EI_z`
  - `GJ`
  - `rhoA`
  - optional effective damping weights

The integration result becomes the contract with the FEM layer.

That means the FEM backend should not own orchard tissue logic. It should consume equivalent coefficients and localized constitutive data produced by `orchard_fem.cross_section` and `orchard_fem.materials`.

## 7. Local Nonlinearity Plan

The current repository already identifies the right nonlinear hotspots:

- clamp region,
- branch joints / bifurcations,
- fruit attachment couplings,
- Duffing-like local restoring laws,
- gap-like piecewise behavior.

Migration rule:

- keep nonlinear locality explicit,
- do not spread strong nonlinearity uniformly over the entire tree,
- keep nonlinear laws modular and callable as orchard-level constitutive objects.

Recommended implementation path:

- P1:
  - preserve current cubic and gap spring semantics in Python,
  - evaluate them as localized residual/tangent contributions,
  - keep time integrator structure compatible with current verification.
- P2:
  - formalize joint/clamp constitutive objects with clearer physical units and tangent evaluation,
  - expose them to PETSc nonlinear solve paths.

## 8. PINN / Surrogate Strategy

PINN / surrogate modules are auxiliary. They do not replace the FEM backbone.

### 8.1 Entry point A: parameter inversion

Target:

- infer a small set of uncertain parameters from sparse synthetic or experimental response data.

Good candidate parameters:

- `E_xylem`
- global density scaling
- Rayleigh `alpha`, `beta`
- clamp stiffness
- one joint nonlinearity parameter

Do not invert too many weakly identifiable parameters at once in the first MVP.

### 8.2 Entry point B: local nonlinear surrogate

Target:

- replace the most expensive local nonlinear constitutive evaluation or submodel.

Best candidates:

- joint restoring law,
- clamp local nonlinear mapping,
- fruit-peduncle local restoring relation.

### 8.3 Entry point C: parameter scan surrogate

Target:

- predict orchard response indicators rapidly from a low-dimensional parameter vector.

Candidate outputs:

- peak displacement,
- peak fruit response,
- dominant resonance shift,
- harvesting proxy metrics.

## 9. DeepXDE vs Pure PyTorch Assessment

### 9.1 Recommendation

Use pure PyTorch as the default ML stack in this repository.

DeepXDE should remain optional and should not be a core dependency in P0-P3.

### 9.2 Why pure PyTorch is the better default here

The first ML tasks in this repository are not primarily PDE-discovery tasks. They are:

- parameter inversion with a forward solver in the loop,
- local constitutive surrogates,
- response-surface surrogates.

These are easier to control with plain PyTorch because:

- training loops need close control over solver calls and custom loss terms;
- data handling and batching are straightforward;
- integration with HDF5 and orchard-specific metadata is simpler;
- dependency risk is lower;
- debugging is easier when coupling with FEniCSx/PETSc-generated data.

### 9.3 When DeepXDE becomes attractive

DeepXDE may be useful later if the project adds:

- PDE-residual-heavy PINNs for local continuum subproblems,
- operator-learning workflows that match DeepXDE abstractions,
- faster experimentation on standard PINN baselines.

### 9.4 Decision for this project

- P0-P5 default: pure PyTorch
- DeepXDE: optional experimental dependency, not required for baseline workflow

## 10. Dependency Management Plan

### 10.1 Recommended environment model

Use a conda environment file as the primary environment specification.

Reason:

- FEniCSx/DOLFINx/PETSc/SLEPc are materially easier to provision through conda-forge than through ad-hoc pip installation;
- PyTorch and scientific Python dependencies are also manageable there;
- it is the most realistic cross-platform research setup for this stack.

### 10.2 Primary dependency groups

#### Mandatory for P1

- Python 3.11 or 3.12
- `dolfinx`
- `petsc4py`
- `slepc4py`
- `mpi4py`
- `numpy`
- `scipy`
- `h5py`
- `pandas`
- `pyyaml`
- `matplotlib`
- `torch`

#### Useful for testing and tooling

- `pytest`
- `pytest-cov`
- `jupyter`
- `mypy`
- `ruff`

#### Optional

- `deepxde`
- `meshio`
- `pyvista`

### 10.3 C++ side during transition

Keep the current C++ CMake build working during P1-P2.

The repository therefore temporarily has two execution modes:

- legacy C++ reference path
- new Python/FEniCSx path

This is acceptable as long as:

- compatibility is maintained,
- benchmark scripts compare them explicitly,
- the C++ path remains clearly supported as reference / baseline / fallback backend.

## 11. Validation Gate by Phase

## P0

- document current architecture and migration plan;
- add dependency draft;
- do not change solver behavior.

## P1

- run current examples through both:
  - legacy C++ path
  - new Python/FEniCSx path
- compare:
  - frequency-response curves,
  - time-history output,
  - modal frequencies
- preserve current CSV semantics.

## P2

- add unit tests for:
  - section integration,
  - tissue partition handling,
  - equivalent property calculation,
  - localized nonlinear constitutive laws
- preserve Duffing-like regression capability.

## P3

- train and validate at least one inversion case on synthetic data;
- report parameter recovery error and forward re-simulation error.

## P4

- compare local surrogate vs baseline local nonlinear model:
  - force error,
  - tangent error,
  - time-step cost,
  - global response impact.

## P5

- compare scan surrogate vs baseline FEM on held-out test set:
  - `R^2`
  - relative peak error
  - runtime speedup

## P6

- documentation coherence,
- example reproducibility,
- final benchmark report.

If a phase fails its gate, the repository does not proceed to the next phase.

## 12. Phase-by-Phase Execution Plan

### P0: audit and design

Deliverables:

- this design document,
- dependency environment draft,
- directory refactor plan,
- change log entry.

### P1: FEM backbone introduction

Deliverables:

- `orchard_fem/` package skeleton,
- JSON compatibility loader,
- CSV compatibility writer,
- modal / frequency / time-history solver skeleton with PETSc/SLEPc backend,
- benchmark script against legacy outputs.

### P2: cross-section + nonlinearity hardening

Deliverables:

- explicit Python-side tissue and cross-section modules,
- equivalent-property tests,
- localized nonlinear constitutive module,
- documentation of simplifying assumptions and error controls.

### P3: inverse MVP

Deliverables:

- synthetic data generator,
- HDF5 dataset format,
- PyTorch inversion workflow,
- recovery benchmark and plots.

### P4: local surrogate

Deliverables:

- one local surrogate demo,
- integration path into orchard solver,
- error vs runtime comparison.

### P5: parameter scan surrogate

Deliverables:

- scan data generator,
- MLP surrogate baseline,
- held-out evaluation,
- prediction interface.

### P6: packaging and documentation

Deliverables:

- updated README,
- architecture docs,
- PINN workflow docs,
- organized examples and scripts,
- coherent baseline and benchmark story.

## 13. First Batch of Files for P1

These are the first files that should change after P0 review approval:

- `config/fenicsx_pinn_environment.yml`
- `orchard_fem/__init__.py`
- `orchard_fem/topology/tree.py`
- `orchard_fem/cross_section/profile.py`
- `orchard_fem/cross_section/tissue.py`
- `orchard_fem/cross_section/integrator.py`
- `orchard_fem/io/json_schema.py`
- `orchard_fem/io/csv_writer.py`
- `orchard_fem/solvers/modal.py`
- `orchard_fem/solvers/frequency_response.py`
- `orchard_fem/solvers/time_history.py`
- `scripts/benchmark_vs_existing.py`
- `tests/integration/`

Current C++ solver files should not be deleted in P1. They remain supported while the Python backend proves parity.

## 14. Key Risks

### Risk 1: overcommitting to pure DOLFINx beam forms too early

If the project tries to force a complete orchard beam/joint/attachment implementation into UFL immediately, P1 will stall.

Mitigation:

- use PETSc/SLEPc backbone first,
- bring in UFL/DOLFINx incrementally where form assembly is actually helpful.

### Risk 2: losing orchard-specific section semantics during backend migration

Mitigation:

- freeze orchard section/tissue interfaces first,
- make equivalent property integration the formal contract with the solver.

### Risk 3: introducing ML before data contracts are stable

Mitigation:

- define HDF5/CSV data schema before P3,
- keep surrogate inputs/outputs small and explicit.

### Risk 4: breaking current examples and CSV output

Mitigation:

- preserve JSON schema and CSV schema as compatibility gates,
- add benchmark scripts early in P1.

### Risk 5: too many uncertain parameters in first inversion task

Mitigation:

- constrain the first inverse problem to a small identifiable subset,
- use synthetic data first,
- require forward re-simulation checks.

## 15. P0 Conclusion

The repository is already strong on orchard-specific semantics and regression assets. The main weakness is numerical backend ownership.

The correct refactor direction is not to discard the current model layer. It is to:

- preserve orchard-specific domain objects,
- preserve C++ as a stable project core,
- add PETSc/SLEPc-backed Python orchestration alongside the C++ backend,
- introduce DOLFINx/UFL where it adds value,
- add PyTorch-based inversion and surrogate modules only after the FEM data path is stable.
