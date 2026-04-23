# Orchard Vibration Solver

A dedicated fast solver framework for orchard tree vibration analysis, focused on vibration harvesting and excitation-response studies.

## Why this project exists

This repository does **not** aim to clone a general-purpose commercial FEM package. It is designed around orchard-specific dynamics:

- hierarchical trunk-branch-fruit topology,
- non-circular and irregular branch sections,
- xylem/pith/phloem partitioned materials,
- localized strong nonlinearities,
- clamp excitation and harvest-oriented response metrics,
- controllable simplification for fast parameter studies.

The guiding principle is:

> keep the nonlinearities and heterogeneities that matter most to harvest-relevant response, simplify the rest in a controlled and testable way.

## Design priorities

- Orchard specificity over generic FEM breadth.
- Fast repeated simulation over uniform maximum fidelity.
- Explicit error assessment alongside each major simplification.
- Python-first PETSc/SLEPc workflow with clean hooks for the FEniCSx stack.
- Clean interfaces for future reduction, enrichment, and backend upgrades.

## Planned capabilities

- Tree topology and branch hierarchy modeling.
- Non-circular, axially varying branch cross-sections.
- Euler-Bernoulli beam discretization with branch-wise multi-node DOFs.
- Tissue-partitioned material modeling for xylem, pith, and phloem.
- Fruit added-mass and attachment effects.
- Clamp/excitation boundary conditions.
- Frequency-domain and time-domain dynamic response.
- Localized nonlinear links at joints and clamp regions.
- Parameter scans for excitation frequency, amplitude, and clamp position.
- Future support for reduction and local high-fidelity correction.

## Repository layout

```text
.
|-- orchard_fem/
|   |-- commands/
|   `-- workflows/
|-- orchard_pinn/
|-- README.md
|-- config/
|-- docs/
|-- examples/
|-- scripts/
|-- tests/
|-- legacy reference: apps/
|-- legacy reference: include/
|-- legacy reference: src/
`-- legacy reference: CMakeLists.txt
```

## MVP contents

- Python CLI: `python -m orchard_fem` or `orchard-fem`
- Demo model: `examples/demo_orchard.json`
- Time-history demo: `examples/demo_orchard_time_history.json`
- Output helpers: `python -m orchard_fem plot-frequency-response`, `python -m orchard_fem visualize`
- Tests: section geometry, material loading, topology assembly, beam-matrix assembly, demo frequency response, cantilever first mode

## Development phases

### Phase 0

- Freeze architecture and modeling principles.
- Draft input schema and test strategy.

### Phase 1

- Build the MVP orchard-specific solver core.
- Provide demo model, CLI runner, and baseline tests.

### Phase 2

- Add localized nonlinearities and stronger validation workflows.

### Phase 3

- Introduce reduction and acceleration strategies with explicit error tracking.

## MVP target

The first runnable milestone will support:

- tree topology definition,
- non-circular section definition,
- xylem/pith/phloem material partitions,
- simplified dynamic-system assembly,
- harmonic excitation,
- basic frequency response or time-history output,
- a demo example and automated tests.

## Status

This repository now includes a runnable orchard-specific MVP:

1. project skeleton and Python package layout,
2. orchard topology, cross-section, materials, fruits, joints, excitation, and solver modules,
3. demo JSON model and Python CLI frequency-response run,
4. baseline automated tests.

Current implemented solver features:

- 3D Euler-Bernoulli beam assembly with 6 DOFs per branch node,
- frequency-response analysis for scan-friendly linearized studies,
- Newmark time-history analysis with localized nonlinear-link support,
- Python-side SLEPc modal analysis and PETSc frequency-response workflow for the FEniCSx migration path,
- legacy polynomial joint and gap-law interfaces preserved for follow-up joint refactor work,
- fruit mass-spring-damper attachment coupling,
- CSV output for both frequency and time histories, including excitation-point channels for visualization.

Current limitation:

- Frequency-response analysis currently uses the zero-amplitude linearized system; localized nonlinearities are fully active in time-history analysis.
- Python time-history now has a PETSc/Newmark path for the current clamp-cubic transient workflow; joint-law nonlinearity is still pending the dedicated 6-DOF beam-joint mapping pass.
- `scripts/visualize_analysis.py` is a strict `matplotlib`/`numpy` workflow. If those packages are missing, it now fails fast and points to the required install command instead of switching to a hand-written renderer.

## Visualization

Frequency-response workflow:

```text
python -m orchard_fem run examples/demo_orchard.json --output-csv build/demo_frequency_response.csv
python -m orchard_fem visualize examples/demo_orchard.json build/demo_frequency_response.csv --output-prefix build/demo_frequency_response
```

Time-history workflow with excitation/measurement time-frequency output:

```text
python -m orchard_fem run examples/demo_orchard_time_history.json --output-csv build/demo_time_history.csv
python -m orchard_fem visualize examples/demo_orchard_time_history.json build/demo_time_history.csv --output-prefix build/demo_time_history
```

The visualizer produces:

- a geometry figure with branch layout, fruit locations, excitation point, and measurement points,
- a frequency-response figure for frequency sweeps,
- or a time-history/time-spectrum/spectrogram figure for transient runs.

Time-history CSV files now include:

- `excitation_signal`
- `excitation_load`
- `excitation_response`

## Verification

- Verification cases now live under `tests/verification/`.
- The Python-first analytical beam benchmarks live in `tests/verification/test_python_beam_benchmarks.py`.
- The Python-first dynamic regression benchmarks live in `tests/verification/test_python_dynamic_benchmarks.py`.
- In `orchard-fenicsx`, prefer `python -m pytest -q tests/verification/test_python_beam_benchmarks.py tests/verification/test_python_dynamic_benchmarks.py tests/integration/test_gravity_prestress.py::test_gravity_prestress_adds_load_and_reduces_first_mode`.
- Archived C++ verification executables now live behind the historical note in `docs/legacy_reference.md`.
- Current verification coverage includes cantilever modal frequencies, simply supported beam static deflection, Duffing hardening response, and a hinged two-bar spring-mass benchmark.

## Ubuntu 24 setup

- For Ubuntu 24.04 local build/test dependencies and validation commands, use `config/ubuntu24_test_dependencies.txt`.

## Full validation

The standard development workflow is now Python-first. For day-to-day validation, prefer:

```bash
python -m orchard_fem full-validate
```

It runs:

- the Python integration tests in `orchard-dev`,
- the PETSc/SLEPc gravity-prestress regression in `orchard-fenicsx`,
- and a Python PETSc/SLEPc demo suite that regenerates the standard frequency-response, time-history, and modal-summary artifacts.

The high-level Python application flow now has two explicit layers:

- `orchard_fem/commands/` as the modular CLI command layer,
- `orchard_fem/application.py` as the top-level orchestration facade used by the CLI,
- `orchard_fem/workflows/` as the shared run / demo / validation workflow layer,
- `orchard_fem/visualization/` as the package-native visualization subsystem,
- `orchard_fem/automation/`, `orchard_fem/postprocess/`, and `orchard_fem/legacy/` as package-native support layers for repository automation, plotting, and archived C++ diagnostics.

For day-to-day solver work, the primary entry point is now:

```bash
python -m orchard_fem --help
```

Or, after `pip install -e .`:

```bash
orchard-fem --help
```

The new Python-first validation command is:

```bash
python -m orchard_fem verify
```

The new Python-first environment audit command is:

```bash
python -m orchard_fem doctor
```

Useful overrides:

```bash
SKIP_FENICSX_TESTS=1 SKIP_PYTHON_DEMO_SUITE=1 python -m orchard_fem full-validate
```

```bash
BUILD_DIR=/tmp/orchard-build VALIDATION_DIR=/tmp/orchard-validation python -m orchard_fem full-validate
```

The repository still keeps `scripts/run_full_validation.sh`, but it is now only a thin wrapper around the package CLI.

If you intentionally need historical context from the old C++ path, keep it out of the main workflow and treat it as archival material:

```bash
python -m orchard_fem legacy-compare --help
```

## Documentation

- Design document: [docs/design.md](docs/design.md)
- Active architecture note: [docs/python_first_architecture.md](docs/python_first_architecture.md)
- Legacy reference note: [docs/legacy_reference.md](docs/legacy_reference.md)
- Input format reference: [docs/input_format.md](docs/input_format.md)
- Verification suite: [docs/verification.md](docs/verification.md)
