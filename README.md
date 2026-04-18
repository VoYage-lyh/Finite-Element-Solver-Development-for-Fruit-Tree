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
- Modular C++20 core with Python-based preprocessing and workflow scripts.
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
|-- apps/
|-- CMakeLists.txt
|-- README.md
|-- config/
|-- docs/
|   `-- design.md
|-- examples/
|-- include/
|-- scripts/
|-- src/
`-- tests/
```

## MVP contents

- CLI app: `apps/orchard_cli.cpp`
- Demo model: `examples/demo_orchard.json`
- Time-history demo: `examples/demo_orchard_time_history.json`
- Output helpers: `scripts/plot_frequency_response.py`, `scripts/visualize_analysis.py`
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

1. project skeleton and CMake layout,
2. orchard topology, cross-section, materials, fruits, joints, excitation, and solver modules,
3. demo JSON model and CLI frequency-response run,
4. baseline automated tests.

Current implemented solver features:

- 3D Euler-Bernoulli beam assembly with 6 DOFs per branch node,
- frequency-response analysis for scan-friendly linearized studies,
- Newmark time-history analysis with localized nonlinear-link support,
- legacy polynomial joint and gap-law interfaces preserved for follow-up joint refactor work,
- fruit mass-spring-damper attachment coupling,
- CSV output for both frequency and time histories, including excitation-point channels for visualization.

Current limitation:

- `CMakeLists.txt` is provided, but in the current local environment `cmake` is not available on `PATH`, so validation was performed with `g++` directly.
- Frequency-response analysis currently uses the zero-amplitude linearized system; localized nonlinearities are fully active in time-history analysis.
- After the beam-element upgrade, clamp cubic nonlinearity is active, but joint nonlinearity still needs a dedicated 6-DOF beam-joint mapping pass.
- `scripts/visualize_analysis.py` uses `matplotlib`/`numpy` when available and falls back to standalone SVG generation otherwise. The fallback renderer is static rather than interactive, but it still shows the orchard geometry, excitation point, measurement points, and response panels.

## Visualization

Frequency-response workflow:

```text
orchard_cli examples/demo_orchard.json build/demo_frequency_response.csv
python scripts/visualize_analysis.py examples/demo_orchard.json build/demo_frequency_response.csv --output-prefix build/demo_frequency_response
```

Time-history workflow with excitation/measurement time-frequency output:

```text
orchard_cli examples/demo_orchard_time_history.json build/demo_time_history.csv
python scripts/visualize_analysis.py examples/demo_orchard_time_history.json build/demo_time_history.csv --output-prefix build/demo_time_history
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
- Run `ctest -L verification` before merging any change that touches `solver_core`, `branches`, or `discretization`.
- Current verification coverage includes cantilever modal frequencies, simply supported beam static deflection, Duffing hardening response, and a hinged two-bar spring-mass benchmark.

## Documentation

- Design document: [docs/design.md](docs/design.md)
- Input format reference: [docs/input_format.md](docs/input_format.md)
- Verification suite: [docs/verification.md](docs/verification.md)
