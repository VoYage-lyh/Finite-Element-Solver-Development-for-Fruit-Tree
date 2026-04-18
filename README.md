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
- Output helper: `scripts/plot_frequency_response.py`
- Tests: section geometry, material loading, topology assembly, matrix assembly, frequency response, nonlinear time history

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

- frequency-response analysis for scan-friendly linearized studies,
- Newmark time-history analysis with localized joint/clamp nonlinear links,
- polynomial joint nonlinearity and gap-ready joint interface,
- fruit mass-spring-damper attachment coupling,
- CSV output for both frequency and time histories.

Current limitation:

- `CMakeLists.txt` is provided, but in the current local environment `cmake` is not available on `PATH`, so validation was performed with `g++` directly.
- Frequency-response analysis currently uses the zero-amplitude linearized system; localized nonlinearities are fully active in time-history analysis.

## Documentation

- Design document: [docs/design.md](docs/design.md)
- Input format reference: [docs/input_format.md](docs/input_format.md)
