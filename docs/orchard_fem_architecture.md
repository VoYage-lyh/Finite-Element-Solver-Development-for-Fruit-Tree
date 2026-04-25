# Orchard FEM Architecture

This document describes the active package architecture on the main branch.

## Design Intent

Orchard FEM is organized around orchard-specific solver workflows:

- typed orchard model loading
- beam-based discretization
- modal, frequency-response, and time-history execution
- reproducible validation and visualization

The architecture is intentionally package-first.  
The active implementation surface is `orchard_fem/`.

## Package Layers

### Interface Layer

- `orchard_fem/commands/`: CLI command registration and argument parsing
- `orchard_fem/application.py`: top-level orchestration facade used by the CLI

### Domain And Geometry Layer

- `orchard_fem/domain/`: enums, dataclasses, and parsing helpers
- `orchard_fem/topology/`: vectors, branch paths, observations, and tree graph logic
- `orchard_fem/cross_section/`: section profiles, tissue regions, and defaults
- `orchard_fem/materials/`: material properties and branch-average material helpers

### Solver Layer

- `orchard_fem/discretization/`: beam properties, DOFs, assembled systems, and matrix utilities
- `orchard_fem/solver_core/`: modal and preload core operations
- `orchard_fem/dynamics/`: frequency-response and time-history execution
- `orchard_fem/numerics/`: PETSc/SLEPc backend helpers

### Workflow Layer

- `orchard_fem/io/loaders/`: model payload and typed-model loading
- `orchard_fem/workflows/`: reusable run, demo, and validation workflows
- `orchard_fem/visualization/`: geometry and response rendering
- `orchard_fem/postprocess/`: postprocessing helpers such as direct CSV plotting
- `orchard_fem/automation/`: multi-environment validation orchestration

### Reserved Extension Surfaces

- `orchard_fem/model_reduction/`
- `orchard_fem/error_metrics.py`
- `orchard_pinn/`

These remain available for future reduction, surrogate, and reporting work.

## Workflow Flow

The normal package flow is:

1. load a model through `orchard_fem.io.loaders`
2. build domain and topology objects
3. assemble beam-based systems in `orchard_fem.discretization`
4. solve through `orchard_fem.solver_core` and `orchard_fem.dynamics`
5. expose results through workflows, CSV output, and visualization

## Validation Boundaries

Correctness is enforced through:

- integration tests
- analytical beam benchmarks
- dynamic engineering benchmarks
- PETSc/SLEPc gravity-prestress regression
- standard demo artifact regeneration

## Repository Boundaries

Active project roots:

- `orchard_fem/`
- `orchard_pinn/`
- `examples/`
- `tests/`
- `docs/`
- `config/`

For stage-by-stage implementation status, see [solver_roadmap.md](solver_roadmap.md).
