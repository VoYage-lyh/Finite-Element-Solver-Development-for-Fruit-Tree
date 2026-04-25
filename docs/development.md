# Development Guide

This document describes how to work on Orchard FEM as a repository, not just how to run a demo.

## Active Implementation Surface

New code should go into these package layers:

- `orchard_fem/domain/`
- `orchard_fem/topology/`
- `orchard_fem/cross_section/`
- `orchard_fem/materials/`
- `orchard_fem/discretization/`
- `orchard_fem/solver_core/`
- `orchard_fem/dynamics/`
- `orchard_fem/io/loaders/`
- `orchard_fem/workflows/`
- `orchard_fem/commands/`
- `orchard_fem/visualization/`
- `orchard_fem/automation/`

## Repository Boundaries

Active work should stay inside the shipped Orchard FEM project roots:

- `orchard_fem/`
- `orchard_pinn/`
- `examples/`
- `tests/`
- `docs/`
- `config/`

## Common Workflows

Run a model:

```bash
python -m orchard_fem run examples/demo_orchard.json
```

Generate plots:

```bash
python -m orchard_fem visualize examples/demo_orchard.json build/demo_frequency_response.csv
```

Run validation:

```bash
python -m orchard_fem verify
python -m orchard_fem full-validate
```

Regenerate standard demo artifacts:

```bash
python -m orchard_fem demo-suite --output-dir build/validation/python
```

## Testing Matrix

Use the smallest test scope that still covers your change.

Lightweight package validation:
- `python -m orchard_fem verify`

PETSc/SLEPc verification:
- use the command block in [verification.md](verification.md)

Full repository validation:
- `python -m orchard_fem full-validate`

## Documentation Rules

When behavior changes, update the docs in the same change set.

Typical sync points:
- `README.md` for user-facing entry points
- `docs/getting_started.md` for setup and first-run commands
- `docs/verification.md` for test and benchmark behavior
- `docs/orchard_fem_architecture.md` for package boundary changes

## Design Philosophy

The project is orchard-specific by design.
The goal is not to mimic a general-purpose commercial FEM suite.
Prefer changes that keep:

- orchard topology explicit
- simplifications testable
- workflows reproducible
- solver outputs easy to validate
