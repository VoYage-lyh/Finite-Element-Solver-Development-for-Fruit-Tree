# Orchard FEM

Orchard FEM is a Python-based finite-element framework for orchard tree vibration analysis.  
It is built for orchard-specific dynamics rather than general-purpose structural analysis, with a focus on branch topology, tissue-aware sections, excitation-response studies, and repeatable validation workflows.

## What The Project Does

- Models hierarchical trunk-branch-fruit structures.
- Supports tissue-partitioned branch sections such as xylem, pith, and phloem.
- Assembles 3D Euler-Bernoulli beam systems with orchard-specific attachments and boundary conditions.
- Runs modal, frequency-response, and time-history analyses.
- Produces CSV outputs and report-ready plots from the same CLI workflow.
- Ships analytical and engineering verification cases for regression control.

## Current Scope

The active implementation lives in the `orchard_fem` package.

Implemented today:
- PETSc/SLEPc-backed modal analysis.
- Frequency-response and Newmark time-history workflows.
- Multi-component observations and trajectory plots.
- Gravity prestress, default circular section helpers, explicit rotational joint laws, and auto nonlinear-link injection.
- Package-native validation and demo regeneration commands.

Current limitations:
- Nonlinear frequency-response currently uses warm-started steady-state time-domain sweep fallback rather than harmonic balance or full continuation.
- Nonlinear joint laws currently act on rotational root DOFs; they are available in transient analysis and in nonlinear sweep-based frequency response, but not yet in a dedicated continuation workflow.
- The discretization layer is still a manual Euler-Bernoulli beam implementation rather than the planned FEniCSx/UFL branch.

## Quick Start

### Lightweight Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ubuntu-test]"
python -m orchard_fem doctor
```

### Recommended PETSc/SLEPc Environment

```bash
conda env create -f config/fenicsx_pinn_environment.yml
conda activate orchard-fenicsx
python -m orchard_fem doctor
```

## First Run

Frequency-response demo:

```bash
python -m orchard_fem run examples/demo_orchard.json --output-csv build/demo_frequency_response.csv
python -m orchard_fem visualize examples/demo_orchard.json build/demo_frequency_response.csv --output-prefix build/demo_frequency_response
```

Time-history demo:

```bash
python -m orchard_fem run examples/demo_orchard_time_history.json --output-csv build/demo_time_history.csv
python -m orchard_fem visualize examples/demo_orchard_time_history.json build/demo_time_history.csv --output-prefix build/demo_time_history
```

## Validation

Fast package-level validation:

```bash
python -m orchard_fem verify
```

Multi-environment validation:

```bash
python -m orchard_fem full-validate
```

Recommended PETSc/SLEPc verification command:

```bash
conda run -n orchard-fenicsx python -m pytest -q \
  tests/verification/test_python_beam_benchmarks.py \
  tests/verification/test_python_dynamic_benchmarks.py \
  tests/integration/test_gravity_prestress.py::test_gravity_prestress_adds_load_and_reduces_first_mode
```

## CLI Surface

The primary entry point is:

```bash
python -m orchard_fem --help
```

Available commands:
- `run`: execute the analysis configured in a model JSON.
- `modal`: export modal frequencies and summary data.
- `visualize`: generate geometry, response, and trajectory figures.
- `plot-frequency-response`: plot a frequency-response CSV directly.
- `demo-suite`: regenerate standard demo artifacts.
- `verify`: run validation in the current environment.
- `full-validate`: orchestrate `orchard-dev` and `orchard-fenicsx` validation flows.
- `doctor`: audit the active Python environment.

After editable install, the console script alias is also available:

```bash
orchard-fem --help
```

## Repository Map

- `orchard_fem/`: active solver, CLI, workflow, and visualization package.
- `orchard_pinn/`: reserved surface for future surrogate and inversion work.
- `examples/`: runnable JSON models.
- `tests/`: integration and verification coverage.
- `docs/`: user, developer, architecture, and roadmap documentation.
- `config/`: environment files and setup guides.

## Documentation

- [docs/README.md](docs/README.md): documentation index.
- [docs/getting_started.md](docs/getting_started.md): installation, first runs, and validation basics.
- [docs/development.md](docs/development.md): developer workflow and repository conventions.
- [docs/orchard_fem_architecture.md](docs/orchard_fem_architecture.md): active package architecture.
- [docs/solver_roadmap.md](docs/solver_roadmap.md): implementation status against the current solver roadmap.
- [docs/input_format.md](docs/input_format.md): model JSON reference.
- [docs/verification.md](docs/verification.md): verification strategy and benchmark coverage.

## Contributing

Contribution guidelines live in [CONTRIBUTING.md](CONTRIBUTING.md).
