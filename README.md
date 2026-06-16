# Orchard FEM

Finite-element tooling for orchard tree vibration modelling, simulation, and validation.

Orchard FEM focuses on fruit-tree structures rather than general structural analysis. It models
trunk/branch/fruit topology, tissue-aware branch sections, harmonic excitation, modal response,
frequency response, time-history response, and repeatable validation workflows.

## Features

- Hierarchical trunk, branch, and fruit models.
- Circular and tissue-partitioned branch sections for xylem, pith, and phloem.
- FEniCSx/PETSc/SLEPc solver backend for modal, frequency-response, and time-history analyses.
- Native Python beam backend for lightweight comparison and fallback runs.
- Gravity prestress, fruit attachments, linear branch constraints, localized nonlinear links, and
  prescribed harmonic displacement/force/acceleration excitation.
- CSV output plus matplotlib plotting commands for frequency-response and time-history results.
- Import tools for skeleton, topology, and TreeQSM-style branch inputs.
- Integration and verification tests for solver behavior and regression control.

## Repository Layout

```text
orchard_fem/       Main solver package, CLI, workflows, post-processing, and visualization
orchard_pinn/      Placeholder package for future surrogate/inverse modelling work
examples/          Small runnable example models
trees/             Orchard architecture examples and generated solver-ready models
docs/              User, input-format, architecture, verification, and development notes
config/            Conda/environment setup files
tests/             Integration and verification tests
build/             Generated local artifacts; safe to recreate
```

## Installation

Python 3.11 or newer is expected.

### Lightweight Development Environment

Use this for code inspection, importers, plotting, and tests that do not require FEniCSx.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ubuntu-test]"
python -m orchard_fem doctor
```

### FEniCSx Solver Environment

Use this for the default solver backend and PETSc/SLEPc-backed analyses.

```bash
conda env create -f config/fenicsx_pinn_environment.yml
conda activate orchard-fenicsx
python -m pip install -e ".[ubuntu-test]"
python -m orchard_fem doctor
```

If the environments already exist, typical local usage is:

```bash
conda run -n orchard-fenicsx python -m pytest -q tests/integration
conda run -n orchard-fenicsx python -m orchard_fem --help
```

## Quick Start

Run the frequency-response demo:

```bash
python -m orchard_fem run examples/demo_orchard.json \
  --output-csv build/demo_frequency_response.csv

python -m orchard_fem plot-frequency-response build/demo_frequency_response.csv \
  --no-show --output build/demo_frequency_response.png
```

Run the time-history demo:

```bash
python -m orchard_fem run examples/demo_orchard_time_history.json \
  --output-csv build/demo_time_history.csv

python -m orchard_fem plot-time-history build/demo_time_history.csv \
  --no-show --output build/demo_time_history.png
```

Run the open-vase tree model:

```bash
python -m orchard_fem run trees/tree_1_multi_stem_open_crown.json \
  --output-csv build/results/tree_1_multi_stem_open_crown/frf.csv

python -m orchard_fem plot-frequency-response \
  build/results/tree_1_multi_stem_open_crown/frf.csv \
  --no-show

python -m orchard_fem visualize \
  trees/tree_1_multi_stem_open_crown.json \
  build/results/tree_1_multi_stem_open_crown/frf.csv \
  --output-prefix build/results/tree_1_multi_stem_open_crown/visual

python -m orchard_fem view-tree \
  trees/tree_1_multi_stem_open_crown.json \
  --no-show \
  --output build/results/tree_1_multi_stem_open_crown/tree_3d.png
```

The time-history plotter writes one excitation figure plus one figure per branch when an output path
is provided.

## CLI

The primary entry point is:

```bash
python -m orchard_fem --help
```

Editable installs also provide:

```bash
orchard-fem --help
```

Common commands:

| Command | Purpose |
| --- | --- |
| `run` | Execute the analysis configured in a model JSON and write a response CSV. |
| `modal` | Solve modal frequencies and write a modal summary CSV. |
| `batch-run` | Run several frequency-response excitation specs against one model. |
| `plot-frequency-response` | Plot a frequency-response CSV. |
| `plot-time-history` | Plot acceleration time histories from a time-history CSV. |
| `visualize` | Generate geometry, response, and trajectory figures from a model and CSV. |
| `import-skeleton` | Convert 3D branch skeleton JSON into solver-ready Orchard FEM JSON. |
| `import-topology` | Convert length/diameter/attachment topology JSON into solver-ready JSON. |
| `import-treeqsm` | Convert TreeQSM-style branch data into solver-ready JSON. |
| `calibrate` | Calibrate material Young's moduli against measured modal frequencies. |
| `harvest` | Estimate fruit detachment response and harvest-frequency candidates. |
| `compare-frf` | Compare measured and simulated FRF data. |
| `demo-suite` | Regenerate standard demo CSV and figure artifacts. |
| `verify` | Run validation in the active environment. |
| `full-validate` | Orchestrate multi-environment validation. |
| `doctor` | Report missing runtime dependencies in the active environment. |

Backend selection can be set in the model JSON through `analysis.solver_backend`, or overridden from
the CLI where supported:

```bash
python -m orchard_fem run examples/demo_orchard.json --solver-backend native
python -m orchard_fem modal trees/tree_1_multi_stem_open_crown.json --solver-backend fenicsx
```

## Inputs And Outputs

Model input is JSON. The most useful starting points are:

- `examples/demo_orchard.json`: compact frequency-response model.
- `examples/demo_orchard_time_history.json`: compact time-history model.
- `trees/tree_1_multi_stem_open_crown.json`: image-derived multi-stem open-crown model.
- `trees/tree_2_asymmetric_multi_leader.json`, `trees/tree_3_broad_vase_multi_leader.json`,
  `trees/tree_4_y_fan_canopy.json`, `trees/tree_5_rounded_two_leader.json`: additional architecture
  examples.

Solver outputs are CSV files. Plot commands write PNG figures when `--output` is provided. Generated
files are usually written under `build/` or `results/`.

See [docs/input_format.md](docs/input_format.md) for the model schema.

## Testing And Validation

Python integration tests:

```bash
conda run -n orchard-fenicsx python -m pytest -q tests/integration
```

FEniCSx/PETSc/SLEPc tests:

```bash
ORCHARD_RUN_DOLFINX_TESTS=1 \
conda run -n orchard-fenicsx python -m pytest -q \
  tests/integration/test_fenicsx_time_history.py \
  tests/integration/test_fenicsx_modal.py \
  tests/integration/test_fenicsx_frequency_response.py
```

Project validation commands:

```bash
python -m orchard_fem verify
python -m orchard_fem full-validate
```

## Documentation

- [docs/README.md](docs/README.md): documentation index.
- [docs/getting_started.md](docs/getting_started.md): installation and first runs.
- [docs/input_format.md](docs/input_format.md): model JSON reference.
- [docs/orchard_fem_architecture.md](docs/orchard_fem_architecture.md): package architecture.
- [docs/verification.md](docs/verification.md): verification strategy and benchmark coverage.
- [docs/development.md](docs/development.md): developer workflow.
- [docs/solver_roadmap.md](docs/solver_roadmap.md): solver roadmap and implementation status.

## Current Scope

The default production path is the FEniCSx backend. The native backend remains useful for lightweight
checks and comparison runs, but not every advanced feature is implemented there.

Nonlinear frequency response and nonlinear transient workflows are available for localized link
models, but they should be treated as active solver-development surfaces. Prefer the verification
commands above after changing material models, joints, prestress, fruit coupling, or time integration.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
