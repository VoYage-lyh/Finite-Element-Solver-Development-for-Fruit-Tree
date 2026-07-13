# Getting Started

This guide is the shortest path from a fresh checkout to a working Orchard FEM run.

## 1. Create an Environment

### Lightweight Local Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,viz,vision]"
```

Use this only for code inspection and explicit native fallback tests.

### Recommended PETSc/SLEPc Environment

```bash
conda env create -f config/orchard_fenicsx.yml
conda activate orchard-fenicsx
python -m pip install -e . --no-deps
```

Use this for normal solver runs. The default backend is FEniCSx.

## 2. Check the Environment

```bash
python -m orchard_fem doctor
```

Use this before debugging runtime issues. It reports missing Python packages and points to the expected environment files.

## 3. Run a First Frequency-Response Example

```bash
python -m orchard_fem run examples/trees/tree_3.json --output-csv build/demo_frequency_response.csv
python -m orchard_fem visualize examples/trees/tree_3.json build/demo_frequency_response.csv --output-prefix build/demo_frequency_response
```

## 4. Run a First Time-History Example

```bash
python -m orchard_fem run tests/fixtures/demo_orchard_time_history.json --output-csv build/demo_time_history.csv
python -m orchard_fem visualize tests/fixtures/demo_orchard_time_history.json build/demo_time_history.csv --output-prefix build/demo_time_history
```

## 5. Run Validation

Fast validation in the current environment:

```bash
python -m orchard_fem verify
```

Repository health check across the lightweight and PETSc/SLEPc workflows:

```bash
python -m orchard_fem full-validate
```

## 6. Where To Go Next

- Learn the model format in [input_format.md](input_format.md).
- Learn the verification model in [verification.md](verification.md).
- Learn the package structure in [orchard_fem_architecture.md](orchard_fem_architecture.md).
- Learn contributor workflow in [development.md](development.md).
- Learn the optional dependency and local-data layout in [environment_setup.md](environment_setup.md).
