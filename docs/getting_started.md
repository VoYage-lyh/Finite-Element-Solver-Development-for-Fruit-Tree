# Getting Started

This guide is the shortest path from a fresh checkout to a working Orchard FEM run.

## 1. Create an Environment

### Lightweight Local Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ubuntu-test]"
```

### Recommended PETSc/SLEPc Environment

```bash
conda env create -f config/fenicsx_pinn_environment.yml
conda activate orchard-fenicsx
```

## 2. Check the Environment

```bash
python -m orchard_fem doctor
```

Use this before debugging runtime issues. It reports missing Python packages and points to the expected environment files.

## 3. Run a First Frequency-Response Example

```bash
python -m orchard_fem run examples/demo_orchard.json --output-csv build/demo_frequency_response.csv
python -m orchard_fem visualize examples/demo_orchard.json build/demo_frequency_response.csv --output-prefix build/demo_frequency_response
```

## 4. Run a First Time-History Example

```bash
python -m orchard_fem run examples/demo_orchard_time_history.json --output-csv build/demo_time_history.csv
python -m orchard_fem visualize examples/demo_orchard_time_history.json build/demo_time_history.csv --output-prefix build/demo_time_history
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
