# Environment setup

`pyproject.toml` is the authoritative Python dependency definition. The Conda
file only adds the FEniCSx/PETSc/SLEPc stack and packages that are more reliable
from Conda on Linux or WSL2.

## Lightweight development environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,viz,vision]"
python -m orchard_fem doctor
```

Install optional capabilities only when needed:

| Extra | Capability |
|---|---|
| `ml` | PyTorch, EfficientFormer (`timm`), and HDF5 utilities |
| `vision-sam` | Ultralytics SAM 2 proposals |
| `uq` | Bayesian calibration and Sobol sensitivity |
| `rig` | DS5L1 serial communication |

For the complete photo-segmentation workflow:

```bash
python -m pip install -e ".[dev,viz,vision,ml,vision-sam]"
```

## FEniCSx solver environment

Use Ubuntu, WSL2 Ubuntu, or native Linux:

```bash
conda env create -f config/orchard_fenicsx.yml
conda activate orchard-fenicsx
python -m pip install -e . --no-deps
python -m orchard_fem doctor
```

The YAML includes the solver, development, visualization, vision, UQ, and rig
runtime packages. `--no-deps` installs this repository itself without replacing
the Conda-provided numerical stack.

Useful runtime checks:

```bash
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from slepc4py import SLEPc; print(SLEPc.getVersion())"
python -c "import dolfinx; print(dolfinx.__version__)"
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Test scopes

```bash
# Fast isolated tests
python -m pytest -q -m "unit and not slow and not ml and not uq"

# Module/CLI integration tests
python -m pytest -q tests/integration

# FEniCSx backend surface; set the flag for tests that instantiate DOLFINx
ORCHARD_RUN_DOLFINX_TESTS=1 python -m pytest -q tests/backend/fenicsx

# Numerical benchmarks
python -m pytest -q tests/verification

# Complete suite in the active environment
python -m pytest -q
```

## Local workspace and rig calibration

Large data, checkpoints, outputs, caches, manuals, and site calibration live in
the ignored `workspace/` directory. Override its location with
`ORCHARD_WORKSPACE=/absolute/path`.

The DS5L1 calibration table defaults to
`workspace/config/ds5l1_freq_calib.json`. Set `ORCHARD_DS5L1_CALIB` to override
that single file. The tracked `config/ds5l1_freq_calib.example.json` documents
its schema.
