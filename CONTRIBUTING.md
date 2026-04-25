# Contributing

This repository is maintained as an Orchard FEM project, not as a temporary transition workspace.  
New work should go into the active Python package and its documentation, tests, and examples.

## Principles

- Keep the active implementation inside `orchard_fem/`.
- Do not reintroduce removed build roots or alias shims.
- Prefer improving package-native workflows over adding ad-hoc scripts.
- Add or update verification coverage when solver behavior changes.
- Keep documentation aligned with the active CLI and package structure.

## Development Environments

### Lightweight Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ubuntu-test]"
python -m orchard_fem doctor
```

### Full PETSc/SLEPc Environment

```bash
conda env create -f config/fenicsx_pinn_environment.yml
conda activate orchard-fenicsx
python -m orchard_fem doctor
```

## Everyday Commands

Run a model:

```bash
python -m orchard_fem run examples/demo_orchard.json
```

Run package validation:

```bash
python -m orchard_fem verify
```

Run multi-environment validation:

```bash
python -m orchard_fem full-validate
```

Regenerate demo artifacts:

```bash
python -m orchard_fem demo-suite --output-dir build/validation/python
```

## Testing Expectations

If you touch package structure, loaders, workflows, or CLI:
- run `python -m orchard_fem verify`

If you touch PETSc/SLEPc-backed solver behavior:
- run the targeted `orchard-fenicsx` verification command from [docs/verification.md](docs/verification.md)
- or run `python -m orchard_fem full-validate`

If you change user-facing examples or plotting behavior:
- run the relevant demo command and check generated outputs in `build/`

## Documentation Expectations

Update documentation whenever you change:
- CLI behavior
- environment setup
- repository structure
- input format
- verification requirements

At minimum, keep these pages in sync when relevant:
- `README.md`
- `docs/README.md`
- `docs/getting_started.md`
- `docs/development.md`
- `docs/verification.md`

## Code Organization

Use these active package surfaces:
- `orchard_fem/domain/` for domain objects and parsing helpers.
- `orchard_fem/topology/` for geometry, branch-path, and tree graph logic.
- `orchard_fem/discretization/` for beam, DOF, and assembled-system logic.
- `orchard_fem/solver_core/` for modal and preload core operations.
- `orchard_fem/dynamics/` for frequency-response and time-history execution.
- `orchard_fem/io/loaders/` for model loading.
- `orchard_fem/workflows/` for reusable orchestration.
- `orchard_fem/commands/` for CLI command registration.

## Pull Request Checklist

Before submitting changes, make sure:
- the code lives in the active Orchard FEM package surface
- the relevant validation commands pass
- stale transitional notes are not reintroduced
- documentation matches the shipped CLI and workflows
- example commands in docs still work
