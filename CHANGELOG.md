# Changelog

## 2026-07-13

### Changed

- Separated version-controlled source from machine-local artifacts under one configurable `workspace/`.
- Moved the five reproducible tree models to `examples/trees/` and centralized runtime paths through `orchard_fem.workspace`.
- Reclassified tests into unit, integration, FEniCSx-backend, and verification scopes with pytest markers.
- Made `pyproject.toml` the authoritative Python dependency source and renamed the solver environment to `config/orchard_fenicsx.yml`.
- Moved DS5L1 site calibration to the ignored workspace while retaining a tracked schema example.
- Added lightweight CI and resolved all Ruff findings.

### Removed

- Redundant `orchard_fem.topology.tree` re-export module.
- Empty `orchard_pinn.inverse` and `orchard_pinn.surrogate` placeholder packages.
- Duplicated dependency checklist files and `requirements.txt`.

## 2026-04-24

### Changed

- Reworked the repository into an Orchard FEM-only main branch with archived non-Python implementation removed from the active workflow.
- Simplified the active package surface to domain, topology, discretization, solver core, dynamics, workflows, visualization, automation, and loaders.
- Removed remaining compatibility and archival comparison code from the runtime path.
- Reorganized repository documentation around a cleaner GitHub-style structure with a focused `README.md`, `CONTRIBUTING.md`, and a `docs/` index.
- Activated explicit `joints[].law` handling in the Python beam assembler so polynomial and gap-style rotational joint nonlinearities now feed the transient nonlinear-link path.
- Added a formal nonlinear frequency-response path for localized nonlinear links.

### Added

- `CONTRIBUTING.md`
- `docs/README.md`
- `docs/getting_started.md`
- `docs/development.md`

## 2026-04-19

### Added

- `docs/design_fenicsx_pinn_refactor.md` with the P0 audit, migration plan, dependency assessment, validation gates, and risk register.
- `config/fenicsx_pinn_environment.yml` as a draft conda environment for the planned FEniCSx + PETSc/SLEPc + PyTorch stack.
- `pyproject.toml` for Python package and tooling configuration.
- `orchard_fem/` package skeleton with topology, cross-section, CSV compatibility, and solver-interface modules.
- `orchard_pinn/` package skeleton with shared surrogate/inverse metrics utilities.
- `tests/integration/test_python_scaffold.py` for the first Python-side compatibility smoke checks.
- `orchard_fem/model.py` and `orchard_fem/io/legacy_loader.py` for typed loading of the existing orchard JSON schema.
- `orchard_fem/solvers/modal.py` now includes a working dense generalized eigen solver and a SLEPc-backed modal solver entry point.
- `orchard_fem/materials/base.py`, `orchard_fem/elements/beam_formulation.py`, and `orchard_fem/solvers/modal_assembler.py` for a first Python-side orchard modal assembly path.
- `scripts/check_python_env.py` for a repo-local dependency audit against `pyproject.toml`, the conda environment draft, and required external build tools.

### Notes

- No solver-path behavior was changed in this phase.
- Existing JSON schema, CSV schema, examples, and verification meaning were preserved.
- The Python modal path is additive and currently serves as a refactor baseline.
