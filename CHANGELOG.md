# Changelog

## 2026-04-24

### Changed

- Reworked the repository into an Orchard FEM-only main branch with the archived C++ implementation removed from the active workflow.
- Simplified the active package surface to domain, topology, discretization, solver core, dynamics, workflows, visualization, automation, and loaders.
- Removed remaining compatibility and archival comparison code from the runtime path.
- Reorganized repository documentation around a cleaner GitHub-style structure with a focused `README.md`, `CONTRIBUTING.md`, and a `docs/` index.

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
- `scripts/benchmark_vs_existing.py` to keep the C++ backend as the comparison baseline while Python/FEniCSx is introduced.
- `tests/integration/test_python_scaffold.py` for the first Python-side compatibility smoke checks.
- `orchard_fem/model.py` and `orchard_fem/io/legacy_loader.py` for typed loading of the existing orchard JSON schema.
- `orchard_fem/solvers/modal.py` now includes a working dense generalized eigen solver and a SLEPc-backed modal solver entry point.
- `orchard_fem/materials/base.py`, `orchard_fem/elements/beam_formulation.py`, and `orchard_fem/solvers/modal_assembler.py` for a first Python-side orchard modal assembly path that mirrors the current C++ beam-based formulation.
- `scripts/benchmark_vs_existing.py` can now also emit a Python modal summary CSV while preserving the C++ runtime baseline workflow.
- `scripts/check_python_env.py` for a repo-local dependency audit against `pyproject.toml`, the conda environment draft, and required external build tools.

### Notes

- No solver-path behavior was changed in this phase.
- Existing JSON schema, CSV schema, examples, and verification meaning were preserved.
- The Python modal path is additive and currently serves as a refactor baseline; the C++ executable remains the primary backend and comparison reference.
