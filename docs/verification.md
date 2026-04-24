# Verification Suite

This document records the active Orchard FEM verification strategy.

The goal is not to certify a general-purpose FEM platform.
The goal is to keep a compact, orchard-focused regression shield around the parts of the solver that matter most:

- beam assembly
- boundary conditions
- modal behavior under refinement
- localized nonlinear time integration
- simplified reduced-joint dynamics

## How to run

Fast package validation:

```bash
python -m orchard_fem verify
```

Multi-environment validation:

```bash
python -m orchard_fem full-validate
```

PETSc/SLEPc verification in `orchard-fenicsx`:

```bash
conda run -n orchard-fenicsx python -m pytest -q \
  tests/verification/test_python_beam_benchmarks.py \
  tests/verification/test_python_dynamic_benchmarks.py \
  tests/integration/test_gravity_prestress.py::test_gravity_prestress_adds_load_and_reduces_first_mode
```

## Case Inventory

### Orchard FEM Analytical Benchmarks

- File: `tests/verification/test_python_beam_benchmarks.py`
- Coverage:
  - cantilever first bending mode against the Euler-Bernoulli analytical reference,
  - cantilever modal convergence under mesh refinement,
  - simply supported beam midspan deflection under a concentrated load.
- Purpose: keep the core beam formulation anchored to analytical references.

### Orchard FEM Dynamic Benchmarks

- File: `tests/verification/test_python_dynamic_benchmarks.py`
- Coverage:
  - Duffing hardening peak shift against the backbone estimate,
  - hinged two-bar first mode against the rigid-link spring-mass asymptote.
- Purpose: protect nonlinear and reduced-joint behavior with compact engineering references.

### Orchard FEM Integration Checks

- Files under `tests/integration/`
- Coverage:
  - loaders
  - CLI surface
  - cross-section defaults
  - auto nonlinear injection
  - gravity-prestress workflows
- Purpose: keep repository-level workflows stable as package APIs evolve.

## Scope Notes

- The analytical beam benchmarks should remain the primary regression shield for element assembly changes.
- The Duffing case is a nonlinear engineering benchmark rather than a closed-form exact solution. It is intentionally used as a practical guardrail for hardening response until harmonic-balance verification is added.
- The hinged two-bar case is currently an asymptotic benchmark, not a full closed-form continuous jointed-beam solution. It is still useful because it will catch regressions in rotational-spring assembly before the dedicated orchard joint model is upgraded.

## Maintenance Guidance

- If Python solver assembly, dynamics, or verification utilities change, rerun `python -m orchard_fem verify`.
- If PETSc/SLEPc-backed solver behavior changes, rerun the targeted `orchard-fenicsx` command above or use `python -m orchard_fem full-validate`.
- If a verification case starts failing after a modeling upgrade, prefer updating the model assumptions and documented reference together rather than silently loosening tolerances.
- Any new reduction or nonlinear-frequency capability should add its own verification case here instead of only adding smoke tests.
