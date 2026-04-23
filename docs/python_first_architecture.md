# Python-First Architecture

This document records the active project direction.

The repository now treats `orchard_fem` as the primary development surface:

- model loading and schema compatibility,
- cross-section and material processing,
- beam assembly,
- modal / frequency-response / time-history solving,
- visualization,
- verification and demo regeneration.

The active Python orchestration now has two explicit layers:

- `orchard_fem/commands/`: modular CLI command registration and argument wiring,
- `orchard_fem/application.py`: top-level application facade for CLI and external callers,
- `orchard_fem/workflows/`: reusable workflow orchestration for run, demo, and validation flows.
- `orchard_fem/visualization/`: package-native geometry, response, and trajectory rendering layer.

The multi-environment repository automation now also lives inside the package:

- `orchard_fem/automation/`: package-native orchestration for full validation across `orchard-dev` and `orchard-fenicsx`.
- `orchard_fem/postprocess/`: package-native postprocessing helpers such as frequency-response plotting.
- `orchard_fem/legacy/`: archived C++ comparison helpers kept only for historical diagnostics.

## Active project roots

These directories are the active Python-first workflow:

- `orchard_fem/`
- `orchard_pinn/`
- `examples/`
- `scripts/`
- `tests/`
- `config/`

## Legacy project roots

These directories remain in the repository only as historical reference material during migration:

- `apps/`
- `include/`
- `src/`
- `CMakeLists.txt`

They are no longer the default entry points for:

- running demos,
- validating changes,
- checking dependencies,
- or onboarding new work.

## Primary commands

Daily solver usage:

```bash
python -m orchard_fem run examples/demo_orchard.json
python -m orchard_fem modal examples/demo_orchard.json
python -m orchard_fem visualize examples/demo_orchard.json build/demo_frequency_response.csv
```

Daily validation:

```bash
python -m orchard_fem verify
```

Environment check:

```bash
python -m orchard_fem doctor
```

## Validation strategy

Correctness is now established by Python-side guardrails:

- integration tests,
- analytical beam benchmarks,
- nonlinear dynamic benchmarks,
- PETSc/SLEPc gravity-prestress regression,
- stable demo artifact generation.

Legacy C++ output comparison is no longer part of the standard workflow.
If needed, `scripts/benchmark_vs_existing.py` can still be used as an archival helper.
Use [legacy_reference.md](legacy_reference.md) for the remaining historical C++ entry points.

## Migration priorities

The remaining migration work should continue in this order:

1. keep adding Python-side analytical and engineering verification cases,
2. keep moving reusable workflow logic from ad-hoc scripts into `orchard_fem`,
3. keep downgrading legacy C++ files from active documentation and tooling,
4. only retain legacy C++ code as historical reference until the Python stack fully covers the needed behavior.
