# C++ Migration Inventory

This ledger tracks how the archived C++ repository surface maps into Orchard FEM.

## Current rule

- `orchard_fem/` is the active implementation surface.
- `apps/`, `include/`, `src/`, `tests/*.cpp`, and `CMakeLists.txt` are now archival.
- The archived C++ headers and sources have been removed from the main branch.
- `CMakeLists.txt` now serves only as an archive marker and points users to Orchard FEM.

## Module map

| Archived C++ surface | Orchard FEM replacement | Status |
| --- | --- | --- |
| `apps/orchard_cli.cpp` | `orchard_fem.cli`, `orchard_fem.application`, `orchard_fem.commands.run`, `orchard_fem.commands.modal` | Removed from main branch |
| `include/orchard_solver/OrchardModel.h`, `src/OrchardModel.cpp` | `orchard_fem.domain.entities`, `orchard_fem.domain.enums` | Removed from main branch |
| `include/orchard_solver/branches/BranchModel.h`, `src/branches/BranchModel.cpp` | `orchard_fem.branches`, `orchard_fem.materials.base` | Removed from main branch |
| `include/orchard_solver/cross_section/CrossSection.h`, `src/cross_section/CrossSection.cpp` | `orchard_fem.cross_section.*` | Removed from main branch |
| `include/orchard_solver/discretization/BeamElement.h`, `src/discretization/BeamElement.cpp` | `orchard_fem.discretization.beam.*` | Removed from main branch |
| `include/orchard_solver/discretization/Assembler.h`, `src/discretization/Assembler.cpp` | `orchard_fem.discretization.system`, `orchard_fem.discretization.types` | Removed from main branch |
| `include/orchard_solver/excitation_and_bc/Excitation.h` | `orchard_fem.excitation_and_bc`, `orchard_fem.dynamics.excitation` | Removed from main branch |
| `include/orchard_solver/fruits/Fruit.h` | `orchard_fem.fruits`, `orchard_fem.domain.entities.FruitAttachment` | Removed from main branch |
| `include/orchard_solver/geometry_topology/TreeTopology.h`, `src/geometry_topology/TreeTopology.cpp` | `orchard_fem.geometry_topology`, `orchard_fem.topology.*` | Removed from main branch |
| `include/orchard_solver/io/ModelIO.h`, `src/io/ModelIO.cpp` | `orchard_fem.io.loaders.*` | Removed from main branch |
| `include/orchard_solver/io/SimpleJson.h`, `src/io/SimpleJson.cpp` | `orchard_fem.io.loaders.payload` | Removed from main branch |
| `include/orchard_solver/joints_and_bifurcations/Joints.h`, `src/joints_and_bifurcations/Joints.cpp` | `orchard_fem.joints_and_bifurcations`, `orchard_fem.dynamics.nonlinear` | Removed from main branch |
| `include/orchard_solver/materials/Materials.h`, `src/materials/Materials.cpp` | `orchard_fem.materials.base`, `orchard_fem.materials` | Removed from main branch |
| `include/orchard_solver/model_reduction/ReductionStrategy.h` | `orchard_fem.model_reduction` | Removed from main branch |
| `include/orchard_solver/solver_core/LinearAlgebra.h`, `src/solver_core/LinearAlgebra.cpp` | `orchard_fem.numerics`, `orchard_fem.discretization.matrix_ops` | Removed from main branch |
| `include/orchard_solver/solver_core/DynamicSystem.h`, `src/solver_core/DynamicSystem.cpp` | `orchard_fem.solver_core`, `orchard_fem.dynamics` | Removed from main branch |
| `include/orchard_solver/solver_core/StaticPreload.h`, `src/solver_core/StaticPreload.cpp` | `orchard_fem.solver_core.static_preload` | Removed from main branch |
| `include/orchard_solver/validation/ErrorMetrics.h` | `orchard_fem.error_metrics` | Removed from main branch |
| `tests/orchard_tests.cpp` | Python integration tests under `tests/integration/` | Removed from main branch |
| `tests/verification/*.cpp` | `tests/verification/test_python_beam_benchmarks.py`, `tests/verification/test_python_dynamic_benchmarks.py`, `tests/integration/test_gravity_prestress.py` | Removed from main branch |

## Decommission batches

### Batch 1: retired from the main branch

- `apps/orchard_cli.cpp`
- `tests/orchard_tests.cpp`
- `tests/verification/*.cpp`
- `tests/verification/common.h`
- `CMakeLists.txt` default target graph

### Batch 2: retired from the main branch

- `include/`
- `src/`

Only archive-marker READMEs remain in those directories on the main branch.
The historical headers and sources now live only in git history.

## Historical closeout criteria

These conditions justified removal of the archived C++ implementation from the main branch:

1. Orchard FEM Python tests cover every benchmark previously carried by `tests/verification/*.cpp`.
2. No active Orchard FEM package module imports or shells out to archived C++ targets.
3. The default user workflow is fully documented through `python -m orchard_fem ...`.
4. The team no longer needs the historical comparison helper on the main branch.
