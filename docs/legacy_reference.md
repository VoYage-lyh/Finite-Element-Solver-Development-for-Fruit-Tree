# Legacy Reference

This document records the historical C++ parts of the repository.

They now remain available only as archival markers and documentation alongside the active Orchard FEM package.
They are not the default development path, not the default validation path, and not the authoritative correctness oracle for new work.

## Archived directories

- `apps/`
- `include/`
- `src/`
- `CMakeLists.txt`

The detailed migration ledger now lives in [cpp_migration_inventory.md](cpp_migration_inventory.md).

The following historical files have already been removed from the main branch and now live only in git history:

- `apps/orchard_cli.cpp`
- `include/orchard_solver/**/*.h`
- `src/**/*.cpp`
- `tests/orchard_tests.cpp`
- `tests/verification/*.cpp`
- `tests/verification/common.h`

## What they are still useful for

- reading the earlier project history,
- understanding how the original orchard-specific solver was structured,
- recovering old implementation details during migration,
- reproducing historical command-line runs when explicitly needed.

## What they are not used for anymore

- day-to-day solver runs,
- dependency checks,
- default validation,
- acceptance criteria for active Orchard FEM modeling changes.

The active project surface is `orchard_fem` plus the package verification and CLI workflow.

## If you intentionally inspect the archived C++ path

Only do this when you explicitly need historical context.

The current branch keeps only archive markers, so trying to enable legacy C++ in CMake now fails on purpose and points you back to history.
If you need the old build, switch to an older git revision first.

Typical legacy-only commands from such a historical checkout:

```bash
cmake -S . -B build -DORCHARD_ENABLE_LEGACY_CPP=ON
cmake --build build -j
```

## Historical comparison helper

The main branch no longer includes the old comparison helper.
If you need to reproduce that historical workflow, use an older git revision that still contains the removed `legacy-compare` command and `scripts/benchmark_vs_existing.py`.
That comparison path is no longer part of the active Orchard FEM project surface.
