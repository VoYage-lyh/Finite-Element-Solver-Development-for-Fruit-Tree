# Legacy Reference

This document records the historical C++ parts of the repository.

They remain available only as archival material during the Python-first migration.
They are not the default development path, not the default validation path, and not the authoritative correctness oracle for new work.

## Archived directories

- `apps/`
- `include/`
- `src/`
- `CMakeLists.txt`
- `tests/verification/*.cpp`

## What they are still useful for

- reading the earlier project history,
- understanding how the original orchard-specific solver was structured,
- recovering old implementation details during migration,
- reproducing historical command-line runs when explicitly needed.

## What they are not used for anymore

- day-to-day solver runs,
- dependency checks,
- default validation,
- acceptance criteria for Python-side modeling changes.

The active project surface is `orchard_fem` plus the Python verification and CLI workflow.

## If you intentionally inspect the archived C++ path

Only do this when you explicitly need historical context.

Typical legacy-only commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
ctest -L verification
```

## Historical comparison helper

The repository still includes:

```bash
python -m orchard_fem legacy-compare --help
```

Treat that script as an archival diagnostic helper only.
It can show how a historical C++ output differs from the Python path, but it does not define correctness for the current project direction.

The old script wrapper is still present too:

```bash
python scripts/benchmark_vs_existing.py --help
```
