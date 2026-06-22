# -*- coding: utf-8 -*-
"""Parallel work-point FE solves via a spawn process pool.

Every ``(clamp, frequency, amplitude)`` :class:`HarvestOutcome` solve is
independent, so they fan out across CPU cores.  dolfinx / PETSc need a fresh
interpreter per worker (a ``fork`` after PETSc init corrupts its state), so we
use the ``spawn`` start method; each worker builds the outcome solver once in
the pool initializer and is then reused across many tasks, amortising the
dolfinx import.

Used by both :func:`orchard_fem.workflows.harvest_recommendation` and
:func:`orchard_fem.workflows.harvest_schedule.build_branch_outcome_grid`.
"""
from __future__ import annotations

import os
from typing import Any, Callable

# Per-worker outcome solver, initialised once in `_init` (one dolfinx import
# per worker process, reused for all its tasks).
_SOLVE: Any = None


def _init(model: Any, solver_kw: dict) -> None:
    # keep each solve single-threaded so N workers don't oversubscribe cores
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    global _SOLVE
    from orchard_fem.calibration.fenicsx_bridge import build_outcome_solver
    _SOLVE = build_outcome_solver(model, **solver_kw)


def _task(args: tuple) -> tuple:
    params, frequency_hz, amplitude, clamp_label = args
    try:
        outcome = _SOLVE(params, frequency_hz, amplitude, clamp_label)
        return (clamp_label, frequency_hz, amplitude, outcome, None)
    except Exception as exc:  # noqa: BLE001 - shipped back to the parent
        return (clamp_label, frequency_hz, amplitude, None, repr(exc))


def resolve_n_jobs(n_jobs: int | None, n_tasks: int) -> int:
    """Clamp a requested worker count to ``[1, min(cpu, n_tasks)]``.

    ``n_jobs <= 0`` or ``None`` means "all cores".
    """
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1
    return max(1, min(int(n_jobs), max(n_tasks, 1)))


def solve_outcomes_parallel(
    model: Any,
    solver_kw: dict,
    work_points: list[tuple[str, float, float]],
    params: dict,
    n_jobs: int,
    *,
    on_done: Callable[[str, float, float], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> dict[tuple[str, float, float], Any]:
    """Solve every ``(clamp, f, A)`` in *work_points* across *n_jobs* processes.

    Returns ``{(clamp, f, A): HarvestOutcome}``.  ``model`` must already be
    displacement-driven (the solver only reroutes clamp/frequency/amplitude).
    Raises ``RuntimeError`` on the first solve failure or on cancellation.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    tasks = [(params, f, a, clamp) for (clamp, f, a) in work_points]
    outcomes: dict[tuple[str, float, float], Any] = {}
    # NB: don't use `with Pool(...)` — its __exit__ calls terminate() (SIGTERM),
    # which makes the workers' PETSc/MPI abort loudly mid-finalize.  On success we
    # close()+join() so workers drain and exit cleanly (quiet); terminate() is
    # reserved for the error / cancel paths.
    pool = ctx.Pool(processes=n_jobs, initializer=_init, initargs=(model, solver_kw))
    try:
        for clamp, f, a, outcome, err in pool.imap_unordered(_task, tasks):
            if err is not None:
                pool.terminate()
                raise RuntimeError(f"FE solve failed at ({clamp}, f={f:g}, A={a:g}): {err}")
            outcomes[(clamp, f, a)] = outcome
            if on_done is not None:
                on_done(clamp, f, a)
            if cancel_cb is not None and cancel_cb():
                pool.terminate()
                raise RuntimeError("cancelled")
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return outcomes
