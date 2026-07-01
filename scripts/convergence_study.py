"""Mesh (h) and time-step (Δt) convergence study for the native Orchard FEM solver.

The goal is *diagnostic*: separate the discretization error (how far the current
mesh / time step is from the converged answer) from the physical-model and
calibration error. Until you know the numerical error is small, you cannot tell
whether a sim-vs-measurement gap comes from the model or just from a coarse mesh.

What it does
------------
1. **Mesh study** — re-meshes every branch at a sweep of ``num_elements`` values
   and, for each mesh, reports

       * the first ``--modes`` natural frequencies (SLEPc modal solve), and
       * a static drive-point compliance plus the static deflection at every
         observation DOF (solve ``K u = f`` for a unit load at the excitation DOF).

   Both quantities are compared against the finest mesh to give a relative error,
   which is the headline "how converged are we" number.

2. **Time-step study** — fixes the mesh (``--dt-elements``) and runs the Newmark
   time-history solver at a sweep of ``--dt`` values, reporting the peak and RMS
   response at each observation over the steady tail of the signal, again with a
   relative error versus the smallest Δt.

Outputs (under ``--out-dir``, default ``results/diagnostics/convergence/<model>``):
    convergence_mesh.csv, convergence_dt.csv, and (unless ``--no-plot``) PNG
    convergence curves.

Example
-------
    python scripts/convergence_study.py tests/fixtures/demo_orchard.json \
        --modes 6 --elements 1,2,4,8,16,32 --dt 4e-3,2e-3,1e-3,5e-4,2.5e-4
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from orchard_fem.discretization import OrchardSystemAssembler
from orchard_fem.dynamics.time_history import solve_time_history_system
from orchard_fem.io import load_orchard_model
from orchard_fem.numerics import create_aij_matrix, solve_linear_system
from orchard_fem.solver_core.modal import ModalAnalysisRequest, SLEPcModalSolver


# --------------------------------------------------------------------------- #
# Model construction with overridden mesh / time step
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ModelFactory:
    """Builds OrchardModels from a base JSON payload with mesh/Δt overrides.

    The domain dataclasses are frozen, so the robust way to override
    ``num_elements`` and the time step is to edit the payload dict and round-trip
    it through the normal loader. Temp files live in ``work_dir``.
    """

    base_payload: dict
    work_dir: Path

    def build(
        self,
        num_elements: int | None = None,
        time_step: float | None = None,
        total_time: float | None = None,
    ):
        payload = copy.deepcopy(self.base_payload)
        if num_elements is not None:
            for branch in payload["branches"]:
                branch.setdefault("discretization", {})["num_elements"] = num_elements
        if time_step is not None:
            payload["analysis"]["time_step_seconds"] = time_step
        if total_time is not None:
            payload["analysis"]["total_time_seconds"] = total_time

        tag = f"nel{num_elements}_dt{time_step}_T{total_time}".replace(".", "p")
        path = self.work_dir / f"_convergence_{tag}.json"
        path.write_text(json.dumps(payload))
        return load_orchard_model(str(path))


# --------------------------------------------------------------------------- #
# Mesh (h) study
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MeshPoint:
    num_elements: int
    dof_count: int
    frequencies_hz: list[float]
    drive_point_compliance: float
    observation_static: list[float]
    solve_seconds: float


def _static_deflection(assembled) -> tuple[float, list[float]]:
    """Drive-point compliance and static deflection at each observation DOF.

    Solves ``K u = f`` with a unit load applied at the excitation DOF. This is a
    damping- and time-independent stiffness metric: a clean way to watch
    displacement converge as the mesh refines (displacement FEM converges from
    below, i.e. coarse meshes are too stiff and under-predict deflection).
    """
    dof_count = len(assembled.dof_labels)
    load = [0.0] * dof_count
    load[assembled.excitation_dof] = 1.0
    displacement = solve_linear_system(create_aij_matrix(assembled.stiffness_matrix), load)
    drive_point = displacement[assembled.excitation_dof]
    observation = [displacement[dof] for dof in assembled.observation_dofs]
    return drive_point, observation


def run_mesh_study(
    factory: _ModelFactory,
    element_counts: list[int],
    num_modes: int,
) -> list[MeshPoint]:
    points: list[MeshPoint] = []
    for num_elements in element_counts:
        start = time.time()
        model = factory.build(num_elements=num_elements)
        assembled = OrchardSystemAssembler().assemble(model)
        dof_count = len(assembled.dof_labels)

        request = ModalAnalysisRequest(
            num_modes=min(num_modes, dof_count - 1),
            stiffness_matrix=assembled.stiffness_matrix,
            mass_matrix=assembled.mass_matrix,
            dof_labels=assembled.dof_labels,
        )
        modes = SLEPcModalSolver().solve(request)
        frequencies = [mode.frequency_hz for mode in modes]

        drive_point, observation = _static_deflection(assembled)
        elapsed = time.time() - start
        points.append(
            MeshPoint(
                num_elements=num_elements,
                dof_count=dof_count,
                frequencies_hz=frequencies,
                drive_point_compliance=drive_point,
                observation_static=observation,
                solve_seconds=elapsed,
            )
        )
        freq_text = ", ".join(f"{value:.4f}" for value in frequencies[:num_modes])
        print(
            f"  nel={num_elements:>3d}  DOF={dof_count:>5d}  "
            f"f[Hz]=[{freq_text}]  drivePtC={drive_point:.6e}  ({elapsed:.1f}s)"
        )
    return points


# --------------------------------------------------------------------------- #
# Time-step (Δt) study
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TimeStepPoint:
    time_step: float
    num_steps: int
    observation_peak: list[float]
    observation_rms: list[float]
    solve_seconds: float


def _steady_tail(values: list[float], tail_fraction: float) -> list[float]:
    if not values:
        return values
    start = int(len(values) * (1.0 - tail_fraction))
    return values[start:]


def run_timestep_study(
    factory: _ModelFactory,
    time_steps: list[float],
    fixed_elements: int,
    total_time: float | None,
    tail_fraction: float,
) -> list[TimeStepPoint]:
    points: list[TimeStepPoint] = []
    for time_step in time_steps:
        start = time.time()
        model = factory.build(
            num_elements=fixed_elements,
            time_step=time_step,
            total_time=total_time,
        )
        assembled = OrchardSystemAssembler().assemble(model)
        result = solve_time_history_system(assembled, model.excitation, model.analysis)

        num_obs = len(assembled.observation_dofs)
        peaks: list[float] = []
        rms: list[float] = []
        for obs_index in range(num_obs):
            series = [point.observation_values[obs_index] for point in result.points]
            tail = _steady_tail(series, tail_fraction)
            peaks.append(max((abs(value) for value in tail), default=0.0))
            rms.append(
                math.sqrt(sum(value * value for value in tail) / len(tail)) if tail else 0.0
            )
        elapsed = time.time() - start
        points.append(
            TimeStepPoint(
                time_step=time_step,
                num_steps=len(result.points),
                observation_peak=peaks,
                observation_rms=rms,
                solve_seconds=elapsed,
            )
        )
        max_peak = max(peaks, default=0.0)
        print(
            f"  dt={time_step:.2e}  pts={len(result.points):>5d}  "
            f"obs={num_obs}  max|peak|={max_peak:.4e}  ({elapsed:.1f}s)"
        )
    return points


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _relative_error(value: float, reference: float) -> float:
    if abs(reference) < 1.0e-30:
        return 0.0
    return abs(value - reference) / abs(reference)


def write_mesh_csv(points: list[MeshPoint], num_modes: int, observation_names: list[str], path: Path) -> None:
    finest = points[-1]
    header = (
        ["num_elements", "dof_count"]
        + [f"f{i + 1}_hz" for i in range(num_modes)]
        + [f"f{i + 1}_relerr" for i in range(num_modes)]
        + ["drive_point_compliance", "drive_point_relerr"]
        + [f"static_{name}" for name in observation_names]
        + ["solve_seconds"]
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for point in points:
            freqs = [point.frequencies_hz[i] if i < len(point.frequencies_hz) else "" for i in range(num_modes)]
            freq_err = [
                _relative_error(point.frequencies_hz[i], finest.frequencies_hz[i])
                if i < len(point.frequencies_hz) and i < len(finest.frequencies_hz)
                else ""
                for i in range(num_modes)
            ]
            writer.writerow(
                [point.num_elements, point.dof_count]
                + [f"{value:.6f}" if value != "" else "" for value in freqs]
                + [f"{value:.6e}" if value != "" else "" for value in freq_err]
                + [
                    f"{point.drive_point_compliance:.6e}",
                    f"{_relative_error(point.drive_point_compliance, finest.drive_point_compliance):.6e}",
                ]
                + [f"{value:.6e}" for value in point.observation_static]
                + [f"{point.solve_seconds:.2f}"]
            )


def write_timestep_csv(points: list[TimeStepPoint], observation_names: list[str], path: Path) -> None:
    finest = points[-1]
    header = (
        ["time_step", "num_steps"]
        + [f"peak_{name}" for name in observation_names]
        + [f"peak_{name}_relerr" for name in observation_names]
        + [f"rms_{name}" for name in observation_names]
        + ["solve_seconds"]
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for point in points:
            peak_err = [
                _relative_error(point.observation_peak[i], finest.observation_peak[i])
                for i in range(len(point.observation_peak))
            ]
            writer.writerow(
                [f"{point.time_step:.6e}", point.num_steps]
                + [f"{value:.6e}" for value in point.observation_peak]
                + [f"{value:.6e}" for value in peak_err]
                + [f"{value:.6e}" for value in point.observation_rms]
                + [f"{point.solve_seconds:.2f}"]
            )


def print_mesh_verdict(points: list[MeshPoint], num_modes: int) -> None:
    if len(points) < 2:
        return
    finest = points[-1]
    print("\n  Mesh convergence vs finest mesh "
          f"(nel={finest.num_elements}, DOF={finest.dof_count}):")
    for point in points[:-1]:
        worst = 0.0
        for i in range(min(num_modes, len(point.frequencies_hz), len(finest.frequencies_hz))):
            worst = max(worst, _relative_error(point.frequencies_hz[i], finest.frequencies_hz[i]))
        comp_err = _relative_error(point.drive_point_compliance, finest.drive_point_compliance)
        print(
            f"    nel={point.num_elements:>3d}: worst |Δf|/f = {worst * 100:6.2f}%   "
            f"|Δcompliance| = {comp_err * 100:6.2f}%"
        )


def print_timestep_verdict(
    points: list[TimeStepPoint],
    observation_names: list[str],
    significance: float = 0.05,
) -> None:
    if len(points) < 2:
        return
    finest = points[-1]
    # Only judge observations whose finest-Δt peak is a meaningful fraction of the
    # largest peak. Tiny near-zero DOFs (e.g. rotations near a clamp) otherwise
    # dominate the "worst relative error" with pure numerical noise.
    reference_max = max(finest.observation_peak, default=0.0)
    threshold = significance * reference_max
    significant = [i for i, value in enumerate(finest.observation_peak) if value >= threshold]
    print(
        f"\n  Time-step convergence vs smallest Δt = {finest.time_step:.2e} "
        f"(over {len(significant)}/{len(finest.observation_peak)} observations "
        f"with peak ≥ {significance:.0%} of max):"
    )
    for point in points[:-1]:
        worst = 0.0
        worst_name = ""
        for i in significant:
            error = _relative_error(point.observation_peak[i], finest.observation_peak[i])
            if error > worst:
                worst = error
                worst_name = observation_names[i] if i < len(observation_names) else f"obs{i}"
        print(
            f"    dt={point.time_step:.2e}: worst |Δpeak|/peak = {worst * 100:6.2f}%"
            f"  (at {worst_name})"
        )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_mesh(points: list[MeshPoint], num_modes: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finest = points[-1]
    dofs = [point.dof_count for point in points]
    fig, (ax_freq, ax_err) = plt.subplots(1, 2, figsize=(11, 4.5))

    for i in range(num_modes):
        series = [point.frequencies_hz[i] if i < len(point.frequencies_hz) else math.nan for point in points]
        ax_freq.plot(dofs, series, marker="o", label=f"mode {i + 1}")
    ax_freq.set_xscale("log")
    ax_freq.set_xlabel("DOF count")
    ax_freq.set_ylabel("natural frequency [Hz]")
    ax_freq.set_title("Mesh convergence: natural frequencies")
    ax_freq.grid(True, which="both", alpha=0.3)
    ax_freq.legend(fontsize=8)

    for i in range(num_modes):
        errors = []
        for point in points[:-1]:
            if i < len(point.frequencies_hz) and i < len(finest.frequencies_hz):
                errors.append(_relative_error(point.frequencies_hz[i], finest.frequencies_hz[i]))
            else:
                errors.append(math.nan)
        ax_err.loglog([point.dof_count for point in points[:-1]], errors, marker="o", label=f"mode {i + 1}")
    ax_err.set_xlabel("DOF count")
    ax_err.set_ylabel("relative error vs finest mesh")
    ax_err.set_title("Mesh convergence: frequency error")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_timestep(
    points: list[TimeStepPoint],
    observation_names: list[str],
    out_path: Path,
    max_curves: int = 8,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finest = points[-1]
    steps = [point.time_step for point in points]
    # With dozens of observations, plot only the highest-amplitude ones so the
    # legend stays readable; the rest are in the CSV.
    ranked = sorted(
        range(len(finest.observation_peak)),
        key=lambda i: finest.observation_peak[i],
        reverse=True,
    )
    selected = ranked[:max_curves]
    fig, (ax_peak, ax_err) = plt.subplots(1, 2, figsize=(11, 4.5))

    for obs_index in selected:
        name = observation_names[obs_index] if obs_index < len(observation_names) else f"obs{obs_index}"
        series = [point.observation_peak[obs_index] for point in points]
        ax_peak.plot(steps, series, marker="o", label=name)
    ax_peak.set_xscale("log")
    ax_peak.invert_xaxis()
    ax_peak.set_xlabel("time step Δt [s]")
    ax_peak.set_ylabel("peak response (steady tail)")
    ax_peak.set_title(f"Time-step convergence: peak response (top {len(selected)})")
    ax_peak.grid(True, which="both", alpha=0.3)
    ax_peak.legend(fontsize=8)

    for obs_index in selected:
        name = observation_names[obs_index] if obs_index < len(observation_names) else f"obs{obs_index}"
        errors = [
            _relative_error(point.observation_peak[obs_index], finest.observation_peak[obs_index])
            for point in points[:-1]
        ]
        ax_err.loglog([point.time_step for point in points[:-1]], errors, marker="o", label=name)
    ax_err.invert_xaxis()
    ax_err.set_xlabel("time step Δt [s]")
    ax_err.set_ylabel("relative error vs smallest Δt")
    ax_err.set_title("Time-step convergence: peak error")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_int_list(text: str) -> list[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(token) for token in text.split(",") if token.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_json", type=Path, nargs="?", default=Path("tests/fixtures/demo_orchard.json"))
    parser.add_argument("--modes", type=int, default=6, help="Number of natural frequencies to track.")
    parser.add_argument("--elements", type=str, default="1,2,4,8,16,32", help="Comma-separated num_elements sweep.")
    parser.add_argument("--dt", type=str, default="4e-3,2e-3,1e-3,5e-4,2.5e-4", help="Comma-separated Δt sweep.")
    parser.add_argument("--dt-elements", type=int, default=None, help="Fixed mesh for the Δt study (default: min(8, max(--elements)); the time-history solver is dense/Python so very fine meshes are slow).")
    parser.add_argument("--dt-total", type=float, default=None, help="Override total_time_seconds for the Δt study.")
    parser.add_argument("--tail-fraction", type=float, default=0.5, help="Fraction of the signal tail used for peak/RMS.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default results/diagnostics/convergence/<model>).")
    parser.add_argument("--no-dt", action="store_true", help="Skip the time-step study (mesh study only).")
    parser.add_argument("--no-mesh", action="store_true", help="Skip the mesh study (Δt study only); leaves existing mesh outputs untouched.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plots.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary re-meshed model JSONs.")
    args = parser.parse_args()

    base_payload = json.loads(args.model_json.read_text())
    element_counts = sorted(set(_parse_int_list(args.elements)))
    time_steps = sorted(set(_parse_float_list(args.dt)), reverse=True)
    dt_elements = args.dt_elements if args.dt_elements is not None else min(8, max(element_counts))

    out_dir = args.out_dir or (Path("results/diagnostics/convergence") / args.model_json.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    factory = _ModelFactory(base_payload=base_payload, work_dir=work_dir)

    # Observation names come from any assembled model (mesh-independent).
    probe = OrchardSystemAssembler().assemble(factory.build(num_elements=element_counts[0]))
    observation_names = list(probe.observation_names)

    print(f"Model: {args.model_json}  ({len(base_payload['branches'])} branches, "
          f"{len(observation_names)} observations)")
    if not args.no_mesh:
        print(f"\n[1/2] Mesh (h) study over num_elements = {element_counts}")
        mesh_points = run_mesh_study(factory, element_counts, args.modes)
        effective_modes = min(args.modes, min(len(point.frequencies_hz) for point in mesh_points))
        write_mesh_csv(mesh_points, effective_modes, observation_names, out_dir / "convergence_mesh.csv")
        print_mesh_verdict(mesh_points, effective_modes)
        if not args.no_plot:
            plot_mesh(mesh_points, effective_modes, out_dir / "convergence_mesh.png")

    if not args.no_dt:
        print(f"\n[2/2] Time-step (Δt) study at fixed nel={dt_elements} over Δt = {time_steps}")
        timestep_points = run_timestep_study(
            factory, time_steps, dt_elements, args.dt_total, args.tail_fraction
        )
        write_timestep_csv(timestep_points, observation_names, out_dir / "convergence_dt.csv")
        print_timestep_verdict(timestep_points, observation_names)
        if not args.no_plot:
            plot_timestep(timestep_points, observation_names, out_dir / "convergence_dt.png")

    if not args.keep_temp:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\nWrote results to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
