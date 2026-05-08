from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from orchard_fem.discretization import OrchardModalAssembler
from orchard_fem.domain import AnalysisMode, SolverBackendKind
from orchard_fem.dynamics import (
    FrequencyResponseRequest,
    PETScFrequencyResponseSolver,
    PETScTimeHistorySolver,
    TimeHistoryRequest,
)
from orchard_fem.io import load_orchard_model
from orchard_fem.solver_core import ModalAnalysisRequest, SLEPcModalSolver


@dataclass(frozen=True)
class AnalysisRunOutputs:
    mode: AnalysisMode
    output_csv: Path


def default_solver_output(model_output_csv: str) -> Path:
    configured_output = Path(model_output_csv)
    if configured_output.is_absolute():
        return configured_output
    return Path("build") / configured_output


def default_modal_output(model_path: Path) -> Path:
    return Path("build") / f"{model_path.stem}_modal_summary.csv"


def resolve_output_path(path: Path | None, fallback: Path) -> Path:
    resolved = path if path is not None else fallback
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def run_configured_analysis(
    model_json: Path,
    output_csv: Path | None = None,
    solver_backend: SolverBackendKind | None = None,
) -> AnalysisRunOutputs:
    model = load_orchard_model(str(model_json))
    if solver_backend is not None:
        model = replace(
            model,
            analysis=replace(model.analysis, solver_backend=solver_backend),
        )
    resolved_output = resolve_output_path(
        output_csv,
        default_solver_output(model.analysis.output_csv),
    )

    if model.analysis.solver_backend == SolverBackendKind.FENICSX:
        from orchard_fem.fenicsx import (
            solve_embedded_beam_frequency_response_experiment,
            solve_embedded_beam_time_history_experiment,
        )

        if model.analysis.mode == AnalysisMode.FREQUENCY_RESPONSE:
            result = solve_embedded_beam_frequency_response_experiment(model)
            result.result.write_csv(str(resolved_output))
        elif model.analysis.mode == AnalysisMode.TIME_HISTORY:
            result = solve_embedded_beam_time_history_experiment(model)
            result.result.write_csv(str(resolved_output))
        else:
            raise RuntimeError(f"Unsupported analysis mode: {model.analysis.mode}")
    elif model.analysis.solver_backend == SolverBackendKind.NATIVE:
        if model.analysis.mode == AnalysisMode.FREQUENCY_RESPONSE:
            PETScFrequencyResponseSolver().solve(
                FrequencyResponseRequest(
                    model_path=str(model_json),
                    output_csv=str(resolved_output),
                )
            )
        elif model.analysis.mode == AnalysisMode.TIME_HISTORY:
            PETScTimeHistorySolver().solve(
                TimeHistoryRequest(
                    model_path=str(model_json),
                    output_csv=str(resolved_output),
                )
            )
        else:
            raise RuntimeError(f"Unsupported analysis mode: {model.analysis.mode}")
    else:
        raise RuntimeError(
            f"Unsupported solver backend: {model.analysis.solver_backend}"
        )

    return AnalysisRunOutputs(mode=model.analysis.mode, output_csv=resolved_output)


def write_modal_summary(
    model_path: Path,
    output_csv: Path,
    num_modes: int,
    solver_backend: SolverBackendKind | None = None,
) -> Path:
    model = load_orchard_model(str(model_path))
    if solver_backend is not None:
        model = replace(
            model,
            analysis=replace(model.analysis, solver_backend=solver_backend),
        )
    if model.analysis.solver_backend == SolverBackendKind.FENICSX:
        from orchard_fem.fenicsx import solve_embedded_beam_modal_experiment

        backend_label = "fenicsx"
        modal_result = solve_embedded_beam_modal_experiment(model, num_modes=num_modes)
        modes = modal_result.modes
    elif model.analysis.solver_backend == SolverBackendKind.NATIVE:
        backend_label = "native"
        assembled = OrchardModalAssembler().assemble(model)
        modes = SLEPcModalSolver().solve(
            ModalAnalysisRequest(
                num_modes=num_modes,
                stiffness_matrix=assembled.stiffness_matrix,
                mass_matrix=assembled.mass_matrix,
                dof_labels=assembled.dof_labels,
            )
        )
    else:
        raise RuntimeError(
            f"Unsupported solver backend: {model.analysis.solver_backend}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode_index", "frequency_hz", "eigenvalue", "modal_mass", "backend"])
        for mode in modes:
            writer.writerow(
                [
                    mode.mode_index,
                    f"{mode.frequency_hz:.12e}",
                    f"{mode.eigenvalue:.12e}",
                    f"{mode.modal_mass:.12e}",
                    backend_label,
                ]
            )
    return output_csv
