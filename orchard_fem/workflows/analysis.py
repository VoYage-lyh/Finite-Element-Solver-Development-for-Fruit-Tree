from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from orchard_fem.io.model_loader import load_orchard_model
from orchard_fem.model import AnalysisMode
from orchard_fem.solvers.frequency_response import (
    FrequencyResponseRequest,
    PETScFrequencyResponseSolver,
)
from orchard_fem.solvers.modal import ModalAnalysisRequest, SLEPcModalSolver
from orchard_fem.solvers.modal_assembler import OrchardModalAssembler
from orchard_fem.solvers.time_history import PETScTimeHistorySolver, TimeHistoryRequest


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
) -> AnalysisRunOutputs:
    model = load_orchard_model(str(model_json))
    resolved_output = resolve_output_path(
        output_csv,
        default_solver_output(model.analysis.output_csv),
    )

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

    return AnalysisRunOutputs(mode=model.analysis.mode, output_csv=resolved_output)


def write_modal_summary(model_path: Path, output_csv: Path, num_modes: int) -> Path:
    model = load_orchard_model(str(model_path))
    assembled = OrchardModalAssembler().assemble(model)
    modes = SLEPcModalSolver().solve(
        ModalAnalysisRequest(
            num_modes=num_modes,
            stiffness_matrix=assembled.stiffness_matrix,
            mass_matrix=assembled.mass_matrix,
            dof_labels=assembled.dof_labels,
        )
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
                    "slepc",
                ]
            )
    return output_csv
