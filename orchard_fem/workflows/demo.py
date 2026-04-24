from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orchard_fem.dynamics import (
    FrequencyResponseRequest,
    PETScFrequencyResponseSolver,
    PETScTimeHistorySolver,
    TimeHistoryRequest,
)
from orchard_fem.workflows.analysis import write_modal_summary


@dataclass(frozen=True)
class DemoSuiteOutputs:
    frequency_response_csv: Path
    time_history_csv: Path
    modal_summary_csv: Path


def run_standard_demo_suite(
    output_dir: Path,
    frequency_model: Path = Path("examples/demo_orchard.json"),
    time_model: Path = Path("examples/demo_orchard_time_history.json"),
    num_modes: int = 6,
) -> DemoSuiteOutputs:
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    frequency_output = resolved_output_dir / "python_demo_frequency_response.csv"
    time_output = resolved_output_dir / "python_demo_time_history.csv"
    modal_output = resolved_output_dir / "python_demo_modal_summary.csv"

    frequency_result = PETScFrequencyResponseSolver().solve(
        FrequencyResponseRequest(
            model_path=str(frequency_model.resolve()),
            output_csv=str(frequency_output),
        )
    )
    if not frequency_result.points:
        raise RuntimeError("Frequency-response demo produced no output points")

    time_result = PETScTimeHistorySolver().solve(
        TimeHistoryRequest(
            model_path=str(time_model.resolve()),
            output_csv=str(time_output),
        )
    )
    if not time_result.points:
        raise RuntimeError("Time-history demo produced no output points")

    write_modal_summary(frequency_model.resolve(), modal_output, num_modes)
    return DemoSuiteOutputs(
        frequency_response_csv=frequency_output,
        time_history_csv=time_output,
        modal_summary_csv=modal_output,
    )
