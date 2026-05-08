from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orchard_fem.visualization import VisualizationOutputs, visualize_analysis
from orchard_fem.workflows.analysis import (
    run_configured_analysis,
    write_modal_summary,
)


@dataclass(frozen=True)
class DemoSuiteOutputs:
    frequency_response_csv: Path
    time_history_csv: Path
    modal_summary_csv: Path
    frequency_visualization: VisualizationOutputs
    time_history_visualization: VisualizationOutputs


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

    run_configured_analysis(frequency_model.resolve(), output_csv=frequency_output)
    if not frequency_output.exists() or frequency_output.stat().st_size <= 0:
        raise RuntimeError("Frequency-response demo produced no output points")

    run_configured_analysis(time_model.resolve(), output_csv=time_output)
    if not time_output.exists() or time_output.stat().st_size <= 0:
        raise RuntimeError("Time-history demo produced no output points")

    write_modal_summary(frequency_model.resolve(), modal_output, num_modes)
    frequency_visualization = visualize_analysis(
        model_json=frequency_model.resolve(),
        response_csv=frequency_output,
        output_prefix=resolved_output_dir / "python_demo_frequency_response",
        show=False,
    )
    time_history_visualization = visualize_analysis(
        model_json=time_model.resolve(),
        response_csv=time_output,
        output_prefix=resolved_output_dir / "python_demo_time_history",
        show=False,
    )
    return DemoSuiteOutputs(
        frequency_response_csv=frequency_output,
        time_history_csv=time_output,
        modal_summary_csv=modal_output,
        frequency_visualization=frequency_visualization,
        time_history_visualization=time_history_visualization,
    )
