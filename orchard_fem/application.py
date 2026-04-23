from __future__ import annotations

from pathlib import Path
from typing import Sequence

from orchard_fem.automation import FullValidationConfig, FullValidationOutputs, run_full_validation
from orchard_fem.environment import run_environment_audit
from orchard_fem.legacy import LegacyCompareRequest, run_legacy_compare
from orchard_fem.postprocess import plot_frequency_response_csv
from orchard_fem.validation import ValidationOutputs
from orchard_fem.visualization import VisualizationOutputs, visualize_analysis
from orchard_fem.workflows import (
    AnalysisRunOutputs,
    DemoSuiteOutputs,
    default_modal_output,
    resolve_output_path,
    run_configured_analysis,
    run_standard_demo_suite,
    run_validation_suite,
    write_modal_summary,
)


class OrchardApplication:
    """High-level Python-first orchestration facade for solver workflows."""

    def run_analysis(
        self,
        model_json: Path,
        output_csv: Path | None = None,
    ) -> AnalysisRunOutputs:
        return run_configured_analysis(model_json=model_json, output_csv=output_csv)

    def run_modal_summary(
        self,
        model_json: Path,
        output_csv: Path | None = None,
        num_modes: int = 6,
    ) -> Path:
        resolved_output = resolve_output_path(
            output_csv,
            default_modal_output(model_json),
        )
        return write_modal_summary(model_json, resolved_output, num_modes)

    def visualize(
        self,
        model_json: Path,
        response_csv: Path,
        output_prefix: Path | None = None,
        measurement_column: str | None = None,
        trajectory_nodes: Sequence[str] | None = None,
        show: bool = False,
    ) -> VisualizationOutputs:
        return visualize_analysis(
            model_json=model_json,
            response_csv=response_csv,
            output_prefix=output_prefix,
            measurement_column=measurement_column,
            trajectory_nodes=trajectory_nodes,
            show=show,
        )

    def run_demo_suite(
        self,
        output_dir: Path,
        frequency_model: Path = Path("examples/demo_orchard.json"),
        time_model: Path = Path("examples/demo_orchard_time_history.json"),
        num_modes: int = 6,
    ) -> DemoSuiteOutputs:
        return run_standard_demo_suite(
            output_dir=output_dir,
            frequency_model=frequency_model,
            time_model=time_model,
            num_modes=num_modes,
        )

    def verify(
        self,
        include_integration: bool = True,
        include_verification: bool = True,
        include_demo_suite: bool = True,
        output_dir: Path = Path("build/validation/python"),
        pytest_args: Sequence[str] | None = None,
    ) -> ValidationOutputs:
        return run_validation_suite(
            include_integration=include_integration,
            include_verification=include_verification,
            include_demo_suite=include_demo_suite,
            output_dir=output_dir,
            pytest_args=pytest_args,
        )

    def doctor(self) -> int:
        return run_environment_audit()

    def plot_frequency_response(self, response_csv: Path, show: bool = True) -> None:
        plot_frequency_response_csv(response_csv, show=show)

    def legacy_compare(self, request: LegacyCompareRequest) -> None:
        run_legacy_compare(request)

    def full_validate(self, config: FullValidationConfig) -> FullValidationOutputs:
        return run_full_validation(config)
