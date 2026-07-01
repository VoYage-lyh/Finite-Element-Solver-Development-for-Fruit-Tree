from __future__ import annotations

from pathlib import Path
from typing import Sequence

from orchard_fem.automation import FullValidationConfig, FullValidationOutputs, run_full_validation
from orchard_fem.domain import SolverBackendKind
from orchard_fem.environment import run_environment_audit
from orchard_fem.postprocess import plot_frequency_response_csv, plot_time_history_csv
from orchard_fem.visualization.scene3d import plot_tree_3d
from orchard_fem.visualization import VisualizationOutputs, visualize_analysis
from orchard_fem.workflows import (
    AnalysisRunOutputs,
    DEFAULT_VALIDATION_OUTPUT_DIR,
    ValidationOutputs,
    default_modal_output,
    resolve_output_path,
    run_configured_analysis,
    run_validation_suite,
    write_modal_summary,
)


class OrchardApplication:
    """High-level Orchard FEM orchestration facade for solver workflows."""

    def run_analysis(
        self,
        model_json: Path,
        output_csv: Path | None = None,
        solver_backend: SolverBackendKind | None = None,
    ) -> AnalysisRunOutputs:
        return run_configured_analysis(
            model_json=model_json,
            output_csv=output_csv,
            solver_backend=solver_backend,
        )

    def run_modal_summary(
        self,
        model_json: Path,
        output_csv: Path | None = None,
        num_modes: int = 6,
        solver_backend: SolverBackendKind | None = None,
    ) -> Path:
        resolved_output = resolve_output_path(
            output_csv,
            default_modal_output(model_json),
        )
        return write_modal_summary(
            model_json,
            resolved_output,
            num_modes,
            solver_backend=solver_backend,
        )

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

    def verify(
        self,
        include_integration: bool = True,
        include_verification: bool = True,
        include_dolfinx_tests: bool = False,
        output_dir: Path = DEFAULT_VALIDATION_OUTPUT_DIR,
        pytest_args: Sequence[str] | None = None,
    ) -> ValidationOutputs:
        return run_validation_suite(
            include_integration=include_integration,
            include_verification=include_verification,
            include_dolfinx_tests=include_dolfinx_tests,
            output_dir=output_dir,
            pytest_args=pytest_args,
        )

    def doctor(self) -> int:
        return run_environment_audit()

    def plot_frequency_response(self, response_csv: Path, show: bool = True) -> None:
        plot_frequency_response_csv(response_csv, show=show)

    def plot_time_history(
        self,
        response_csv: Path,
        show: bool = True,
        output_path: Path | None = None,
    ) -> None:
        plot_time_history_csv(response_csv, show=show, output_path=output_path)

    def view_tree(
        self,
        model_json: Path,
        show: bool = True,
        output_path: Path | None = None,
    ) -> None:
        import json

        with open(model_json, encoding="utf-8") as fh:
            model_data = json.load(fh)
        plot_tree_3d(model_data, show=show, output_path=output_path)

    def full_validate(self, config: FullValidationConfig) -> FullValidationOutputs:
        return run_full_validation(config)

    # ────────────────────────────────────────────────────────────────────
    #  Bridges for the `recommend` command (Bayesian → Pareto → Sobol).
    #  These wire the abstract analysis modules to the actual FEM solver.
    #  Default implementations are stubs that raise — projects that ship
    #  a custom FEM backend override these in a subclass.
    # ────────────────────────────────────────────────────────────────────
    def build_recommend_forward_operator(self, model, frf_frequencies_hz):
        """Return ``params_dict -> ForwardResult`` for Bayesian calibration.

        Default implementation uses
        :func:`orchard_fem.calibration.fenicsx_bridge.build_fenicsx_forward_operator`,
        averaging over every observation whose ID contains "target". To use a
        different aggregation, override this method or call the bridge
        directly with explicit ``target_observation_ids``.
        """
        from orchard_fem.calibration.fenicsx_bridge import (
            build_fenicsx_forward_operator,
        )
        target_ids = self._resolve_target_observation_ids(model)
        return build_fenicsx_forward_operator(
            model, frf_frequencies_hz,
            target_observation_ids=target_ids,
        )

    def build_pareto_evaluator(self, model, forward_operator):
        """Return ``(params, f, A, clamp) -> HarvestObjective``.

        Default implementation delegates to
        :func:`orchard_fem.calibration.fenicsx_bridge.build_fenicsx_pareto_evaluator`,
        which auto-augments the model's observations with one fruit
        observation per :attr:`OrchardModel.fruits` entry plus two trunk
        rotation observations for stress computation. No observation-ID
        naming convention is required.
        """
        from orchard_fem.calibration.fenicsx_bridge import (
            build_fenicsx_pareto_evaluator,
        )
        return build_fenicsx_pareto_evaluator(model)

    def build_sobol_inputs(self, model, priors):
        """Return the list of SobolInputDef matching the calibrated params + geometry."""
        from orchard_fem.sensitivity import SobolInputDef

        inputs = []
        for p in priors:
            log_scale = p.kind == "loguniform"
            inputs.append(SobolInputDef(
                name=p.name, bounds=p.bounds, log_scale=log_scale,
            ))
        return inputs

    def build_sobol_forward(self, model, evaluator, f_grid, A_grid, *,
                             clamp_label: str, constraints: dict):
        """Return ``params -> recommended_freq`` for Sobol sensitivity.

        Default implementation: for each Saltelli sample, build a one-sample
        posterior set, propagate to Pareto, take the median knee frequency.
        """
        from orchard_fem.recommendation import propagate_posterior_to_pareto

        def sobol_forward(params: dict) -> float:
            try:
                rec = propagate_posterior_to_pareto(
                    [params], clamp_label, f_grid, A_grid, evaluator,
                    credible_alpha=0.50,
                    coverage_min=constraints.get("coverage_min"),
                    stress_max=constraints.get("stress_max"),
                )
                return rec.frequency_hz_median
            except Exception:
                return float("nan")
        return sobol_forward
