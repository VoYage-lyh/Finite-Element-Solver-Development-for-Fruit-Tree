from __future__ import annotations

from pathlib import Path

from orchard_fem.visualization import VisualizationOutputs
from orchard_fem.workflows.demo import run_standard_demo_suite


def test_demo_suite_uses_configured_analysis_workflow(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_run_configured_analysis(model_json: Path, *, output_csv: Path, **kwargs):
        del kwargs
        calls.append((model_json.name, output_csv.name))
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_csv.write_text("frequency_hz,response\n1.0,0.0\n", encoding="utf-8")

    def fake_write_modal_summary(model_path: Path, output_csv: Path, num_modes: int):
        del model_path, num_modes
        output_csv.write_text(
            "mode_index,frequency_hz,eigenvalue,modal_mass,backend\n",
            encoding="utf-8",
        )
        return output_csv

    def fake_visualize_analysis(
        model_json: Path,
        response_csv: Path,
        output_prefix: Path,
        **kwargs,
    ) -> VisualizationOutputs:
        del model_json, response_csv, kwargs
        geometry = Path(f"{output_prefix}_geometry.png")
        analysis = Path(f"{output_prefix}_analysis.png")
        trajectory = Path(f"{output_prefix}_trajectory_tip.png")
        for path in (geometry, analysis, trajectory):
            path.write_text("fake figure", encoding="utf-8")
        return VisualizationOutputs(
            geometry_figure=geometry,
            analysis_figure=analysis,
            trajectory_figures=[trajectory],
        )

    monkeypatch.setattr(
        "orchard_fem.workflows.demo.run_configured_analysis",
        fake_run_configured_analysis,
    )
    monkeypatch.setattr(
        "orchard_fem.workflows.demo.write_modal_summary",
        fake_write_modal_summary,
    )
    monkeypatch.setattr(
        "orchard_fem.workflows.demo.visualize_analysis",
        fake_visualize_analysis,
    )

    outputs = run_standard_demo_suite(tmp_path)

    assert calls == [
        ("demo_orchard.json", "python_demo_frequency_response.csv"),
        ("demo_orchard_time_history.json", "python_demo_time_history.csv"),
    ]
    assert outputs.frequency_response_csv.exists()
    assert outputs.time_history_csv.exists()
    assert outputs.modal_summary_csv.exists()
    assert outputs.frequency_visualization.geometry_figure.exists()
    assert outputs.time_history_visualization.trajectory_figures[0].exists()
