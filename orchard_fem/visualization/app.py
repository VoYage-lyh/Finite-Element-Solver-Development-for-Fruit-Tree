from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from orchard_fem.visualization.dependencies import MissingDependencyError, require_plotting_dependencies
from orchard_fem.visualization.io import (
    build_parser,
    frequency_figure_path,
    geometry_figure_path,
    load_model,
    load_response,
    parse_args,
    time_frequency_figure_path,
    trajectory_figure_path,
)
from orchard_fem.visualization.rendering import (
    available_trajectory_nodes,
    plot_frequency_response,
    plot_geometry,
    plot_time_frequency,
    plot_trajectory,
)


@dataclass(frozen=True)
class VisualizationOutputs:
    geometry_figure: Path
    analysis_figure: Path
    trajectory_figures: list[Path]


def visualize_analysis(
    model_json: Path,
    response_csv: Path,
    output_prefix: Path | None = None,
    measurement_column: str | None = None,
    trajectory_nodes: Sequence[str] | None = None,
    show: bool = False,
) -> VisualizationOutputs:
    if not model_json.exists():
        raise FileNotFoundError("Model JSON not found: {0}".format(model_json))
    if not response_csv.exists():
        raise FileNotFoundError("Response CSV not found: {0}".format(response_csv))

    require_plotting_dependencies(show=show)

    resolved_prefix = output_prefix if output_prefix is not None else response_csv.with_suffix("")
    model = load_model(model_json)
    headers, rows = load_response(response_csv)

    geometry_path = geometry_figure_path(resolved_prefix)
    plot_geometry(model, geometry_path, show)

    trajectory_paths: list[Path] = []
    first_column = headers[0]
    if first_column == "time_s":
        analysis_path = time_frequency_figure_path(resolved_prefix)
        plot_time_frequency(headers, rows, analysis_path, measurement_column, show)
        node_ids = list(trajectory_nodes) if trajectory_nodes is not None else available_trajectory_nodes(headers)
        for node_id in node_ids:
            trajectory_path = trajectory_figure_path(resolved_prefix, node_id)
            plot_trajectory(headers, rows, trajectory_path, node_id, show)
            trajectory_paths.append(trajectory_path)
    elif first_column == "frequency_hz":
        analysis_path = frequency_figure_path(resolved_prefix)
        plot_frequency_response(headers, rows, analysis_path, show)
    else:
        raise RuntimeError("Unsupported response CSV first column: {0}".format(first_column))

    return VisualizationOutputs(
        geometry_figure=geometry_path,
        analysis_figure=analysis_path,
        trajectory_figures=trajectory_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = visualize_analysis(
            model_json=args.model_json,
            response_csv=args.response_csv,
            output_prefix=args.output_prefix,
            measurement_column=args.measurement_column,
            trajectory_nodes=args.trajectory_node,
            show=args.show,
        )
    except MissingDependencyError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("Saved geometry figure to {0}".format(outputs.geometry_figure))
    print("Saved analysis figure to {0}".format(outputs.analysis_figure))
    for trajectory_figure in outputs.trajectory_figures:
        print("Saved trajectory figure to {0}".format(trajectory_figure))
    return 0
