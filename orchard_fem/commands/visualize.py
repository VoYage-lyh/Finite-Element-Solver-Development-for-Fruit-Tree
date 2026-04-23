from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_visualize(args: argparse.Namespace, application: OrchardApplication) -> int:
    outputs = application.visualize(
        model_json=args.model_json,
        response_csv=args.response_csv,
        output_prefix=args.output_prefix,
        measurement_column=args.measurement_column,
        trajectory_nodes=args.trajectory_node,
        show=args.show,
    )
    print(f"Saved geometry figure to {outputs.geometry_figure}")
    print(f"Saved analysis figure to {outputs.analysis_figure}")
    for trajectory_figure in outputs.trajectory_figures:
        print(f"Saved trajectory figure to {trajectory_figure}")
    return 0


def register_visualize_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "visualize",
        help="Generate geometry, spectrum, and trajectory figures from a model JSON and response CSV.",
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument("response_csv", type=Path, help="Path to the solver response CSV file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Prefix for saved figures. Defaults to the response CSV stem in the same directory.",
    )
    parser.add_argument(
        "--measurement-column",
        type=str,
        default=None,
        help="Measurement column to highlight in the time-history figure.",
    )
    parser.add_argument(
        "--trajectory-node",
        action="append",
        default=None,
        help="Observation id to use for trajectory plots. Repeat to request multiple nodes.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive figures after saving them.",
    )
    parser.set_defaults(handler=partial(_handle_visualize, application=application))
