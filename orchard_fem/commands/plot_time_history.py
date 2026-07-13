from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_plot_time_history(
    args: argparse.Namespace,
    application: OrchardApplication,
) -> int:
    application.plot_time_history(
        args.response_csv,
        show=not args.no_show,
        output_path=args.output,
    )
    return 0


def register_plot_time_history_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "plot-time-history",
        help="Plot acceleration time history from a time-history CSV.",
    )
    parser.add_argument("response_csv", type=Path, help="Path to a time-history CSV file.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plot without opening an interactive window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the figure to this path (e.g. build/accel.png).",
    )
    parser.set_defaults(handler=partial(_handle_plot_time_history, application=application))
