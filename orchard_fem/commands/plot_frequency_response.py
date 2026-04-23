from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_plot_frequency_response(
    args: argparse.Namespace,
    application: OrchardApplication,
) -> int:
    application.plot_frequency_response(args.response_csv, show=not args.no_show)
    return 0


def register_plot_frequency_response_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "plot-frequency-response",
        help="Plot the columns in a frequency-response CSV with matplotlib.",
    )
    parser.add_argument("response_csv", type=Path, help="Path to a frequency-response CSV file.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the plot without opening an interactive window.",
    )
    parser.set_defaults(handler=partial(_handle_plot_frequency_response, application=application))
