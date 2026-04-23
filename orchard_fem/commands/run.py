from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_run(args: argparse.Namespace, application: OrchardApplication) -> int:
    outputs = application.run_analysis(
        model_json=args.model_json,
        output_csv=args.output_csv,
    )
    print(f"Wrote {outputs.mode.value} results to {outputs.output_csv}")
    return 0


def register_run_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run the analysis mode configured in a model JSON and write the response CSV.",
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override the response CSV path. Defaults to build/<analysis.output_csv>.",
    )
    parser.set_defaults(handler=partial(_handle_run, application=application))
