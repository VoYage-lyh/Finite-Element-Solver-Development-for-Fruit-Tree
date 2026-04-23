from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_modal(args: argparse.Namespace, application: OrchardApplication) -> int:
    output_csv = application.run_modal_summary(
        model_json=args.model_json,
        output_csv=args.output_csv,
        num_modes=args.num_modes,
    )
    print(f"Wrote modal summary to {output_csv}")
    return 0


def register_modal_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "modal",
        help="Solve modal frequencies for a model and write a modal summary CSV.",
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Override the modal summary CSV path. Defaults to build/<model>_modal_summary.csv.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=6,
        help="Number of physical modes to export.",
    )
    parser.set_defaults(handler=partial(_handle_modal, application=application))
