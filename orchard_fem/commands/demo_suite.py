from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_demo_suite(args: argparse.Namespace, application: OrchardApplication) -> int:
    outputs = application.run_demo_suite(
        output_dir=args.output_dir,
        frequency_model=args.frequency_model,
        time_model=args.time_model,
        num_modes=args.num_modes,
    )
    print("Orchard FEM demo suite completed.")
    print(f"  frequency_response: {outputs.frequency_response_csv}")
    print(f"  time_history: {outputs.time_history_csv}")
    print(f"  modal_summary: {outputs.modal_summary_csv}")
    return 0


def register_demo_suite_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "demo-suite",
        help="Run the standard Orchard FEM demo suite and regenerate validation CSV artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/validation/python"),
        help="Directory for generated CSV validation artifacts.",
    )
    parser.add_argument(
        "--frequency-model",
        type=Path,
        default=Path("examples/demo_orchard.json"),
        help="Frequency-response demo model.",
    )
    parser.add_argument(
        "--time-model",
        type=Path,
        default=Path("examples/demo_orchard_time_history.json"),
        help="Time-history demo model.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=6,
        help="Number of modes in the modal summary export.",
    )
    parser.set_defaults(handler=partial(_handle_demo_suite, application=application))
