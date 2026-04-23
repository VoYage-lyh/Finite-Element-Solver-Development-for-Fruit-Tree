from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from orchard_fem.workflows.demo import DemoSuiteOutputs, run_standard_demo_suite
from orchard_fem.workflows.analysis import write_modal_summary

__all__ = ["DemoSuiteOutputs", "run_standard_demo_suite", "write_modal_summary"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standard Python PETSc/SLEPc demo suite and write validation artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = run_standard_demo_suite(
        output_dir=args.output_dir,
        frequency_model=args.frequency_model,
        time_model=args.time_model,
        num_modes=args.num_modes,
    )
    print("Python demo suite completed.")
    print(f"  frequency_response: {outputs.frequency_response_csv}")
    print(f"  time_history: {outputs.time_history_csv}")
    print(f"  modal_summary: {outputs.modal_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
