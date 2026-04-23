from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from orchard_fem.workflows.validation import (
    DEFAULT_VALIDATION_OUTPUT_DIR,
    ValidationOutputs,
    run_validation_suite,
)

__all__ = ["DEFAULT_VALIDATION_OUTPUT_DIR", "ValidationOutputs", "run_validation_suite"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Python-first orchard validation workflow in the current environment."
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip the general Python integration tests.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the PETSc/SLEPc verification benchmarks.",
    )
    parser.add_argument(
        "--skip-demo-suite",
        action="store_true",
        help="Skip regeneration of the standard demo CSV artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT_DIR,
        help="Directory for demo-suite CSV artifacts.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=None,
        help="Additional argument to forward to pytest. Repeat to pass multiple args.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_validation_suite(
        include_integration=not args.skip_integration,
        include_verification=not args.skip_verification,
        include_demo_suite=not args.skip_demo_suite,
        output_dir=args.output_dir,
        pytest_args=args.pytest_arg,
    )

    print()
    print("Validation completed.")
    if outputs.pytest_targets:
        print("  pytest targets:")
        for target in outputs.pytest_targets:
            print(f"    - {target}")
    if outputs.demo_suite_outputs is not None:
        print(f"  frequency_response: {outputs.demo_suite_outputs.frequency_response_csv}")
        print(f"  time_history: {outputs.demo_suite_outputs.time_history_csv}")
        print(f"  modal_summary: {outputs.demo_suite_outputs.modal_summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
