from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_verify(args: argparse.Namespace, application: OrchardApplication) -> int:
    outputs = application.verify(
        include_integration=not args.skip_integration,
        include_verification=not args.skip_verification,
        include_dolfinx_tests=args.enable_dolfinx_tests,
        include_demo_suite=not args.skip_demo_suite,
        output_dir=args.output_dir,
        pytest_args=args.pytest_arg,
    )
    print("Validation completed.")
    if outputs.pytest_targets:
        print("  pytest targets:")
        for target in outputs.pytest_targets:
            print(f"    - {target}")
    if outputs.demo_suite_outputs is not None:
        demo_outputs = outputs.demo_suite_outputs
        print(f"  frequency_response: {demo_outputs.frequency_response_csv}")
        print(f"  time_history: {demo_outputs.time_history_csv}")
        print(f"  modal_summary: {demo_outputs.modal_summary_csv}")
        print(f"  frequency_geometry: {demo_outputs.frequency_visualization.geometry_figure}")
        print(f"  frequency_figure: {demo_outputs.frequency_visualization.analysis_figure}")
        print(f"  time_geometry: {demo_outputs.time_history_visualization.geometry_figure}")
        print(f"  time_figure: {demo_outputs.time_history_visualization.analysis_figure}")
        for trajectory_figure in demo_outputs.time_history_visualization.trajectory_figures:
            print(f"  trajectory: {trajectory_figure}")
    return 0


def register_verify_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "verify",
        help="Run the Orchard FEM validation workflow in the current environment.",
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip the general Orchard FEM integration tests.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the PETSc/SLEPc verification benchmarks.",
    )
    parser.add_argument(
        "--skip-demo-suite",
        action="store_true",
        help="Skip regeneration of the standard demo CSV and figure artifacts.",
    )
    parser.add_argument(
        "--enable-dolfinx-tests",
        action="store_true",
        help="Enable the optional DOLFINx smoke and benchmark tests in the active environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/validation/python"),
        help="Directory for demo-suite CSV and figure artifacts.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=None,
        help="Additional argument to forward to pytest. Repeat to pass multiple args.",
    )
    parser.set_defaults(handler=partial(_handle_verify, application=application))
