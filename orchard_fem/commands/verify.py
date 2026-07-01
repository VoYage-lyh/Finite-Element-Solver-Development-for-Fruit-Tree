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
        output_dir=args.output_dir,
        pytest_args=args.pytest_arg,
    )
    print("Validation completed.")
    if outputs.pytest_targets:
        print("  pytest targets:")
        for target in outputs.pytest_targets:
            print(f"    - {target}")
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
        "--enable-dolfinx-tests",
        action="store_true",
        help="Enable the optional DOLFINx smoke and benchmark tests in the active environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/validation/python"),
        help="Directory for validation artifacts.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=None,
        help="Additional argument to forward to pytest. Repeat to pass multiple args.",
    )
    parser.set_defaults(handler=partial(_handle_verify, application=application))
