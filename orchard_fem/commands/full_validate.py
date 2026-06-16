from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication
from orchard_fem.automation import FullValidationConfig
from orchard_fem.automation.full_validation import DEFAULT_REPO_ROOT


def _handle_full_validate(args: argparse.Namespace, application: OrchardApplication) -> int:
    config = FullValidationConfig(
        repo_root=DEFAULT_REPO_ROOT,
        build_dir=args.build_dir,
        validation_dir=args.validation_dir,
        orchard_fenicsx_env=args.orchard_fenicsx_env,
        skip_integration_tests=args.skip_integration_tests,
        skip_fenicsx_tests=args.skip_fenicsx_tests,
        skip_python_demo_suite=args.skip_python_demo_suite,
    )
    application.full_validate(config)
    return 0


def register_full_validate_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    defaults = FullValidationConfig.from_environment(DEFAULT_REPO_ROOT)
    parser = subparsers.add_parser(
        "full-validate",
        help="Run the multi-environment Orchard FEM validation workflow.",
        description=(
            "Run the Orchard FEM integration + verification workflow in the orchard-fenicsx "
            "environment from the package CLI. Environment variables such as BUILD_DIR and "
            "ORCHARD_FENICSX_ENV are used as defaults."
        ),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=defaults.build_dir,
        help="Build artifact directory. Defaults to $BUILD_DIR or ./build.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=defaults.validation_dir,
        help="Validation artifact directory. Defaults to $VALIDATION_DIR or <build>/validation.",
    )
    parser.add_argument(
        "--orchard-fenicsx-env",
        type=str,
        default=defaults.orchard_fenicsx_env,
        help="Conda environment name for the PETSc/SLEPc workflow.",
    )
    parser.add_argument(
        "--skip-integration-tests",
        action="store_true",
        default=defaults.skip_integration_tests,
        help="Skip the integration-test step.",
    )
    parser.add_argument(
        "--skip-fenicsx-tests",
        action="store_true",
        default=defaults.skip_fenicsx_tests,
        help="Skip the orchard-fenicsx verification step.",
    )
    parser.add_argument(
        "--skip-python-demo-suite",
        action="store_true",
        default=defaults.skip_python_demo_suite,
        help="Skip demo CSV regeneration in the orchard-fenicsx step.",
    )
    parser.set_defaults(handler=partial(_handle_full_validate, application=application))
