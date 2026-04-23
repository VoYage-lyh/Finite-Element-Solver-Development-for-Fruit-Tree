from __future__ import annotations

import argparse
from functools import partial

from orchard_fem.application import OrchardApplication


def _handle_doctor(_: argparse.Namespace, application: OrchardApplication) -> int:
    return application.doctor()


def register_doctor_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="Inspect the active Python environment and report missing dependencies.",
        description="Inspect the active Python environment and report missing dependencies.",
    )
    parser.set_defaults(handler=partial(_handle_doctor, application=application))
