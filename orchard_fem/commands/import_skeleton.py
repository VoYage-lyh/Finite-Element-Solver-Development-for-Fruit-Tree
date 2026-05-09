from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication
from orchard_fem.io.skeleton_import import convert_skeleton_file


def _handle_import_skeleton(args: argparse.Namespace, application: OrchardApplication) -> int:
    del application
    output_path = convert_skeleton_file(Path(args.input_json), Path(args.output_json))
    print(f"Converted skeleton model: {output_path}")
    return 0


def register_import_skeleton_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "import-skeleton",
        help="Convert a 3D orchard skeleton JSON into the solver-ready Orchard FEM JSON format.",
    )
    parser.add_argument("input_json", type=Path, help="Input skeleton JSON path.")
    parser.add_argument("output_json", type=Path, help="Output Orchard FEM model JSON path.")
    parser.set_defaults(handler=partial(_handle_import_skeleton, application=application))
