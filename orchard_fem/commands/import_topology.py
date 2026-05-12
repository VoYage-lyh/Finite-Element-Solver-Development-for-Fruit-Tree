from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication
from orchard_fem.io.topology_import import (
    convert_topology_file,
    convert_topology_to_orchard_file,
)


def _handle_import_topology(args: argparse.Namespace, application: OrchardApplication) -> int:
    del application
    if args.skeleton_only:
        output = convert_topology_file(Path(args.input_json), Path(args.output_json))
        print(f"Wrote skeleton JSON: {output}")
    else:
        output = convert_topology_to_orchard_file(Path(args.input_json), Path(args.output_json))
        print(f"Wrote orchard FEM model JSON: {output}")
    return 0


def register_import_topology_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "import-topology",
        help=(
            "Convert a topological tree description (length, diameter, attachment "
            "position, inclination, azimuth) into a solver-ready orchard FEM JSON."
        ),
    )
    parser.add_argument("input_json", type=Path, help="Input topology JSON path.")
    parser.add_argument("output_json", type=Path, help="Output orchard FEM model JSON path.")
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        default=False,
        help="Stop after producing the intermediate skeleton JSON (instead of the full model).",
    )
    parser.set_defaults(handler=partial(_handle_import_topology, application=application))
