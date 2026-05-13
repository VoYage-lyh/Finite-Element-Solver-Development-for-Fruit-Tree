from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_view_tree(
    args: argparse.Namespace,
    application: OrchardApplication,
) -> int:
    application.view_tree(
        model_json=args.model_json,
        show=not args.no_show,
        output_path=args.output,
    )
    return 0


def register_view_tree_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "view-tree",
        help="Open an interactive 3D visualization of the fruit-tree structure.",
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Render without opening an interactive window (useful with --output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the figure to this path (e.g. results/tree.png).",
    )
    parser.set_defaults(handler=partial(_handle_view_tree, application=application))
