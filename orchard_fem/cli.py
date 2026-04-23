from __future__ import annotations

import argparse
from typing import Sequence

from orchard_fem.application import OrchardApplication
from orchard_fem.commands import register_all_commands


def build_parser(application: OrchardApplication | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Python-first CLI for orchard vibration modeling, solving, and visualization."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_all_commands(subparsers, application or OrchardApplication())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
