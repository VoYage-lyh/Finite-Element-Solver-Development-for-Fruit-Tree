from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication
from orchard_fem.legacy import LegacyCompareRequest


def _handle_legacy_compare(
    args: argparse.Namespace,
    application: OrchardApplication,
) -> int:
    request = LegacyCompareRequest(
        model_json=args.model_json,
        baseline_csv=args.baseline_csv,
        cli_path=args.cli,
        candidate_csv=args.candidate_csv,
        run_python_candidate=args.run_python_candidate,
        python_modal_summary=args.python_modal_summary,
        num_modes=args.num_modes,
    )
    application.legacy_compare(request)
    return 0


def register_legacy_compare_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "legacy-compare",
        help="Run the archived C++ comparison helper for historical diagnostics.",
        description=(
            "Archival C++ comparison helper. This command is kept only for historical diagnostics "
            "and is not part of the Python-first correctness workflow."
        ),
    )
    parser.add_argument("model_json", type=Path, help="Input orchard model JSON file.")
    parser.add_argument(
        "baseline_csv",
        type=Path,
        help="Output CSV path for the historical C++ run.",
    )
    parser.add_argument(
        "--cli",
        type=Path,
        default=None,
        help="Path to the archived orchard_cli executable. Defaults to build/orchard_cli(.exe) if present.",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help="Optional Python/FEniCSx candidate output to compare against the baseline.",
    )
    parser.add_argument(
        "--run-python-candidate",
        action="store_true",
        help="Generate the candidate CSV with the current orchard_fem PETSc backend before comparing.",
    )
    parser.add_argument(
        "--python-modal-summary",
        type=Path,
        default=None,
        help="Optional CSV path for a Python-side modal summary generated from the current orchard_fem assembler.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=6,
        help="Number of modes to export when --python-modal-summary is provided.",
    )
    parser.set_defaults(handler=partial(_handle_legacy_compare, application=application))
