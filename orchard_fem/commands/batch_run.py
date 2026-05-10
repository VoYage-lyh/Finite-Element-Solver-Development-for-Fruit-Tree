"""batch-run CLI command — run frequency response for multiple excitation points."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from orchard_fem.application import OrchardApplication


def _handle_batch_run(args: argparse.Namespace, application: OrchardApplication) -> int:
    del application

    from orchard_fem.fenicsx.availability import require_dolfinx
    from orchard_fem.io import load_orchard_model
    from orchard_fem.workflows.batch_excitation import ExcitationSpec, run_batch_frequency_response

    require_dolfinx()

    model = load_orchard_model(str(args.model_json))

    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    raw_specs = config.get("excitation_specs", [])
    if not raw_specs:
        print("No excitation_specs found in config JSON — nothing to do.")
        return 0

    specs = [
        ExcitationSpec(
            branch_id=str(s["branch_id"]),
            node=str(s["node"]),
            component=str(s["component"]),
            amplitude=float(s.get("amplitude", 1.0)),
            phase_degrees=float(s.get("phase_degrees", 0.0)),
        )
        for s in raw_specs
    ]

    results = run_batch_frequency_response(model, specs)

    output_dir = Path(args.output_dir) if args.output_dir else Path("build") / "batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_result in results:
        spec = batch_result.excitation_spec
        stem = f"{spec.branch_id}_{spec.node}_{spec.component}"
        csv_path = output_dir / f"{stem}.csv"
        batch_result.result.write_csv(str(csv_path))
        print(f"  {stem}: {len(batch_result.result.points)} freq points → {csv_path}")

    print(f"Batch run complete: {len(results)} excitation specs written to {output_dir}")
    return 0


def register_batch_run_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    parser = subparsers.add_parser(
        "batch-run",
        help=(
            "Run frequency response for multiple excitation points defined in a JSON config, "
            "assembling K/M/C once and sweeping all specs per frequency step."
        ),
    )
    parser.add_argument("model_json", type=Path, help="Path to the Orchard FEM model JSON.")
    parser.add_argument(
        "config_json",
        type=Path,
        help=(
            "Path to a batch config JSON with an 'excitation_specs' array. "
            "Each entry: {branch_id, node, component, amplitude?, phase_degrees?}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write per-spec CSV files (default: build/batch/).",
    )
    parser.set_defaults(handler=partial(_handle_batch_run, application=application))
