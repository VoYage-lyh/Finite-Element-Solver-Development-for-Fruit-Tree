"""Per-branch adaptive re-meshing of an Orchard FEM model by target element length.

Background
----------
The convergence study (``scripts/convergence_study.py``) showed the real trees
are under-resolved: at the stored per-branch ``num_elements`` (4-9) the natural
frequencies still carry several percent of discretization error, and the
coarsest branches have elements up to ~0.26 m. Natural frequencies set the
harvest resonance / excitation frequency, so that error feeds straight into the
recommended drive frequency.

This script caps every beam element at a target length ``h`` and lets each branch
take as many elements as its own length needs:

    num_elements(branch) = clamp(ceil(branch_length / h), min_elements, max_elements)

That is the same length-driven rule the skeleton importer already uses
(``orchard_fem/io/skeleton_import.py``), here applied to an existing solver-ready
model so short branches stay cheap and long branches get refined.

It is non-destructive: only ``discretization.num_elements`` is rewritten (other
discretization keys such as ``hotspot`` are preserved); every other field is
copied through untouched. Output goes to ``<model>_adaptive.json`` unless
``--in-place`` is given.

Example
-------
    python scripts/remesh_adaptive.py trees/tree_1.json --target-h 0.10
    python scripts/remesh_adaptive.py trees/tree_*.json --target-h 0.10 --in-place
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

from orchard_fem.io import load_orchard_model


def adaptive_num_elements(
    length: float,
    target_element_length: float,
    min_elements: int,
    max_elements: int,
) -> int:
    """Elements needed to keep every element <= ``target_element_length``, clamped."""
    raw = int(ceil(length / max(target_element_length, 1.0e-12)))
    return max(min_elements, min(raw, max_elements))


def remesh_model(
    model_path: Path,
    target_h: float,
    min_elements: int,
    max_elements: int,
) -> tuple[dict, list[tuple[str, float, int, int]]]:
    """Return the re-meshed payload and a (branch_id, length, old_nel, new_nel) report."""
    payload = json.loads(model_path.read_text())
    # Use the domain model for an accurate polyline arc length per branch
    # (handles start/end and multi-point stations consistently).
    model = load_orchard_model(str(model_path))
    length_by_id = {branch.branch_id: branch.path.length() for branch in model.branches}

    report: list[tuple[str, float, int, int]] = []
    for branch in payload["branches"]:
        branch_id = branch["id"]
        length = length_by_id[branch_id]
        discretization = branch.setdefault("discretization", {})
        old_nel = int(discretization.get("num_elements", 4))
        new_nel = adaptive_num_elements(length, target_h, min_elements, max_elements)
        discretization["num_elements"] = new_nel
        report.append((branch_id, length, old_nel, new_nel))
    return payload, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_json", type=Path, nargs="+", help="Orchard model JSON file(s).")
    parser.add_argument(
        "--target-h",
        type=float,
        default=0.10,
        help="Target maximum element length in metres (default 0.10, ~the nel=8 convergence level).",
    )
    parser.add_argument("--min-elements", type=int, default=4, help="Floor on elements per branch.")
    parser.add_argument("--max-elements", type=int, default=24, help="Cap on elements per branch.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing <model>_adaptive.json.",
    )
    parser.add_argument("--suffix", type=str, default="_adaptive", help="Suffix for non-in-place output.")
    args = parser.parse_args()

    for model_path in args.model_json:
        payload, report = remesh_model(
            model_path, args.target_h, args.min_elements, args.max_elements
        )
        old_total = sum(old for _, _, old, _ in report)
        new_total = sum(new for _, _, _, new in report)
        changed = [row for row in report if row[2] != row[3]]

        if args.in_place:
            out_path = model_path
        else:
            out_path = model_path.with_name(f"{model_path.stem}{args.suffix}{model_path.suffix}")
        out_path.write_text(json.dumps(payload, indent=2))

        print(
            f"{model_path.name}: {len(report)} branches, "
            f"elements {old_total} -> {new_total} "
            f"({len(changed)} branches changed, target h={args.target_h} m) -> {out_path.name}"
        )
        for branch_id, length, old_nel, new_nel in report:
            if old_nel != new_nel:
                print(f"    {branch_id:<32} L={length:5.3f} m  nel {old_nel:>2d} -> {new_nel:>2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
