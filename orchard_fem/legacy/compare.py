from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from orchard_fem.io.model_loader import load_orchard_model
from orchard_fem.model import AnalysisMode
from orchard_fem.solvers.frequency_response import (
    FrequencyResponseRequest,
    PETScFrequencyResponseSolver,
)
from orchard_fem.solvers.modal import ModalAnalysisRequest, SLEPcModalSolver
from orchard_fem.solvers.modal_assembler import OrchardModalAssembler
from orchard_fem.solvers.time_history import PETScTimeHistorySolver, TimeHistoryRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
IS_WINDOWS = sys.platform.startswith("win")


@dataclass(frozen=True)
class LegacyCompareRequest:
    model_json: Path
    baseline_csv: Path
    cli_path: Path | None = None
    candidate_csv: Path | None = None
    run_python_candidate: bool = False
    python_modal_summary: Path | None = None
    num_modes: int = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archival C++ comparison helper. This tool is kept only for historical diagnostics "
            "and is not part of the Python-first correctness workflow."
        )
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
    return parser


def default_cli_path() -> Path | None:
    candidates = [
        Path("build/orchard_cli"),
        Path("build/Debug/orchard_cli"),
        Path("build/Release/orchard_cli"),
        Path("build/orchard_cli.exe"),
        Path("build/Debug/orchard_cli.exe"),
        Path("build/Release/orchard_cli.exe"),
    ]
    if IS_WINDOWS:
        candidates = sorted(candidates, key=lambda candidate: 0 if candidate.suffix == ".exe" else 1)

    for candidate in candidates:
        resolved = (REPO_ROOT / candidate).resolve()
        if resolved.exists():
            return resolved

    executable = shutil.which("orchard_cli")
    return Path(executable) if executable is not None else None


def run_cpp_baseline(cli_path: Path, model_json: Path, baseline_csv: Path) -> None:
    baseline_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [str(cli_path), str(model_json), str(baseline_csv)]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "C++ baseline run failed.\nSTDOUT:\n{0}\nSTDERR:\n{1}".format(
                completed.stdout, completed.stderr
            )
        )
    sys.stdout.write(completed.stdout)


def load_csv(file_path: Path) -> tuple[list[str], list[list[float]]]:
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"CSV contains no data rows: {file_path}")

    return rows[0], [[float(value) for value in row] for row in rows[1:]]


def compare_csv(baseline_csv: Path, candidate_csv: Path) -> None:
    baseline_header, baseline_rows = load_csv(baseline_csv)
    candidate_header, candidate_rows = load_csv(candidate_csv)

    if baseline_header != candidate_header:
        raise ValueError(
            "CSV headers differ.\nBaseline: {0}\nCandidate: {1}".format(
                baseline_header, candidate_header
            )
        )
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError(
            "CSV row counts differ. Baseline={0}, Candidate={1}".format(
                len(baseline_rows), len(candidate_rows)
            )
        )

    max_abs_error = 0.0
    max_rel_error = 0.0
    worst_column = baseline_header[0]

    for row_index, (baseline_row, candidate_row) in enumerate(zip(baseline_rows, candidate_rows)):
        if len(baseline_row) != len(candidate_row):
            raise ValueError(f"Column count mismatch at row {row_index}")

        for column_index, (baseline_value, candidate_value) in enumerate(
            zip(baseline_row, candidate_row)
        ):
            abs_error = abs(candidate_value - baseline_value)
            rel_error = abs_error / max(abs(baseline_value), 1.0e-12)
            if abs_error > max_abs_error:
                max_abs_error = abs_error
                worst_column = baseline_header[column_index]
            if rel_error > max_rel_error:
                max_rel_error = rel_error

    print("Comparison summary")
    print("  columns: {0}".format(", ".join(baseline_header)))
    print("  max_abs_error: {0:.6e}".format(max_abs_error))
    print("  max_rel_error: {0:.6e}".format(max_rel_error))
    print("  worst_column: {0}".format(worst_column))


def run_python_candidate(model_json: Path, candidate_csv: Path) -> None:
    model = load_orchard_model(str(model_json))
    candidate_csv.parent.mkdir(parents=True, exist_ok=True)

    if model.analysis.mode == AnalysisMode.FREQUENCY_RESPONSE:
        PETScFrequencyResponseSolver().solve(
            FrequencyResponseRequest(
                model_path=str(model_json),
                output_csv=str(candidate_csv),
            )
        )
        print("Python frequency-response candidate written to {0}".format(candidate_csv))
        return

    if model.analysis.mode == AnalysisMode.TIME_HISTORY:
        PETScTimeHistorySolver().solve(
            TimeHistoryRequest(
                model_path=str(model_json),
                output_csv=str(candidate_csv),
            )
        )
        print("Python time-history candidate written to {0}".format(candidate_csv))
        return

    raise ValueError("Unsupported analysis mode for candidate generation: {0}".format(model.analysis.mode))


def write_python_modal_summary(model_json: Path, output_csv: Path, num_modes: int) -> None:
    model = load_orchard_model(str(model_json))
    assembled = OrchardModalAssembler().assemble(model)

    solver = SLEPcModalSolver()
    modes = solver.solve(
        ModalAnalysisRequest(
            num_modes=num_modes,
            stiffness_matrix=assembled.stiffness_matrix,
            mass_matrix=assembled.mass_matrix,
            dof_labels=assembled.dof_labels,
        )
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode_index", "frequency_hz", "eigenvalue", "modal_mass", "backend"])
        backend_name = "slepc"
        for mode in modes:
            writer.writerow(
                [
                    mode.mode_index,
                    f"{mode.frequency_hz:.12e}",
                    f"{mode.eigenvalue:.12e}",
                    f"{mode.modal_mass:.12e}",
                    backend_name,
                ]
            )

    print("Python modal summary written to {0}".format(output_csv))


def run_legacy_compare(request: LegacyCompareRequest) -> None:
    cli_path = request.cli_path.resolve() if request.cli_path is not None else default_cli_path()
    if cli_path is None or not cli_path.exists():
        raise FileNotFoundError(
            "Could not locate the archived orchard_cli executable. "
            "See docs/legacy_reference.md or pass --cli."
        )
    if request.run_python_candidate and request.candidate_csv is None:
        raise ValueError("--run-python-candidate requires --candidate-csv")

    model_json = request.model_json.resolve()
    baseline_csv = request.baseline_csv.resolve()
    candidate_csv = request.candidate_csv.resolve() if request.candidate_csv is not None else None
    python_modal_summary = (
        request.python_modal_summary.resolve()
        if request.python_modal_summary is not None
        else None
    )

    run_cpp_baseline(cli_path, model_json, baseline_csv)
    print("Baseline CSV written to {0}".format(baseline_csv))

    if request.run_python_candidate and candidate_csv is not None:
        run_python_candidate(model_json, candidate_csv)

    if candidate_csv is None:
        print("No candidate CSV provided. Baseline generation completed.")
    else:
        compare_csv(baseline_csv, candidate_csv)

    if python_modal_summary is not None:
        write_python_modal_summary(model_json, python_modal_summary, request.num_modes)


def main(argv: Sequence[str] | None = None) -> int:
    print(
        "Legacy notice: this utility is archival only. "
        "Prefer `python -m orchard_fem verify` and Python-side analytical benchmarks for current validation."
    )
    args = build_parser().parse_args(argv)
    request = LegacyCompareRequest(
        model_json=args.model_json,
        baseline_csv=args.baseline_csv,
        cli_path=args.cli,
        candidate_csv=args.candidate_csv,
        run_python_candidate=args.run_python_candidate,
        python_modal_summary=args.python_modal_summary,
        num_modes=args.num_modes,
    )
    run_legacy_compare(request)
    return 0
