from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize orchard geometry together with excitation and response data."
    )
    parser.add_argument("model_json", type=Path, help="Path to the orchard model JSON file.")
    parser.add_argument("response_csv", type=Path, help="Path to the solver response CSV file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Prefix for saved figures. Defaults to the response CSV stem in the same directory.",
    )
    parser.add_argument(
        "--measurement-column",
        type=str,
        default=None,
        help="Measurement column to highlight in the time-history figure.",
    )
    parser.add_argument(
        "--trajectory-node",
        action="append",
        default=None,
        help="Observation id to use for trajectory plots. Repeat to request multiple nodes.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive figures after saving them.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_model(model_path: Path) -> dict:
    with model_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_response(response_path: Path) -> tuple[list[str], list[list[float]]]:
    with response_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise RuntimeError("Response CSV contains no data rows")

    headers = rows[0]
    data = [[float(value) for value in row] for row in rows[1:]]
    return headers, data


def build_series_map(headers: list[str], rows: list[list[float]]) -> dict[str, list[float]]:
    columns = list(zip(*rows))
    return {header: list(column) for header, column in zip(headers, columns)}


def choose_measurement_column(headers: list[str], preferred: Optional[str]) -> str:
    reserved = {
        "time_s",
        "frequency_hz",
        "excitation_signal",
        "excitation_load",
        "excitation_response",
    }
    measurement_headers = [header for header in headers if header not in reserved]

    if preferred is not None:
        if preferred not in headers:
            raise RuntimeError("Requested measurement column not found: {0}".format(preferred))
        return preferred

    if measurement_headers:
        return measurement_headers[0]

    return "excitation_response"


def frequency_figure_path(prefix: Path) -> Path:
    return Path(f"{prefix}_frequency_response.png")


def geometry_figure_path(prefix: Path) -> Path:
    return Path(f"{prefix}_geometry.png")


def time_frequency_figure_path(prefix: Path) -> Path:
    return Path(f"{prefix}_time_frequency.png")


def trajectory_figure_path(prefix: Path, node_id: str) -> Path:
    return Path(f"{prefix}_trajectory_{node_id}.png")
