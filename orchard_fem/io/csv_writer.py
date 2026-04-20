from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrequencyResponseRow:
    frequency_hz: float
    excitation_response: float
    observation_values: list[float]


@dataclass(frozen=True)
class TimeHistoryRow:
    time_seconds: float
    excitation_signal: float
    excitation_load: float
    excitation_response: float
    observation_values: list[float]


def write_frequency_response_csv(
    file_path: str | Path,
    observation_names: list[str],
    rows: list[FrequencyResponseRow],
) -> None:
    path = Path(file_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "excitation_response", *observation_names])
        for row in rows:
            writer.writerow([row.frequency_hz, row.excitation_response, *row.observation_values])


def write_time_history_csv(
    file_path: str | Path,
    observation_names: list[str],
    rows: list[TimeHistoryRow],
) -> None:
    path = Path(file_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "excitation_signal",
                "excitation_load",
                "excitation_response",
                *observation_names,
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.time_seconds,
                    row.excitation_signal,
                    row.excitation_load,
                    row.excitation_response,
                    *row.observation_values,
                ]
            )
