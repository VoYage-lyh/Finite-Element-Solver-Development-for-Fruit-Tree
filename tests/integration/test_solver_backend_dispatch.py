from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from orchard_fem.workflows.analysis import run_configured_analysis, write_modal_summary


def _fenicsx_frequency_payload() -> dict:
    return {
        "metadata": {"name": "fenicsx_dispatch_frequency"},
        "materials": [
            {
                "id": "xylem_default",
                "tissue": "xylem",
                "model": "linear",
                "density": 750.0,
                "youngs_modulus": 1.0e10,
                "poisson_ratio": 0.30,
                "damping_ratio": 0.01,
            }
        ],
        "branches": [
            {
                "id": "cantilever",
                "parent_branch_id": None,
                "level": 0,
                "start": [0.0, 0.0, 0.0],
                "end": [0.0, 0.0, 1.0],
                "discretization": {"num_elements": 4, "hotspot": False},
                "stations": [
                    {
                        "s": 0.0,
                        "profile_type": "parameterized",
                        "regions": [
                            {
                                "tissue": "xylem",
                                "material_id": "xylem_default",
                                "shape": {
                                    "type": "solid_ellipse",
                                    "center": [0.0, 0.0],
                                    "radii": [0.02, 0.02],
                                    "samples": 48,
                                },
                            }
                        ],
                    },
                    {
                        "s": 1.0,
                        "profile_type": "parameterized",
                        "regions": [
                            {
                                "tissue": "xylem",
                                "material_id": "xylem_default",
                                "shape": {
                                    "type": "solid_ellipse",
                                    "center": [0.0, 0.0],
                                    "radii": [0.02, 0.02],
                                    "samples": 48,
                                },
                            }
                        ],
                    },
                ],
            }
        ],
        "joints": [],
        "fruits": [],
        "clamps": [
            {
                "branch_id": "cantilever",
                "support_stiffness": 1.0,
                "support_damping": 0.0,
                "cubic_stiffness": 0.0,
            }
        ],
        "excitation": {
            "kind": "harmonic_force",
            "target_branch_id": "cantilever",
            "target_node": "tip",
            "target_component": "uy",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 4.0,
        },
        "analysis": {
            "mode": "frequency_response",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 5.0,
            "frequency_steps": 3,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 1.0e-4,
            "output_csv": "unused.csv",
        },
        "observations": [
            {
                "id": "tip",
                "target_type": "branch",
                "target_id": "cantilever",
                "target_node": "tip",
                "target_component": "uy",
            }
        ],
    }


def _fenicsx_time_history_payload() -> dict:
    payload = _fenicsx_frequency_payload()
    payload["metadata"]["name"] = "fenicsx_dispatch_time_history"
    payload["analysis"] = {
        "mode": "time_history",
        "frequency_start_hz": 1.0,
        "frequency_end_hz": 5.0,
        "frequency_steps": 3,
        "time_step_seconds": 0.01,
        "total_time_seconds": 0.05,
        "output_stride": 1,
        "rayleigh_alpha": 0.0,
        "rayleigh_beta": 1.0e-4,
        "output_csv": "unused.csv",
    }
    return payload


@dataclass(frozen=True)
class _FakePoint:
    frequency_hz: float
    excitation_response_magnitude: float
    observation_magnitudes: list[float]


class _FakeFrequencyResult:
    def __init__(self) -> None:
        self.observation_names = ["tip"]
        self.points = [_FakePoint(1.0, 2.0, [3.0])]

    def write_csv(self, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frequency_hz", "excitation_response", "tip"])
            writer.writerow(["1.0", "2.0", "3.0"])


@dataclass(frozen=True)
class _FakeFrequencyExperiment:
    result: _FakeFrequencyResult


@dataclass(frozen=True)
class _FakeMode:
    mode_index: int
    frequency_hz: float
    eigenvalue: float
    modal_mass: float
    mode_shape: list[float]


@dataclass(frozen=True)
class _FakeModalExperiment:
    modes: list[_FakeMode]


class _FakeTimeHistoryResult:
    def __init__(self) -> None:
        self.observation_names = ["tip"]
        self.points = []

    def write_csv(self, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "excitation_signal", "excitation_load", "excitation_response", "tip"])
            writer.writerow(["0.0", "0.0", "0.0", "0.0", "0.0"])


@dataclass(frozen=True)
class _FakeTimeHistoryExperiment:
    result: _FakeTimeHistoryResult


def test_run_configured_analysis_dispatches_to_fenicsx_backend(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "fenicsx_dispatch_frequency.json"
    model_path.write_text(json.dumps(_fenicsx_frequency_payload()), encoding="utf-8")
    output_csv = tmp_path / "fenicsx_frequency.csv"

    calls: list[str] = []

    def fake_frequency_solver(model, **kwargs):
        del model, kwargs
        calls.append("fenicsx-frequency")
        return _FakeFrequencyExperiment(result=_FakeFrequencyResult())

    monkeypatch.setattr(
        "orchard_fem.fenicsx.solve_embedded_beam_frequency_response_experiment",
        fake_frequency_solver,
    )

    outputs = run_configured_analysis(model_path, output_csv=output_csv)

    assert calls == ["fenicsx-frequency"]
    assert outputs.output_csv == output_csv
    assert output_csv.exists()


def test_run_configured_time_history_dispatches_to_fenicsx_backend(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "fenicsx_dispatch_time_history.json"
    model_path.write_text(json.dumps(_fenicsx_time_history_payload()), encoding="utf-8")
    output_csv = tmp_path / "fenicsx_time_history.csv"

    calls: list[str] = []

    def fake_time_history_solver(model, **kwargs):
        del model, kwargs
        calls.append("fenicsx-time-history")
        return _FakeTimeHistoryExperiment(result=_FakeTimeHistoryResult())

    monkeypatch.setattr(
        "orchard_fem.fenicsx.solve_embedded_beam_time_history_experiment",
        fake_time_history_solver,
    )

    outputs = run_configured_analysis(model_path, output_csv=output_csv)

    assert calls == ["fenicsx-time-history"]
    assert outputs.output_csv == output_csv
    assert output_csv.exists()


def test_write_modal_summary_dispatches_to_fenicsx_backend(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "fenicsx_dispatch_frequency.json"
    model_path.write_text(json.dumps(_fenicsx_frequency_payload()), encoding="utf-8")
    output_csv = tmp_path / "fenicsx_modal.csv"

    calls: list[str] = []

    def fake_modal_solver(model, *, num_modes: int, **kwargs):
        del model, kwargs
        calls.append(f"fenicsx-modal-{num_modes}")
        return _FakeModalExperiment(
            modes=[
                _FakeMode(
                    mode_index=1,
                    frequency_hz=4.2,
                    eigenvalue=123.0,
                    modal_mass=1.0,
                    mode_shape=[0.0, 1.0],
                )
            ]
        )

    monkeypatch.setattr(
        "orchard_fem.fenicsx.solve_embedded_beam_modal_experiment",
        fake_modal_solver,
    )

    resolved = write_modal_summary(model_path, output_csv, 1)

    assert calls == ["fenicsx-modal-1"]
    assert resolved == output_csv
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][-1] == "fenicsx"
