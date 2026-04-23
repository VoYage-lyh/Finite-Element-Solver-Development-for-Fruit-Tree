import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orchard_fem import OrchardApplication
from orchard_fem.cli import build_parser
from orchard_fem.cli import main as cli_main


def _build_visualization_payload() -> dict:
    return {
        "metadata": {"name": "cli_visualization_demo"},
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
                                    "samples": 96,
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
                                    "samples": 96,
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
            "target_component": "ux",
            "amplitude": 1.0,
            "phase_degrees": 0.0,
            "driving_frequency_hz": 8.0,
        },
        "analysis": {
            "mode": "time_history",
            "frequency_start_hz": 1.0,
            "frequency_end_hz": 20.0,
            "frequency_steps": 10,
            "time_step_seconds": 0.002,
            "total_time_seconds": 0.05,
            "output_stride": 1,
            "max_nonlinear_iterations": 10,
            "nonlinear_tolerance": 1.0e-8,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 0.0,
            "output_csv": "cantilever_time_history.csv",
        },
        "observations": [
            {
                "id": "obs_tip",
                "target_type": "branch",
                "target_id": "cantilever",
                "target_node": "tip",
                "target_components": ["ux", "uy"],
            }
        ],
    }


def test_python_module_entrypoint_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "orchard_fem", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "modal" in result.stdout
    assert "visualize" in result.stdout
    assert "plot-frequency-response" in result.stdout
    assert "doctor" in result.stdout
    assert "full-validate" in result.stdout
    assert "legacy-compare" in result.stdout
    assert "verify" in result.stdout


def test_package_exports_orchard_application() -> None:
    application = OrchardApplication()
    assert application is not None


def test_cli_parser_registers_expected_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "run" in help_text
    assert "modal" in help_text
    assert "visualize" in help_text
    assert "plot-frequency-response" in help_text
    assert "demo-suite" in help_text
    assert "verify" in help_text
    assert "doctor" in help_text
    assert "full-validate" in help_text
    assert "legacy-compare" in help_text


def test_python_doctor_subcommand_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "orchard_fem", "doctor", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Inspect the active Python environment" in result.stdout


def test_python_doctor_subcommand_runs_environment_audit() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "orchard_fem", "doctor"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Missing summary" in result.stdout


def test_check_python_env_wrapper_runs_environment_audit() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_python_env.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Missing summary" in result.stdout


def test_python_verify_subcommand_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "orchard_fem", "verify", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--skip-integration" in result.stdout
    assert "--skip-verification" in result.stdout
    assert "--skip-demo-suite" in result.stdout


def test_python_full_validate_subcommand_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "orchard_fem", "full-validate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--orchard-dev-env" in result.stdout
    assert "--orchard-fenicsx-env" in result.stdout
    assert "--skip-dev-tests" in result.stdout
    assert "--skip-fenicsx-tests" in result.stdout


def test_python_plot_frequency_response_subcommand_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "orchard_fem", "plot-frequency-response", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "frequency-response CSV" in result.stdout
    assert "--no-show" in result.stdout


def test_python_legacy_compare_subcommand_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "orchard_fem", "legacy-compare", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--candidate-csv" in result.stdout
    assert "--python-modal-summary" in result.stdout


def test_run_full_validation_wrapper_forwards_help() -> None:
    result = subprocess.run(
        ["bash", "scripts/run_full_validation.sh", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "full-validate" in result.stdout


def test_plot_frequency_response_wrapper_forwards_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/plot_frequency_response.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "frequency-response CSV" in result.stdout


def test_benchmark_vs_existing_wrapper_forwards_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_vs_existing.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "archival only" in result.stdout
    assert "--candidate-csv" in result.stdout


def test_cli_run_writes_frequency_response_csv(tmp_path) -> None:
    pytest.importorskip("petsc4py")

    output_csv = tmp_path / "frequency_response.csv"
    exit_code = cli_main(
        [
            "run",
            "examples/demo_orchard.json",
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    assert output_csv.exists()
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "frequency_hz"
    assert len(rows) > 2


def test_cli_modal_writes_summary_csv(tmp_path) -> None:
    pytest.importorskip("slepc4py")

    output_csv = tmp_path / "modal_summary.csv"
    exit_code = cli_main(
        [
            "modal",
            "examples/demo_orchard.json",
            "--output-csv",
            str(output_csv),
            "--num-modes",
            "3",
        ]
    )

    assert exit_code == 0
    assert output_csv.exists()
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["mode_index", "frequency_hz", "eigenvalue", "modal_mass", "backend"]
    assert len(rows) == 4


def test_cli_visualize_writes_expected_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")

    model_path = tmp_path / "visualization_model.json"
    response_csv = tmp_path / "visualization_time_history.csv"
    output_prefix = tmp_path / "visualization"

    model_path.write_text(json.dumps(_build_visualization_payload()), encoding="utf-8")
    with response_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "excitation_response", "obs_tip_ux", "obs_tip_uy"])
        for index in range(64):
            writer.writerow(
                [
                    f"{index * 0.01:.6f}",
                    f"{0.1 * index:.6f}",
                    f"{0.5 * index:.6f}",
                    f"{0.25 * index:.6f}",
                ]
            )

    exit_code = cli_main(
        [
            "visualize",
            str(model_path),
            str(response_csv),
            "--output-prefix",
            str(output_prefix),
        ]
    )

    assert exit_code == 0
    assert Path(f"{output_prefix}_geometry.png").exists()
    assert Path(f"{output_prefix}_time_frequency.png").exists()
    assert Path(f"{output_prefix}_trajectory_obs_tip.png").exists()
