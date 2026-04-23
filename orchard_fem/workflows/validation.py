from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from orchard_fem.workflows.demo import DemoSuiteOutputs, run_standard_demo_suite

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_OUTPUT_DIR = Path("build/validation/python")

INTEGRATION_TEST_TARGETS = [
    "tests/integration/test_cross_section_defaults.py",
    "tests/integration/test_auto_nonlinear_injection.py",
    "tests/integration/test_gravity_prestress.py",
    "tests/integration/test_python_cli.py",
    "tests/integration/test_python_scaffold.py",
]

VERIFICATION_TEST_TARGETS = [
    "tests/verification/test_python_beam_benchmarks.py",
    "tests/verification/test_python_dynamic_benchmarks.py",
    "tests/integration/test_gravity_prestress.py::test_gravity_prestress_adds_load_and_reduces_first_mode",
]


@dataclass(frozen=True)
class ValidationOutputs:
    pytest_targets: list[str]
    demo_suite_outputs: DemoSuiteOutputs | None


def print_validation_step(message: str) -> None:
    print()
    print(f"==> {message}")


def require_pytest() -> None:
    if importlib.util.find_spec("pytest") is not None:
        return
    raise RuntimeError(
        "Validation requires pytest. Install the repository test extras with "
        '`python -m pip install -e ".[ubuntu-test]"` or use the conda environment from '
        "`config/fenicsx_pinn_environment.yml`. Run `python -m orchard_fem doctor` to audit "
        "the active environment."
    )


def run_pytest_targets(targets: Sequence[str], extra_args: Sequence[str] | None = None) -> None:
    if not targets:
        return

    require_pytest()
    command = [sys.executable, "-m", "pytest", "-q", *targets, *(extra_args or [])]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Validation pytest run failed with exit code "
            f"{completed.returncode}: {' '.join(command)}"
        )


def run_validation_suite(
    include_integration: bool = True,
    include_verification: bool = True,
    include_demo_suite: bool = True,
    output_dir: Path = DEFAULT_VALIDATION_OUTPUT_DIR,
    pytest_args: Sequence[str] | None = None,
) -> ValidationOutputs:
    pytest_targets: list[str] = []
    if include_integration:
        pytest_targets.extend(INTEGRATION_TEST_TARGETS)
    if include_verification:
        pytest_targets.extend(VERIFICATION_TEST_TARGETS)

    if pytest_targets:
        print_validation_step("Run Python verification tests")
        run_pytest_targets(pytest_targets, extra_args=pytest_args)

    demo_suite_outputs = None
    if include_demo_suite:
        print_validation_step("Run Python PETSc/SLEPc demo suite")
        demo_suite_outputs = run_standard_demo_suite(output_dir=output_dir)

    return ValidationOutputs(
        pytest_targets=pytest_targets,
        demo_suite_outputs=demo_suite_outputs,
    )
