from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FullValidationStep:
    label: str
    command: list[str]


@dataclass(frozen=True)
class FullValidationConfig:
    repo_root: Path
    build_dir: Path
    validation_dir: Path
    orchard_dev_env: str = "orchard-dev"
    orchard_fenicsx_env: str = "orchard-fenicsx"
    skip_dev_tests: bool = False
    skip_fenicsx_tests: bool = False
    skip_python_demo_suite: bool = False

    @classmethod
    def from_environment(cls, repo_root: Path | None = None) -> "FullValidationConfig":
        resolved_repo_root = (repo_root or DEFAULT_REPO_ROOT).resolve()
        build_dir = Path(os.environ.get("BUILD_DIR", str(resolved_repo_root / "build")))
        validation_dir = Path(
            os.environ.get("VALIDATION_DIR", str(build_dir / "validation"))
        )
        return cls(
            repo_root=resolved_repo_root,
            build_dir=build_dir,
            validation_dir=validation_dir,
            orchard_dev_env=os.environ.get("ORCHARD_DEV_ENV", "orchard-dev"),
            orchard_fenicsx_env=os.environ.get("ORCHARD_FENICSX_ENV", "orchard-fenicsx"),
            skip_dev_tests=os.environ.get("SKIP_DEV_TESTS", "0") == "1",
            skip_fenicsx_tests=os.environ.get("SKIP_FENICSX_TESTS", "0") == "1",
            skip_python_demo_suite=os.environ.get("SKIP_PYTHON_DEMO_SUITE", "0") == "1",
        )


@dataclass(frozen=True)
class FullValidationOutputs:
    build_dir: Path
    validation_dir: Path
    executed_steps: list[FullValidationStep]


def _require_command(command: str) -> None:
    if shutil.which(command) is not None:
        return
    raise RuntimeError(f"Missing required command: {command}")


def build_full_validation_steps(config: FullValidationConfig) -> list[FullValidationStep]:
    steps: list[FullValidationStep] = []

    if not config.skip_dev_tests:
        steps.append(
            FullValidationStep(
                label="Run orchard-dev Orchard FEM integration tests",
                command=[
                    "conda",
                    "run",
                    "-n",
                    config.orchard_dev_env,
                    "python",
                    "-m",
                    "orchard_fem",
                    "verify",
                    "--skip-verification",
                    "--skip-demo-suite",
                ],
            )
        )

    if not config.skip_fenicsx_tests or not config.skip_python_demo_suite:
        command = [
            "conda",
            "run",
            "-n",
            config.orchard_fenicsx_env,
            "python",
            "-m",
            "orchard_fem",
            "verify",
            "--skip-integration",
            "--output-dir",
            str(config.validation_dir / "python"),
        ]
        if config.skip_fenicsx_tests:
            command.append("--skip-verification")
        if config.skip_python_demo_suite:
            command.append("--skip-demo-suite")
        steps.append(
            FullValidationStep(
                label="Run orchard-fenicsx Orchard FEM verification and demo workflow",
                command=command,
            )
        )

    return steps


def _print_step(message: str) -> None:
    print(flush=True)
    print(f"==> {message}", flush=True)


def run_full_validation(config: FullValidationConfig) -> FullValidationOutputs:
    _require_command("python3")
    _require_command("conda")

    config.validation_dir.mkdir(parents=True, exist_ok=True)
    executed_steps: list[FullValidationStep] = []

    for step in build_full_validation_steps(config):
        _print_step(step.label)
        completed = subprocess.run(
            step.command,
            cwd=config.repo_root,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Full validation step failed with exit code "
                f"{completed.returncode}: {' '.join(step.command)}"
            )
        executed_steps.append(step)

    print(flush=True)
    print("Full validation completed.", flush=True)
    print("Artifacts:", flush=True)
    print(f"  build directory: {config.build_dir}", flush=True)
    print(f"  validation outputs: {config.validation_dir}", flush=True)

    return FullValidationOutputs(
        build_dir=config.build_dir,
        validation_dir=config.validation_dir,
        executed_steps=executed_steps,
    )
