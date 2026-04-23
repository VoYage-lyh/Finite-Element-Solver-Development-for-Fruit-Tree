from __future__ import annotations

from pathlib import Path

from orchard_fem.automation import FullValidationConfig, build_full_validation_steps


def test_full_validation_plan_builds_both_environment_steps(tmp_path) -> None:
    config = FullValidationConfig(
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        validation_dir=tmp_path / "build" / "validation",
        orchard_dev_env="orchard-dev",
        orchard_fenicsx_env="orchard-fenicsx",
    )

    steps = build_full_validation_steps(config)

    assert [step.label for step in steps] == [
        "Run orchard-dev Python integration tests",
        "Run orchard-fenicsx Python verification and demo workflow",
    ]
    assert steps[0].command[:7] == [
        "conda",
        "run",
        "-n",
        "orchard-dev",
        "python",
        "-m",
        "orchard_fem",
    ]
    assert "--skip-verification" in steps[0].command
    assert "--skip-demo-suite" in steps[0].command
    assert steps[1].command[:7] == [
        "conda",
        "run",
        "-n",
        "orchard-fenicsx",
        "python",
        "-m",
        "orchard_fem",
    ]
    assert "--skip-integration" in steps[1].command
    assert str(tmp_path / "build" / "validation" / "python") in steps[1].command


def test_full_validation_plan_honors_skip_flags(tmp_path) -> None:
    config = FullValidationConfig(
        repo_root=tmp_path,
        build_dir=tmp_path / "build",
        validation_dir=tmp_path / "build" / "validation",
        skip_dev_tests=True,
        skip_fenicsx_tests=True,
        skip_python_demo_suite=True,
    )

    steps = build_full_validation_steps(config)

    assert steps == []
