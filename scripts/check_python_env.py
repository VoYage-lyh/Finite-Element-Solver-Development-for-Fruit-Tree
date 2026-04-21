from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CONDA_ENV_PATH = REPO_ROOT / "config" / "fenicsx_pinn_environment.yml"
PREFERRED_PYPROJECT_GROUPS = ("project", "dev", "viz", "ml", "ubuntu-test")

PACKAGE_IMPORT_ALIASES = {
    "pytest-cov": "pytest_cov",
    "pyyaml": "yaml",
    "pytorch": "torch",
}

REQUIRED_TOOL_COMMANDS = {
    "cmake": "cmake",
    "ctest": "ctest",
    "g++": "g++",
    "gcc": "gcc",
    "git": "git",
}

OPTIONAL_TOOL_COMMANDS = {
    "ninja": "ninja",
    "mpiexec": "mpiexec",
}


def normalize_requirement_name(requirement: str) -> str:
    requirement = requirement.strip()
    if not requirement or requirement.startswith("#"):
        return ""
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    return match.group(1).lower() if match else ""


def module_name_for_package(package_name: str) -> str:
    return PACKAGE_IMPORT_ALIASES.get(package_name.lower(), package_name.replace("-", "_"))


def is_module_available(package_name: str) -> bool:
    module_name = module_name_for_package(package_name)
    return importlib.util.find_spec(module_name) is not None


def parse_pyproject_dependencies(path: Path) -> dict[str, list[str]]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    project = payload.get("project", {})
    groups: dict[str, list[str]] = {}
    groups["project"] = [normalize_requirement_name(item) for item in project.get("dependencies", [])]

    for group_name, requirements in project.get("optional-dependencies", {}).items():
        groups[group_name] = [normalize_requirement_name(item) for item in requirements]

    return {group: [item for item in items if item] for group, items in groups.items()}


def parse_conda_dependencies(path: Path) -> list[str]:
    packages: list[str] = []
    in_dependencies = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        if not stripped.startswith("- "):
            continue
        value = stripped[2:].strip()
        if value == "pip:":
            continue
        name = normalize_requirement_name(value)
        if name and name != "python":
            packages.append(name)
    return packages


def ordered_pyproject_groups(groups: dict[str, list[str]]) -> list[str]:
    ordered: list[str] = []
    for name in PREFERRED_PYPROJECT_GROUPS:
        if name in groups:
            ordered.append(name)

    for name in sorted(groups):
        if name not in ordered:
            ordered.append(name)
    return ordered


def print_group_status(title: str, packages: list[str]) -> list[str]:
    print(title)
    missing: list[str] = []
    if not packages:
        print("  (none)")
        return missing
    for package in packages:
        status = "yes" if is_module_available(package) else "no"
        print(f"  {package}: {status}")
        if status == "no":
            missing.append(package)
    return missing


def print_tool_status(title: str, commands: dict[str, str]) -> list[str]:
    print(title)
    missing: list[str] = []
    for name, command in commands.items():
        available = shutil.which(command) is not None
        print(f"  {name}: {'yes' if available else 'no'}")
        if not available:
            missing.append(name)
    return missing


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print()

    pyproject_groups = parse_pyproject_dependencies(PYPROJECT_PATH)
    conda_packages = parse_conda_dependencies(CONDA_ENV_PATH)

    all_missing: dict[str, list[str]] = {}
    for group_name in ordered_pyproject_groups(pyproject_groups):
        missing = print_group_status(f"pyproject::{group_name}", pyproject_groups.get(group_name, []))
        if missing:
            all_missing[f"pyproject::{group_name}"] = missing
        print()

    conda_missing = print_group_status("conda::fenicsx_pinn_environment", conda_packages)
    if conda_missing:
        all_missing["conda::fenicsx_pinn_environment"] = conda_missing
    print()

    tool_missing = print_tool_status("External tools (required)", REQUIRED_TOOL_COMMANDS)
    if tool_missing:
        all_missing["tools"] = tool_missing
    print()

    print_tool_status("External tools (optional / backend-specific)", OPTIONAL_TOOL_COMMANDS)
    print()

    print("Missing summary")
    if not all_missing:
        print("  none")
        return 0

    for group_name, missing_packages in all_missing.items():
        print(f"  {group_name}: {', '.join(missing_packages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
