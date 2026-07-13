"""Central paths for local data, models, caches, and generated outputs.

The source tree stays reproducible while large or machine-specific artifacts live
under one ignored workspace.  Set ``ORCHARD_WORKSPACE`` to use another location;
relative overrides are resolved from the repository root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_VARIABLE = "ORCHARD_WORKSPACE"


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved paths for machine-local Orchard FEM artifacts."""

    root: Path

    @property
    def raw_photos(self) -> Path:
        return self.root / "data" / "raw" / "orchard_photos"

    @property
    def wood_annotations(self) -> Path:
        return self.root / "data" / "annotations" / "wood"

    @property
    def tree_models(self) -> Path:
        return self.root / "tree_models"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def wood_checkpoint(self) -> Path:
        return self.models / "wood_seg.pt"

    @property
    def sam_checkpoint(self) -> Path:
        return self.models / "sam2_t.pt"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def references(self) -> Path:
        return self.root / "references"

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def ds5l1_calibration(self) -> Path:
        return self.config / "ds5l1_freq_calib.json"

    @property
    def harvest_runs(self) -> Path:
        return self.outputs / "harvest_runs"


def workspace_paths(root: str | Path | None = None) -> WorkspacePaths:
    """Return workspace paths using an explicit root or ``ORCHARD_WORKSPACE``."""

    configured = root if root is not None else os.environ.get(ENVIRONMENT_VARIABLE, "workspace")
    resolved = Path(configured).expanduser()
    if not resolved.is_absolute():
        resolved = REPOSITORY_ROOT / resolved
    return WorkspacePaths(root=resolved.resolve())


def example_trees_dir() -> Path:
    """Return the version-controlled example-tree directory."""

    return REPOSITORY_ROOT / "examples" / "trees"


def display_path(path: str | Path) -> str:
    """Format a path relative to the repository when possible."""

    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)
