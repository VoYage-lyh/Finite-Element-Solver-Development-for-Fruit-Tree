from pathlib import Path

from orchard_fem.workspace import REPOSITORY_ROOT, example_trees_dir, workspace_paths


def test_default_workspace_is_inside_repository(monkeypatch) -> None:
    monkeypatch.delenv("ORCHARD_WORKSPACE", raising=False)
    paths = workspace_paths()
    assert paths.root == (REPOSITORY_ROOT / "workspace").resolve()
    assert paths.wood_annotations == paths.root / "data" / "annotations" / "wood"
    assert paths.wood_checkpoint == paths.root / "models" / "wood_seg.pt"
    assert paths.harvest_runs == paths.root / "outputs" / "harvest_runs"


def test_environment_workspace_override_supports_absolute_path(
    monkeypatch, tmp_path: Path
) -> None:
    custom = tmp_path / "orchard-local"
    monkeypatch.setenv("ORCHARD_WORKSPACE", str(custom))
    assert workspace_paths().root == custom.resolve()


def test_explicit_workspace_root_takes_precedence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORCHARD_WORKSPACE", str(tmp_path / "ignored"))
    paths = workspace_paths(tmp_path / "explicit")
    assert paths.root == (tmp_path / "explicit").resolve()


def test_example_trees_are_version_controlled() -> None:
    assert (example_trees_dir() / "tree_1.json").is_file()
    assert (example_trees_dir() / "tree_5.json").is_file()
