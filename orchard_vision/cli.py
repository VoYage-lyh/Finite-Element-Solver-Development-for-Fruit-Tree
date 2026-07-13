"""Command-line entry: tree photo(s) → skeleton JSON (+ overlay PNG).

Example::

    python -m orchard_vision.cli path/to/tree.jpg --tree-height-m 3.5
    python -m orchard_vision.cli path/to/photos/*.jpg

Each input writes ``<name>.skeleton.json`` (feed to ``orchard_fem
import-skeleton``) and ``<name>.overlay.png`` (level-coloured sanity check).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchard_fem.workspace import workspace_paths
from orchard_vision.pipeline import PhotoToSkeletonPipeline, PipelineConfig


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    paths = workspace_paths()
    parser = argparse.ArgumentParser(description="Monocular tree photo → branch skeleton JSON.")
    parser.add_argument("inputs", nargs="+", help="Photo path(s), e.g. path/to/tree.jpg")
    parser.add_argument(
        "--out-dir",
        default=str(paths.outputs / "vision"),
        help="Output directory",
    )
    parser.add_argument("--tree-height-m", type=float, default=3.0, help="Real tree height for scale (fallback)")
    parser.add_argument("--trunk-diameter-m", type=float, default=None,
                        help="Measured trunk base diameter (m) → metric scale (preferred over tree height)")
    parser.add_argument("--max-dim", type=int, default=1024, help="Downscale longer side to this")
    parser.add_argument("--max-turn-deg", type=float, default=80.0, help="Junction continuation tolerance")
    parser.add_argument("--min-spur-px", type=float, default=12.0, help="Prune leaf hairs shorter than this")
    parser.add_argument("--max-level", type=int, default=2, help="Deepest branch order to trace (trunk=0)")
    parser.add_argument("--min-primary-height-m", type=float, default=0.15,
                        help="Forbid primary branches attaching below this height (m) above the base")
    parser.add_argument("--segmenter", choices=("classical", "sam2", "wood"), default="classical",
                        help="Branch-mask front-end (a GPU is recommended for sam2/wood)")
    parser.add_argument(
        "--sam-checkpoint",
        default=str(paths.sam_checkpoint),
        help="SAM 2 checkpoint path",
    )
    parser.add_argument("--sam-device", default="cuda:1", help="Device for SAM 2 (e.g. cuda:1, cpu)")
    parser.add_argument("--wood-checkpoint", default=str(paths.wood_checkpoint),
                        help="Trained EfficientFormer wood-seg checkpoint (--segmenter wood)")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay PNGs")
    return parser.parse_args(argv)


def _build_segmenter(args: argparse.Namespace):
    if args.segmenter == "sam2":
        from orchard_vision.segmentation_sam2 import Sam2Segmenter  # lazy: keeps torch optional

        return Sam2Segmenter(checkpoint=args.sam_checkpoint, device=args.sam_device)
    if args.segmenter == "wood":
        from orchard_vision.wood_seg import WoodSegmenter  # lazy: keeps torch optional

        return WoodSegmenter(checkpoint=args.wood_checkpoint, device=args.sam_device)
    from orchard_vision.segmentation import ClassicalSegmenter

    return ClassicalSegmenter()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = PipelineConfig(
        tree_height_m=args.tree_height_m,
        trunk_diameter_m=args.trunk_diameter_m,
        max_dimension=args.max_dim,
        max_turn_deg=args.max_turn_deg,
        min_spur_px=args.min_spur_px,
        max_level=args.max_level,
        min_primary_height_m=args.min_primary_height_m,
        segmenter=_build_segmenter(args),
    )
    pipeline = PhotoToSkeletonPipeline(config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw in args.inputs:
        path = Path(raw)
        try:
            result = pipeline.run(path)
        except Exception as error:  # noqa: BLE001 - CLI reports and continues
            print(f"[skip] {path.name}: {error}")
            continue

        json_path = out_dir / f"{result.name}.skeleton.json"
        json_path.write_text(
            json.dumps(result.payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts = ", ".join(f"L{level}:{n}" for level, n in result.level_counts().items())
        print(f"[ok]   {path.name}: {len(result.branches)} branches ({counts}) → {json_path}")

        if not args.no_overlay:
            pipeline.save_overlay(result, out_dir / f"{result.name}.overlay.png")
            pipeline.save_instance_overlay(result, out_dir / f"{result.name}.instances.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
