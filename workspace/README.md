# Local workspace

This directory contains machine-local data and generated artifacts. Everything
below it, except this file, is ignored by Git.

```text
workspace/
├── data/
│   ├── raw/orchard_photos/       Original field photographs
│   └── annotations/wood/         Visible-wood masks and annotation metadata
├── tree_models/                  User-generated and experimental tree models
├── models/                       Model checkpoints such as wood_seg.pt and sam2_t.pt
├── outputs/                      Solver, vision, calibration, and paper outputs
├── cache/                        Rebuildable computation caches
└── references/                   Large local manuals and reference material
```

Set `ORCHARD_WORKSPACE=/absolute/path` to keep these artifacts elsewhere. A
relative value is resolved from the repository root. CLI path options continue
to override these defaults.
