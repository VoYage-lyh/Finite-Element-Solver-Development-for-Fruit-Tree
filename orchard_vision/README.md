# orchard_vision — monocular tree photo → branch skeleton

Turns a **single ordinary RGB photo** of a fruit tree into the 3D branch skeleton
the vibration solver consumes, labelling **trunk / primary / secondary / tertiary**
branches along the way. The classical baseline is pure NumPy / SciPy /
scikit-image and needs no GPU; the optional EfficientFormer and SAM front-ends use
PyTorch. No stage requires dolfinx or LiDAR.

```
photo ─▶ segmentation ─▶ skeleton_graph ─▶ branch_ordering ─▶ lift_3d ─▶ export_skeleton
        (branch mask)     (pixel graph)      (level-labelled)   (metric)   (JSON)
                                                                              │
                                              python -m orchard_fem import-skeleton ─▶ FEM / PINN
```

## Usage

```bash
# one photo (add --tree-height-m for correct metric scale)
python -m orchard_vision.cli path/to/tree.jpg --tree-height-m 3.0

# a whole batch
python -m orchard_vision.cli path/to/photos/*.jpg --out-dir results/vision
```

Each input writes two files to `--out-dir` (default `results/vision/`):

* `<name>.skeleton.json` — feed straight to `orchard_fem import-skeleton`;
* `<name>.overlay.png` — the ordered branches drawn over the photo, coloured by
  order (trunk red · primary orange · secondary green · tertiary blue), for a
  quick sanity check.

End-to-end into the solver:

```bash
python -m orchard_vision.cli path/to/tree.jpg --out-dir results/vision
python -m orchard_fem import-skeleton results/vision/tree.skeleton.json build/tree.json
```

## Manual correction in Harvest Console

Launch the existing harvest front-end and use **① Tree Model → Photo skeleton
extraction / manual correction**:

```bash
python -m orchard_fem.actuator.harvest_console
```

The embedded editor supports hierarchy/reparenting, node dragging, branch
extension/trimming, evidence-mask painting, and adding image-supported branches.
It saves a reloadable ``*.skeleton-project.json`` plus compressed array sidecar.
Export converts the edited skeleton to a solver-ready model and makes that model
active in Harvest Console; recommendations from the previous geometry are cleared.

## Pipeline stages (one module each)

| Module | Role |
| --- | --- |
| `segmentation.py` | `BranchSegmenter` protocol + `ClassicalSegmenter` (foreground + multiscale Sato ridge). **Swap in a learned model here.** |
| `skeleton_graph.py` | Skeletonise the mask, sample per-pixel radius (distance transform), trace an 8-connected node/edge graph. |
| `branch_ordering.py` | Resolve junctions by straightest-continuation, find the trunk base, BFS-assign branch orders. |
| `lift_3d.py` | Monocular planar 2D→3D: image-x→world-x, image-up→world-z (gravity), y=0; pixel→metre scale from tree height. |
| `export_skeleton.py` | Emit the `import-skeleton` JSON (`id, parent_branch_id, level, points, outer_radius_root/tip`). |
| `pipeline.py` / `cli.py` | Orchestration, level-coloured overlay, command line. |

Everything is configured through `PipelineConfig`, and the segmenter is injected,
so a learned front-end can replace the classical one without touching the
geometry post-processing.

## What works and what does not

* **Reliable:** trunk and primary/secondary branches in the clearly-visible woody
  region; topology (parent links), branch order, root at the base, radii that
  taper root→tip. This is exactly the structure the harvest solver needs most
  (clamp candidates are trunk + primaries).
* **Weak by design:** the classical segmenter cannot see woody branches **behind
  dense foliage**, so canopy structure is under-segmented and finer twigs are
  dropped (`--max-level` caps the traced order at 3). Metric scale is a single
  planar assumption from tree height — **out-of-plane geometry is approximate**.

## Wood-segmentation training workflow (to see through canopy)

The classical/SAM masks can't cleanly separate wood from leaves in the canopy —
that needs a **trained** wood segmenter (EfficientFormer backbone + a light head).
Training needs labels made directly on the raw field photographs (not only on
white-background cut-outs).  The annotator selects one target tree per image,
records its trunk root, and uses SAM 2 only as an editable proposal tool:

```bash
# raw orchard photographs; quote the glob if the shell should not expand it
python -m orchard_vision.annotate_wood trees/tree*.jpg \
    --out-dir datasets/wood_field \
    --sam-checkpoint weights/sam2_t.pt --sam-device cuda:1
```

For every source image:

1. `r`: drag a tight target-tree ROI. Labels outside it are removed and that
   region is recorded as "ignore" rather than training background, avoiding
   false-negative labels on neighbouring trees.
2. `t`: click the visible base of the target trunk. This anchors later connected
   skeleton selection.
3. `b`: drag a tight box around one visible limb; `c` commits the red SAM proposal
   and `x` discards it.
4. `m`: left-drag to add missed visible wood and right-drag to erase leaves or
   background. `[` / `]` changes brush size; `z` / `y` undo and redo.
5. `s` saves, `n` saves and advances, and `q` saves and closes.

The tool automatically resumes an existing annotation. It writes
`datasets/wood_field/<name>.png`, `<name>_wood.png`, and `<name>_wood.json`; the
JSON sidecar preserves the original image shape, working scale, target ROI, and
trunk-root point. Label only visible wood belonging to the target tree—do not
guess fully occluded branches.

### EfficientFormer training and evaluation

Install both image-processing and learned-model dependencies:

```bash
python -m pip install -e ".[vision,ml]"
```

Name multiple views of the same physical tree with a double underscore, for
example `tree012__view01.jpg` and `tree012__view02.jpg`. The text before `__` is
the tree group, and every view/crop from that group stays in exactly one of the
training, validation, or test splits.

```bash
python -m orchard_vision.wood_seg train \
    --data datasets/wood_field \
    --out weights/wood_seg.pt \
    --device cuda:1 \
    --epochs 160 --batch-size 8 \
    --tile-size 224 --overlap 64 \
    --freeze-epochs 12 --patience 25

python -m orchard_vision.wood_seg evaluate \
    --data datasets/wood_field \
    --checkpoint weights/wood_seg.pt \
    --split test --device cuda:1
```

Training reads each JSON ROI, ignores pixels outside it, creates overlapping
224×224 tiles only after tree-level splitting, balances empty/positive tiles,
derives the positive-class weight from the training set, and applies colour,
brightness, rotation, and horizontal-flip augmentation. The EfficientFormer
backbone is initially frozen, then fine-tuned with a smaller learning rate. The
best validation-Dice checkpoint is saved and early stopping prevents needless
overfitting; the checkpoint also stores the exact split group names. A companion
`weights/wood_seg.metrics.json` records every epoch, the best epoch, configuration,
and held-out metrics for later plots and thesis tables.

Inference uses overlapping 224×224 windows and weighted probability blending,
preserving thin branches and rectangular image geometry instead of shrinking an
entire photograph to 224×224. It is selected automatically by Harvest Console
when `weights/wood_seg.pt` and its dependencies exist, or explicitly from the CLI:

```bash
python -m orchard_vision.cli trees/tree1.jpg --segmenter wood \
    --wood-checkpoint weights/wood_seg.pt --out-dir results/vision_field
```

The skeleton and ordering stages downstream are unchanged, and residual errors
remain correctable in Harvest Console.

## Other next steps

* **Metric scale / out-of-plane** — add a scale marker (ArUco) or a monocular
  depth map (Depth Anything V2) in `lift_3d.py` for true 3D.
* **SAM 2 scaffold** — `--segmenter sam2 --max-level 1` already gives a clean
  trunk + primary scaffold today (needs a GPU); secondaries await the trained head.
