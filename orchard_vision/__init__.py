"""Monocular tree-photo → ordered branch skeleton → solver-ready JSON.

This package turns a single RGB photograph of a fruit tree into the 3D branch
skeleton that the vibration solver consumes (see
:mod:`orchard_fem.io.skeleton_import`). It is split into small, replaceable
stages so the rough classical segmenter can later be swapped for a learned model
(EfficientFormer semantic head, SAM 2, …) without touching the geometry
post-processing that computes the trunk / primary / secondary / … hierarchy::

    photo ─▶ segmentation ─▶ skeleton_graph ─▶ branch_ordering ─▶ lift_3d
            (branch mask)    (pixel graph)      (level-labelled)   (metric xyz)
                                                                       │
                                                          export_skeleton (JSON)
                                                                       │
                                              orchard_fem import-skeleton ─▶ FEM

The classical front-end is pure NumPy / SciPy / scikit-image; optional learned
EfficientFormer and SAM front-ends use PyTorch. Geometry post-processing never
depends on dolfinx, so the vision stage remains independently deployable.
"""
from orchard_vision.pipeline import (
    PhotoToSkeletonPipeline,
    PipelineConfig,
    PipelineResult,
)

__all__ = ["PhotoToSkeletonPipeline", "PipelineConfig", "PipelineResult"]
