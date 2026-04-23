from __future__ import annotations

import warnings

from orchard_fem.io.model_loader import load_orchard_model as _load_orchard_model


def load_orchard_model(file_path):
    warnings.warn(
        "orchard_fem.io.legacy_loader is deprecated. "
        "Import load_orchard_model from orchard_fem.io.model_loader instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _load_orchard_model(file_path)

__all__ = ["load_orchard_model"]
