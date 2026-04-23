from __future__ import annotations

import warnings

from orchard_fem.io.model_payload import (
    REQUIRED_TOP_LEVEL_KEYS,
    build_topology_from_model_payload,
    load_model_payload,
)


def load_legacy_model(file_path):
    warnings.warn(
        "orchard_fem.io.json_schema.load_legacy_model is deprecated. "
        "Use orchard_fem.io.model_payload.load_model_payload instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_model_payload(file_path)


def build_topology_from_legacy_model(payload):
    warnings.warn(
        "orchard_fem.io.json_schema.build_topology_from_legacy_model is deprecated. "
        "Use orchard_fem.io.model_payload.build_topology_from_model_payload instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_topology_from_model_payload(payload)
