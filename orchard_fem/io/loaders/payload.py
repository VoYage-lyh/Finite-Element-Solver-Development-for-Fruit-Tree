from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = (
    "materials",
    "branches",
    "excitation",
    "analysis",
)


def load_model_payload(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Orchard model is missing required keys: {', '.join(missing)}")

    return payload
