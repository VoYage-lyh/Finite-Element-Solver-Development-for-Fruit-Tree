from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorMetrics:
    modal_error: float = 0.0
    response_rms_error: float = 0.0
    peak_error: float = 0.0
    runtime_speedup: float = 1.0
