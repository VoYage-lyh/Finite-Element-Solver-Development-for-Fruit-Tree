from __future__ import annotations

from math import sqrt


def root_mean_square_error(reference: list[float], prediction: list[float]) -> float:
    if len(reference) != len(prediction):
        raise ValueError("RMSE requires sequences of equal length")
    if not reference:
        return 0.0

    squared_sum = 0.0
    for ref_value, pred_value in zip(reference, prediction):
        difference = pred_value - ref_value
        squared_sum += difference * difference
    return sqrt(squared_sum / len(reference))


def relative_l2_error(reference: list[float], prediction: list[float], eps: float = 1.0e-12) -> float:
    if len(reference) != len(prediction):
        raise ValueError("Relative L2 error requires sequences of equal length")

    numerator = 0.0
    denominator = 0.0
    for ref_value, pred_value in zip(reference, prediction):
        difference = pred_value - ref_value
        numerator += difference * difference
        denominator += ref_value * ref_value

    return sqrt(numerator) / max(sqrt(denominator), eps)


def r2_score(reference: list[float], prediction: list[float], eps: float = 1.0e-12) -> float:
    if len(reference) != len(prediction):
        raise ValueError("R2 requires sequences of equal length")
    if not reference:
        return 1.0

    mean_reference = sum(reference) / len(reference)
    residual_sum = 0.0
    total_sum = 0.0
    for ref_value, pred_value in zip(reference, prediction):
        residual = ref_value - pred_value
        residual_sum += residual * residual
        centered = ref_value - mean_reference
        total_sum += centered * centered

    return 1.0 - (residual_sum / max(total_sum, eps))
