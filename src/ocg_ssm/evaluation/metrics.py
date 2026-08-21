"""Explicitly defined metrics with consistent graph direction conventions."""

from __future__ import annotations

import math

import numpy as np

from ocg_ssm.discovery.graph import threshold_graph


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def gaussian_crps(actual: np.ndarray, mean: np.ndarray, std: np.ndarray | float) -> float:
    """Average Gaussian CRPS; lower is better and values are nonnegative."""
    observed = np.asarray(actual, dtype=float)
    location = np.asarray(mean, dtype=float)
    scale = np.asarray(std, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("Gaussian CRPS requires a strictly positive standard deviation")
    standardized = (observed - location) / scale
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(standardized / math.sqrt(2.0)))
    density = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
    scores = scale * (standardized * (2.0 * cdf - 1.0) + 2.0 * density - 1.0 / math.sqrt(math.pi))
    return float(np.mean(scores))


def graph_metrics(
    truth: np.ndarray, estimate: np.ndarray, *, threshold: float = 0.10
) -> dict[str, float | int]:
    """Evaluate directed off-diagonal edges; a reversal costs two SHD edits."""
    actual = threshold_graph(truth, threshold, include_diagonal=False).astype(bool)
    predicted = threshold_graph(estimate, threshold, include_diagonal=False).astype(bool)
    mask = ~np.eye(actual.shape[0], dtype=bool)
    true_positive = int(np.sum(actual & predicted & mask))
    false_positive = int(np.sum(~actual & predicted & mask))
    false_negative = int(np.sum(actual & ~predicted & mask))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "shd": false_positive + false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
    }


def recovery_time(losses: np.ndarray, change_point: int, reference: float, tolerance: float = 0.15) -> int | None:
    """Return steps until loss recovers below ``reference * (1 + tolerance)``."""
    values = np.asarray(losses, dtype=float)
    for index in range(change_point, len(values)):
        if values[index] <= reference * (1.0 + tolerance):
            return index - change_point
    return None


def cumulative_regret(model_losses: np.ndarray, comparator_losses: np.ndarray) -> np.ndarray:
    model = np.asarray(model_losses, dtype=float)
    comparator = np.asarray(comparator_losses, dtype=float)
    if model.shape != comparator.shape:
        raise ValueError("Regret requires loss sequences with matching shapes")
    return np.cumsum(model - comparator)
