"""Predict-then-reveal-then-update streaming evaluation without target leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ocg_ssm.evaluation.metrics import mae, mse


class OnlineForecaster(Protocol):
    def predict(self, previous: np.ndarray) -> np.ndarray: ...

    def update(self, previous: np.ndarray, observed: np.ndarray) -> None: ...


@dataclass(frozen=True)
class PrequentialResult:
    actual: np.ndarray
    predictions: np.ndarray
    squared_losses: np.ndarray
    mean_squared_error: float
    mean_absolute_error: float


def evaluate_prequential(
    model: OnlineForecaster, observations: np.ndarray, *, start: int
) -> PrequentialResult:
    values = np.asarray(observations, dtype=float)
    if start < 1 or start >= len(values):
        raise ValueError("Streaming evaluation must begin inside the observation sequence")
    predictions, targets, losses = [], [], []
    for index in range(start, len(values)):
        previous = values[index - 1]
        prediction = np.asarray(model.predict(previous), dtype=float).copy()
        actual = values[index]
        predictions.append(prediction)
        targets.append(actual.copy())
        losses.append(float(np.mean((actual - prediction) ** 2)))
        model.update(previous, actual)
    actual_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    return PrequentialResult(
        actual=actual_array,
        predictions=prediction_array,
        squared_losses=np.asarray(losses),
        mean_squared_error=mse(actual_array, prediction_array),
        mean_absolute_error=mae(actual_array, prediction_array),
    )
