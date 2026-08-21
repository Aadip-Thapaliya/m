"""Chronological splits and strictly past-conditioned forecasting windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChronologicalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_end: int
    validation_end: int


def chronological_split(
    observations: np.ndarray,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> ChronologicalSplit:
    """Split in time order; never shuffle or normalize on future observations."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Split fractions must be in their respective valid ranges")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition")
    values = np.asarray(observations)
    train_end = int(len(values) * train_fraction)
    validation_end = int(len(values) * (train_fraction + validation_fraction))
    return ChronologicalSplit(
        train=values[:train_end],
        validation=values[train_end:validation_end],
        test=values[validation_end:],
        train_end=train_end,
        validation_end=validation_end,
    )


def sliding_windows(
    observations: np.ndarray, context: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return contexts ``x[t-context:t]`` and targets ``x[t:t+horizon]``."""
    values = np.asarray(observations, dtype=float)
    if context < 1 or horizon < 1 or context + horizon > len(values):
        raise ValueError("Context and horizon must fit inside the observations")
    inputs, targets = [], []
    for stop in range(context, len(values) - horizon + 1):
        inputs.append(values[stop - context : stop])
        targets.append(values[stop : stop + horizon])
    return np.stack(inputs), np.stack(targets)
