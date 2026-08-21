"""Leakage-safe readers for the unmodified official ETT-small CSV datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ETT_FEATURES = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")


@dataclass(frozen=True)
class ETTDataset:
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    columns: tuple[str, ...]


@dataclass(frozen=True)
class TrainingPrefixScaler:
    mean: np.ndarray
    standard_deviation: np.ndarray

    @classmethod
    def fit(cls, training_prefix: np.ndarray) -> "TrainingPrefixScaler":
        values = np.asarray(training_prefix, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("A scaler requires a non-empty two-dimensional training prefix")
        deviation = values.std(axis=0)
        deviation[deviation < 1e-12] = 1.0
        return cls(values.mean(axis=0), deviation)

    def transform(self, observations: np.ndarray) -> np.ndarray:
        return (np.asarray(observations, dtype=float) - self.mean) / self.standard_deviation

    def inverse_transform(self, standardized: np.ndarray) -> np.ndarray:
        return np.asarray(standardized, dtype=float) * self.standard_deviation + self.mean


def load_ett(path: str | Path, *, max_rows: int | None = None) -> ETTDataset:
    source = Path(path)
    frame = pd.read_csv(source, nrows=max_rows)
    expected = {"date", *ETT_FEATURES}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"ETT file is missing required columns: {sorted(missing)}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    if not timestamps.is_monotonic_increasing:
        raise ValueError("ETT timestamps must be in chronological order")
    values = frame.loc[:, ETT_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ETT numeric values must be finite; define causal missing-value handling first")
    return ETTDataset(timestamps=timestamps, values=values, columns=ETT_FEATURES)
