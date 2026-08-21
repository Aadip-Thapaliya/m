"""Simple, inspectable reference models for a leakage-safe first benchmark."""

from __future__ import annotations

import numpy as np

from ocg_ssm.discovery.online_var import OnlineVARDiscovery


class PersistenceForecaster:
    """Predict the next observation as the most recently revealed observation."""

    def predict(self, previous: np.ndarray) -> np.ndarray:
        return np.asarray(previous, dtype=float).copy()

    def update(self, previous: np.ndarray, observed: np.ndarray) -> None:
        del previous, observed


class StaticVARForecaster:
    """Ridge-regularized lag-one VAR fit once on the training prefix."""

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = ridge
        self.coefficients: np.ndarray | None = None

    def fit(self, observations: np.ndarray) -> "StaticVARForecaster":
        values = np.asarray(observations, dtype=float)
        if len(values) < 2:
            raise ValueError("A static VAR requires at least two observations")
        predictors, targets = values[:-1], values[1:]
        penalty = self.ridge * np.eye(predictors.shape[1])
        self.coefficients = np.linalg.solve(predictors.T @ predictors + penalty, predictors.T @ targets)
        return self

    def predict(self, previous: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Fit the static VAR before forecasting")
        return np.asarray(previous, dtype=float) @ self.coefficients

    def update(self, previous: np.ndarray, observed: np.ndarray) -> None:
        del previous, observed


class OnlineVARForecaster:
    """Forecast using the current RLS graph and update only after target reveal."""

    def __init__(self, discovery: OnlineVARDiscovery) -> None:
        self.discovery = discovery

    def predict(self, previous: np.ndarray) -> np.ndarray:
        return self.discovery.predict(previous)

    def update(self, previous: np.ndarray, observed: np.ndarray) -> None:
        self.discovery.update(previous, observed)
