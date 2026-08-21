"""Shared-covariance recursive least squares for streaming lagged discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocg_ssm.discovery.graph import soft_threshold


@dataclass(frozen=True)
class OnlineUpdate:
    prediction: np.ndarray
    residual: np.ndarray
    score: float
    adjacency: np.ndarray


class OnlineVARDiscovery:
    """Estimate a lag-one graph with one shared RLS covariance matrix.

    With ``D`` variables and one lag, computing the gain and updating the
    coefficient matrix both cost ``O(D²)`` per observation. This tracks linear
    predictive relationships under the stated assumptions; observational graph
    estimates are not automatically identified causal effects.
    """

    def __init__(
        self,
        n_variables: int,
        *,
        forgetting_factor: float = 0.985,
        ridge: float = 8.0,
        sparsity_threshold: float = 0.0,
    ) -> None:
        if n_variables < 1:
            raise ValueError("At least one variable is required")
        if not 0 < forgetting_factor <= 1:
            raise ValueError("The forgetting factor must be in (0, 1]")
        if ridge <= 0 or sparsity_threshold < 0:
            raise ValueError("Ridge must be positive and sparsity nonnegative")
        self.n_variables = n_variables
        self.forgetting_factor = forgetting_factor
        self.ridge = ridge
        self.sparsity_threshold = sparsity_threshold
        self.coefficients = np.zeros((n_variables, n_variables), dtype=float)
        self.covariance = np.eye(n_variables, dtype=float) / ridge
        self.n_updates = 0

    @property
    def adjacency(self) -> np.ndarray:
        """Return coefficients in the ``[target, parent]`` convention."""
        return self.coefficients.T.copy()

    def predict(self, previous: np.ndarray) -> np.ndarray:
        values = np.asarray(previous, dtype=float)
        if values.shape != (self.n_variables,):
            raise ValueError(f"Expected a vector with {self.n_variables} variables")
        return values @ self.coefficients

    def update(self, previous: np.ndarray, observed: np.ndarray) -> OnlineUpdate:
        """Score the old graph, then update it with the newly revealed target."""
        features = np.asarray(previous, dtype=float)
        target = np.asarray(observed, dtype=float)
        prediction = self.predict(features)
        if target.shape != prediction.shape:
            raise ValueError("Observed target does not match the configured dimensions")
        residual = target - prediction

        projected = self.covariance @ features
        denominator = self.forgetting_factor + float(features @ projected)
        gain = projected / denominator
        self.coefficients += np.outer(gain, residual)
        self.covariance = (
            self.covariance - np.outer(gain, features @ self.covariance)
        ) / self.forgetting_factor
        self.covariance = (self.covariance + self.covariance.T) / 2.0

        if self.sparsity_threshold:
            # Scale the proximal step so the shrinkage does not erase the graph.
            self.coefficients = soft_threshold(
                self.coefficients,
                self.sparsity_threshold * (1.0 - self.forgetting_factor),
            )
        self.n_updates += 1
        return OnlineUpdate(
            prediction=prediction,
            residual=residual,
            score=float(np.mean(residual**2)),
            adjacency=self.adjacency,
        )

    def reset_covariance(self) -> None:
        """Increase adaptation speed after an externally detected regime shift."""
        self.covariance = np.eye(self.n_variables, dtype=float) / self.ridge
