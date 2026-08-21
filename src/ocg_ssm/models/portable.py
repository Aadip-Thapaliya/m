"""A transparent NumPy graph-conditioned recurrent state-space prototype.

This is a runnable research scaffold, not the optimized CUDA Mamba selective
scan. Its role is to validate online graph conditioning, prequential evaluation,
and regime-change instrumentation before implementing the full neural model.
"""

from __future__ import annotations

import numpy as np

from ocg_ssm.adaptation.drift import PageHinkley
from ocg_ssm.adaptation.memory import RegimeMemory
from ocg_ssm.discovery.online_var import OnlineVARDiscovery


class GraphConditionedSSM:
    """Blend graph-mediated transitions with a stable diagonal latent state."""

    def __init__(
        self,
        n_variables: int,
        *,
        forgetting_factor: float = 0.985,
        ridge: float = 8.0,
        sparsity_threshold: float = 0.05,
        hidden_decay: float = 0.80,
        graph_weight: float = 0.85,
        drift_threshold: float = 3.0,
        drift_delta: float = 0.005,
    ) -> None:
        if not 0 <= hidden_decay < 1 or not 0 <= graph_weight <= 1:
            raise ValueError("Decay and graph mixing weights must be valid probabilities")
        self.discovery = OnlineVARDiscovery(
            n_variables,
            forgetting_factor=forgetting_factor,
            ridge=ridge,
            sparsity_threshold=sparsity_threshold,
        )
        self.detector = PageHinkley(delta=drift_delta, threshold=drift_threshold)
        self.memory = RegimeMemory()
        self.hidden = np.zeros(n_variables, dtype=float)
        self.hidden_decay = hidden_decay
        self.graph_weight = graph_weight
        self.drift_events: list[int] = []
        self.steps = 0

    @property
    def adjacency(self) -> np.ndarray:
        return self.discovery.adjacency

    def predict(self, previous: np.ndarray) -> np.ndarray:
        graph_forecast = self.discovery.predict(previous)
        return self.graph_weight * graph_forecast + (1.0 - self.graph_weight) * self.hidden

    def update(self, previous: np.ndarray, observed: np.ndarray) -> None:
        # The score comes from the graph available *before* the target was seen.
        result = self.discovery.update(previous, observed)
        signal = self.detector.update(result.score)
        if signal.detected:
            self.drift_events.append(self.steps)
            self.memory.add(result.adjacency)
            self.discovery.reset_covariance()
        observed_values = np.asarray(observed, dtype=float)
        self.hidden = self.hidden_decay * self.hidden + (1.0 - self.hidden_decay) * observed_values
        self.steps += 1
