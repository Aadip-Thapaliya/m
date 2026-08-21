"""Prototype graph memory for recurring causal regimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegimeMatch:
    index: int
    distance: float
    adjacency: np.ndarray


class RegimeMemory:
    """Store graph snapshots and retrieve the closest prior regime."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("Regime memory capacity must be positive")
        self.capacity = capacity
        self._graphs: list[np.ndarray] = []

    def add(self, adjacency: np.ndarray) -> None:
        graph = np.asarray(adjacency, dtype=float).copy()
        self._graphs.append(graph)
        if len(self._graphs) > self.capacity:
            self._graphs.pop(0)

    def nearest(self, adjacency: np.ndarray) -> RegimeMatch | None:
        if not self._graphs:
            return None
        current = np.asarray(adjacency, dtype=float)
        distances = [float(np.linalg.norm(graph - current)) for graph in self._graphs]
        index = int(np.argmin(distances))
        return RegimeMatch(index, distances[index], self._graphs[index].copy())

    def __len__(self) -> int:
        return len(self._graphs)
