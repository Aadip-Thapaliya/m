"""Single-pass Page-Hinkley change detection on pre-update residual scores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftSignal:
    detected: bool
    statistic: float
    running_mean: float
    n_seen: int


class PageHinkley:
    """Detect persistent upward shifts while using only observations seen so far."""

    def __init__(
        self, *, delta: float = 0.005, threshold: float = 3.0, min_instances: int = 30
    ) -> None:
        if delta < 0 or threshold <= 0 or min_instances < 1:
            raise ValueError("Invalid Page-Hinkley detector parameters")
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.reset()

    def reset(self) -> None:
        self.n_seen = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def update(self, score: float) -> DriftSignal:
        self.n_seen += 1
        self.mean += (score - self.mean) / self.n_seen
        self.cumulative += score - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        statistic = self.cumulative - self.minimum
        detected = self.n_seen >= self.min_instances and statistic > self.threshold
        signal = DriftSignal(detected, statistic, self.mean, self.n_seen)
        if detected:
            self.reset()
        return signal
