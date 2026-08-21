"""Explicit graph conventions and structural diagnostics."""

from __future__ import annotations

import numpy as np


def threshold_graph(
    weights: np.ndarray, threshold: float, *, include_diagonal: bool = False
) -> np.ndarray:
    """Convert a weighted ``[target, parent]`` matrix into a binary graph."""
    if threshold < 0:
        raise ValueError("The graph threshold must be nonnegative")
    result = (np.abs(np.asarray(weights)) >= threshold).astype(int)
    if not include_diagonal:
        np.fill_diagonal(result, 0)
    return result


def graph_density(adjacency: np.ndarray, *, include_diagonal: bool = False) -> float:
    values = np.asarray(adjacency)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Adjacency must be a square matrix")
    n_variables = values.shape[0]
    mask = np.ones_like(values, dtype=bool)
    if not include_diagonal:
        np.fill_diagonal(mask, False)
    possible = int(mask.sum())
    return float(np.count_nonzero(values[mask]) / possible) if possible else 0.0


def soft_threshold(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Apply elementwise proximal shrinkage without imposing a lagged DAG."""
    values = np.asarray(weights, dtype=float)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def contemporaneous_order_projection(weights: np.ndarray, order: list[int]) -> np.ndarray:
    """Project onto a DAG under an *explicitly supplied and justified* ordering.

    This function is not applied to lagged causal graphs. A fixed order is a
    substantive identifiability assumption, not a generic DAG projection.
    """
    values = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Contemporaneous adjacency must be square")
    if sorted(order) != list(range(values.shape[0])):
        raise ValueError("Order must be a permutation of the variable indices")
    ordered = values[np.ix_(order, order)]
    projected = np.tril(ordered, k=-1)
    restored = np.zeros_like(projected)
    restored[np.ix_(order, order)] = projected
    return restored
