"""Ground-truth regime-switching structural vector autoregressive generators.

Adjacency convention: ``adjacency[target, parent]`` is the coefficient from the
parent at ``t - 1`` to the target at ``t``. Lagged feedback cycles are valid;
only contemporaneous causal graphs require a DAG assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCENARIOS = ("stationary", "abrupt", "gradual", "recurring", "confounded")


@dataclass(frozen=True)
class SyntheticDataset:
    """Observations and the exact graph active at every observation."""

    observations: np.ndarray
    adjacency: np.ndarray
    regime_ids: np.ndarray
    change_points: tuple[int, ...]
    scenario: str
    seed: int
    observed_confounder: bool = False


def _stable_graph(rng: np.random.Generator, n_variables: int) -> np.ndarray:
    matrix = np.zeros((n_variables, n_variables), dtype=float)
    np.fill_diagonal(matrix, rng.uniform(0.20, 0.48, size=n_variables))
    for target in range(n_variables):
        for parent in range(n_variables):
            if target != parent and rng.random() < 0.23:
                matrix[target, parent] = rng.choice((-1.0, 1.0)) * rng.uniform(0.12, 0.36)

    # Scale the spectral radius to leave room for stable autoregressive dynamics.
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral_radius > 0.88:
        matrix *= 0.88 / spectral_radius
    return matrix


def _regime_graphs(rng: np.random.Generator, n_variables: int) -> tuple[np.ndarray, np.ndarray]:
    first = _stable_graph(rng, n_variables)
    second = _stable_graph(rng, n_variables)
    if n_variables >= 3:
        first[1, 0] = 0.34
        second[1, 0] = 0.0
        second[2, 0] = -0.32
    for graph in (first, second):
        radius = float(np.max(np.abs(np.linalg.eigvals(graph))))
        if radius > 0.90:
            graph *= 0.90 / radius
    return first, second


def generate_dataset(
    *,
    scenario: str = "abrupt",
    n_steps: int = 1600,
    n_variables: int = 6,
    change_points: tuple[int, ...] | list[int] | None = None,
    noise_scale: float = 0.18,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate one reproducible dataset with pointwise causal ground truth.

    The confounded scenario injects an *unobserved* common driver into variables
    zero and one. Its reported adjacency intentionally excludes that latent
    variable, so graph recovery must be interpreted as a robustness stress test.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if n_variables < 2 or n_steps < 20:
        raise ValueError("At least two variables and twenty observations are required")

    rng = np.random.default_rng(seed)
    points = tuple(sorted(change_points or (n_steps // 3, 2 * n_steps // 3)))
    if any(point <= 1 or point >= n_steps for point in points):
        raise ValueError("Change points must lie strictly inside the observation range")

    graph_a, graph_b = _regime_graphs(rng, n_variables)
    graph_c = _stable_graph(rng, n_variables)
    observations = np.zeros((n_steps, n_variables), dtype=float)
    graphs = np.zeros((n_steps, n_variables, n_variables), dtype=float)
    regime_ids = np.zeros(n_steps, dtype=int)
    transition_width = max(12, n_steps // 12)
    latent_state = 0.0

    for step in range(n_steps):
        regime = sum(step >= point for point in points)
        if scenario == "stationary":
            graph, regime = graph_a, 0
        elif scenario == "gradual":
            if step < points[0]:
                graph, regime = graph_a, 0
            elif len(points) > 1 and step >= points[1]:
                alpha = np.clip((step - points[1]) / transition_width, 0.0, 1.0)
                graph = (1.0 - alpha) * graph_b + alpha * graph_c
                regime = 1 + int(alpha >= 0.5)
            else:
                alpha = np.clip((step - points[0]) / transition_width, 0.0, 1.0)
                graph = (1.0 - alpha) * graph_a + alpha * graph_b
                regime = int(alpha >= 0.5)
        elif scenario == "recurring":
            graph = graph_a if regime % 2 == 0 else graph_b
        else:
            graph = graph_a if regime == 0 else graph_b if regime == 1 else graph_c

        graphs[step] = graph
        regime_ids[step] = regime
        if step == 0:
            observations[step] = rng.normal(scale=noise_scale, size=n_variables)
            continue

        innovations = rng.normal(scale=noise_scale, size=n_variables)
        if scenario == "confounded":
            latent_state = 0.78 * latent_state + rng.normal(scale=noise_scale)
            innovations[0] += 0.60 * latent_state
            innovations[1] += 0.55 * latent_state
        observations[step] = graph @ observations[step - 1] + innovations

    effective_points = () if scenario == "stationary" else points
    return SyntheticDataset(
        observations=observations,
        adjacency=graphs,
        regime_ids=regime_ids,
        change_points=effective_points,
        scenario=scenario,
        seed=seed,
    )
