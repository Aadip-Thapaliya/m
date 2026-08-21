"""Executable synthetic benchmark linking discovery, adaptation, and forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np

from ocg_ssm.data.synthetic import generate_dataset
from ocg_ssm.discovery.online_var import OnlineVARDiscovery
from ocg_ssm.evaluation.metrics import graph_metrics
from ocg_ssm.evaluation.prequential import evaluate_prequential
from ocg_ssm.models.baselines import OnlineVARForecaster, PersistenceForecaster, StaticVARForecaster
from ocg_ssm.models.portable import GraphConditionedSSM


def _warm_up(model: Any, training: np.ndarray) -> None:
    for index in range(1, len(training)):
        model.update(training[index - 1], training[index])


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    dataset_config = config["dataset"]
    discovery_config = config["discovery"]
    adaptation_config = config["adaptation"]
    model_config = config["model"]
    n_variables = int(dataset_config["n_variables"])
    dataset = generate_dataset(
        scenario=dataset_config.get("scenario", "abrupt"),
        n_steps=int(dataset_config["n_steps"]),
        n_variables=n_variables,
        change_points=dataset_config.get("change_points"),
        noise_scale=float(dataset_config.get("noise_scale", 0.18)),
        seed=int(config.get("seed", 42)),
    )
    start = int(len(dataset.observations) * float(dataset_config.get("train_fraction", 0.35)))
    training = dataset.observations[:start]

    persistence = PersistenceForecaster()
    static_var = StaticVARForecaster().fit(training)
    online_var = OnlineVARForecaster(
        OnlineVARDiscovery(
            n_variables,
            forgetting_factor=float(discovery_config["forgetting_factor"]),
            ridge=float(discovery_config["ridge"]),
            sparsity_threshold=float(discovery_config["sparsity_threshold"]),
        )
    )
    graph_ssm = GraphConditionedSSM(
        n_variables,
        forgetting_factor=float(discovery_config["forgetting_factor"]),
        ridge=float(discovery_config["ridge"]),
        sparsity_threshold=float(discovery_config["sparsity_threshold"]),
        hidden_decay=float(model_config["hidden_decay"]),
        graph_weight=float(model_config["graph_weight"]),
        drift_threshold=float(adaptation_config["page_hinkley_threshold"]),
        drift_delta=float(adaptation_config["page_hinkley_delta"]),
    )
    _warm_up(online_var, training)
    _warm_up(graph_ssm, training)

    models = {
        "persistence": persistence,
        "static_var": static_var,
        "online_var": online_var,
        "graph_conditioned_ssm": graph_ssm,
    }
    scores = {}
    for name, model in models.items():
        result = evaluate_prequential(model, dataset.observations, start=start)
        scores[name] = {
            "mse": result.mean_squared_error,
            "mae": result.mean_absolute_error,
            "n_predictions": len(result.predictions),
        }

    graph_score = graph_metrics(
        dataset.adjacency[-1],
        graph_ssm.adjacency,
        threshold=float(config["evaluation"].get("graph_threshold", 0.10)),
    )
    return {
        "scenario": dataset.scenario,
        "seed": dataset.seed,
        "n_steps": len(dataset.observations),
        "n_variables": n_variables,
        "evaluation_start": start,
        "true_change_points": list(dataset.change_points),
        "detected_drifts": graph_ssm.drift_events,
        "models": scores,
        "final_graph_metrics": graph_score,
    }
