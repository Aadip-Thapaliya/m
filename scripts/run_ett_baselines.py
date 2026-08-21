#!/usr/bin/env python
"""Run leakage-safe one-step streaming baselines on a bundled ETT dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocg_ssm.data.ett import TrainingPrefixScaler, load_ett  # noqa: E402
from ocg_ssm.discovery.online_var import OnlineVARDiscovery  # noqa: E402
from ocg_ssm.evaluation.prequential import evaluate_prequential  # noqa: E402
from ocg_ssm.models.baselines import OnlineVARForecaster, PersistenceForecaster, StaticVARForecaster  # noqa: E402
from ocg_ssm.models.portable import GraphConditionedSSM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2"), default="ETTh1")
    parser.add_argument("--max-rows", type=int, default=4000)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    args = parser.parse_args()
    source = ROOT / "LEVEL_03_DATASETS" / "real" / "ETT" / f"{args.dataset}.csv"
    dataset = load_ett(source, max_rows=args.max_rows)
    start = int(len(dataset.values) * args.train_fraction)
    scaler = TrainingPrefixScaler.fit(dataset.values[:start])
    standardized = scaler.transform(dataset.values)
    train = standardized[:start]
    n_variables = len(dataset.columns)

    online = OnlineVARForecaster(OnlineVARDiscovery(n_variables, forgetting_factor=0.99, ridge=5.0))
    graph = GraphConditionedSSM(n_variables, forgetting_factor=0.99, ridge=5.0)
    for index in range(1, len(train)):
        online.update(train[index - 1], train[index])
        graph.update(train[index - 1], train[index])

    models = {
        "persistence": PersistenceForecaster(),
        "static_var": StaticVARForecaster().fit(train),
        "online_var": online,
        "graph_conditioned_ssm": graph,
    }
    results = {}
    for name, model in models.items():
        result = evaluate_prequential(model, standardized, start=start)
        results[name] = {"standardized_mse": result.mean_squared_error, "standardized_mae": result.mean_absolute_error}
    print(json.dumps({"dataset": args.dataset, "rows": len(dataset.values), "evaluation_start": start, "models": results}, indent=2))


if __name__ == "__main__":
    main()
