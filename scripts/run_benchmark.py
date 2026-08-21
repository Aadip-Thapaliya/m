#!/usr/bin/env python
"""Execute the reproducible scenario-by-seed NumPy benchmark matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocg_ssm.config import load_config  # noqa: E402
from ocg_ssm.experiment import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--quick", action="store_true", help="Use two seeds and 360 observations")
    args = parser.parse_args()
    matrix = load_config(args.config)
    base = load_config(ROOT / matrix["base_config"])
    output = ROOT / matrix["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    seeds = matrix["seeds"][:2] if args.quick else matrix["seeds"]
    steps = 360 if args.quick else int(matrix["n_steps"])
    summaries, rows = [], []

    for scenario in matrix["scenarios"]:
        for seed in seeds:
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["dataset"].update(
                scenario=scenario,
                n_steps=steps,
                n_variables=int(matrix["n_variables"]),
                change_points=[steps // 3, 2 * steps // 3],
            )
            result = run_experiment(config)
            summaries.append(result)
            for model, metrics in result["models"].items():
                rows.append({"scenario": scenario, "seed": seed, "model": model, **metrics})
            print(f"Completed scenario={scenario:11s} seed={seed:3d}")

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "benchmark_results.csv", index=False)
    (output / "benchmark_details.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    grouped = frame.groupby(["scenario", "model"])[["mse", "mae"]].agg(["mean", "std"])
    grouped.to_csv(output / "benchmark_summary.csv")
    print("\n" + grouped.round(5).to_string())
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()
