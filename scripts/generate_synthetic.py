#!/usr/bin/env python
"""Export reproducible synthetic observations and exact causal ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocg_ssm.data.synthetic import SCENARIOS, generate_dataset  # noqa: E402


def save_scenario(scenario: str, args: argparse.Namespace) -> None:
    dataset = generate_dataset(
        scenario=scenario,
        n_steps=args.steps,
        n_variables=args.variables,
        noise_scale=args.noise,
        seed=args.seed,
    )
    destination = Path(args.output) / scenario
    destination.mkdir(parents=True, exist_ok=True)
    columns = [f"x{index}" for index in range(args.variables)]
    frame = pd.DataFrame(dataset.observations, columns=columns)
    frame.insert(0, "time", np.arange(args.steps))
    frame["regime_id"] = dataset.regime_ids
    frame.to_csv(destination / "observations.csv", index=False, float_format="%.8f")

    # A pointwise compressed array is convenient for full research experiments.
    np.savez_compressed(destination / "ground_truth_graphs.npz", adjacency=dataset.adjacency)
    metadata = {
        "scenario": scenario,
        "seed": args.seed,
        "n_steps": args.steps,
        "n_variables": args.variables,
        "change_points": list(dataset.change_points),
        "adjacency_convention": "adjacency[target, parent] at lag one",
        "latent_confounder": scenario == "confounded",
        "regime_graphs": {
            str(int(regime)): dataset.adjacency[np.flatnonzero(dataset.regime_ids == regime)[0]].round(8).tolist()
            for regime in np.unique(dataset.regime_ids)
        },
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {scenario:11s} {args.steps:5d} rows -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="abrupt")
    parser.add_argument("--all", action="store_true", help="Generate every documented stress scenario")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--noise", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "LEVEL_03_DATASETS" / "synthetic"))
    args = parser.parse_args()
    for scenario in SCENARIOS if args.all else (args.scenario,):
        save_scenario(scenario, args)


if __name__ == "__main__":
    main()
