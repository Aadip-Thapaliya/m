#!/usr/bin/env python
"""Run the smallest complete graph-discovery-to-forecast experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocg_ssm.config import load_config  # noqa: E402
from ocg_ssm.experiment import run_experiment  # noqa: E402


def main() -> None:
    result = run_experiment(load_config(ROOT / "configs" / "smoke.yaml"))
    if len(result["models"]) != 4:
        raise SystemExit("Expected four benchmark models")
    if any(not 0 <= values["mse"] < 10 for values in result["models"].values()):
        raise SystemExit("Smoke experiment produced an invalid forecasting loss")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
