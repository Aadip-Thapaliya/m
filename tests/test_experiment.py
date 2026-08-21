from __future__ import annotations

import unittest
from pathlib import Path

from ocg_ssm.config import load_config
from ocg_ssm.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def test_smoke_configuration_runs_all_four_models(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = run_experiment(load_config(root / "configs" / "smoke.yaml"))
        self.assertEqual(set(result["models"]), {"persistence", "static_var", "online_var", "graph_conditioned_ssm"})
        self.assertGreater(result["models"]["online_var"]["n_predictions"], 0)
        self.assertIn("f1", result["final_graph_metrics"])
