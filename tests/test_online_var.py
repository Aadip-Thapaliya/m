from __future__ import annotations

import unittest

import numpy as np

from ocg_ssm.discovery.online_var import OnlineVARDiscovery


class OnlineVARTests(unittest.TestCase):
    def test_rls_recovers_linear_coefficients(self) -> None:
        rng = np.random.default_rng(14)
        true_adjacency = np.array([[0.6, 0.0], [0.35, 0.4]])
        model = OnlineVARDiscovery(2, forgetting_factor=1.0, ridge=0.01)
        for _ in range(1200):
            features = rng.normal(size=2)
            target = true_adjacency @ features + rng.normal(scale=0.015, size=2)
            model.update(features, target)
        np.testing.assert_allclose(model.adjacency, true_adjacency, atol=0.03)

    def test_score_is_based_on_pre_update_prediction(self) -> None:
        model = OnlineVARDiscovery(2)
        result = model.update(np.array([1.0, 0.0]), np.array([2.0, 0.0]))
        np.testing.assert_array_equal(result.prediction, np.array([0.0, 0.0]))
        self.assertEqual(result.score, 2.0)

    def test_invalid_forgetting_factor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OnlineVARDiscovery(2, forgetting_factor=1.2)
