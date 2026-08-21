from __future__ import annotations

import unittest

import numpy as np

from ocg_ssm.evaluation.metrics import cumulative_regret, gaussian_crps, mae, mse


class MetricTests(unittest.TestCase):
    def test_mse_and_mae(self) -> None:
        actual = np.array([0.0, 2.0])
        predicted = np.array([1.0, 0.0])
        self.assertEqual(mse(actual, predicted), 2.5)
        self.assertEqual(mae(actual, predicted), 1.5)

    def test_gaussian_crps_is_positive_and_rewards_accuracy(self) -> None:
        accurate = gaussian_crps(np.array([0.0]), np.array([0.0]), 1.0)
        inaccurate = gaussian_crps(np.array([0.0]), np.array([3.0]), 1.0)
        self.assertGreaterEqual(accurate, 0.0)
        self.assertLess(accurate, inaccurate)

    def test_crps_rejects_zero_variance(self) -> None:
        with self.assertRaises(ValueError):
            gaussian_crps(np.array([0.0]), np.array([0.0]), 0.0)

    def test_cumulative_regret(self) -> None:
        result = cumulative_regret(np.array([2.0, 1.0]), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.array([1.0, 0.0]))
