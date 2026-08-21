from __future__ import annotations

import unittest

import numpy as np

from ocg_ssm.data.windows import chronological_split, sliding_windows
from ocg_ssm.evaluation.prequential import evaluate_prequential


class RecordingForecaster:
    def __init__(self) -> None:
        self.events: list[str] = []

    def predict(self, previous: np.ndarray) -> np.ndarray:
        self.events.append("predict")
        return previous.copy()

    def update(self, previous: np.ndarray, observed: np.ndarray) -> None:
        self.events.append("update")


class PrequentialTests(unittest.TestCase):
    def test_prediction_always_precedes_target_update(self) -> None:
        model = RecordingForecaster()
        values = np.arange(12, dtype=float).reshape(6, 2)
        result = evaluate_prequential(model, values, start=2)
        self.assertEqual(model.events, ["predict", "update"] * 4)
        self.assertEqual(result.predictions.shape, (4, 2))

    def test_chronological_splits_preserve_time_order(self) -> None:
        split = chronological_split(np.arange(20), train_fraction=0.5, validation_fraction=0.25)
        self.assertEqual(split.train[-1], 9)
        self.assertEqual(split.validation[0], 10)
        self.assertEqual(split.test[0], 15)

    def test_sliding_windows_do_not_contain_future_targets(self) -> None:
        contexts, targets = sliding_windows(np.arange(8), context=3, horizon=2)
        np.testing.assert_array_equal(contexts[0], np.array([0, 1, 2]))
        np.testing.assert_array_equal(targets[0], np.array([3, 4]))
