from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocg_ssm.data.ett import ETT_FEATURES, TrainingPrefixScaler, load_ett


class ETTTests(unittest.TestCase):
    def test_ett_reader_preserves_order_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "example.csv"
            source.write_text(
                "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
                "2020-01-01 00:00:00,1,2,3,4,5,6,7\n"
                "2020-01-01 01:00:00,2,3,4,5,6,7,8\n",
                encoding="utf-8",
            )
            dataset = load_ett(source)
            self.assertEqual(dataset.columns, ETT_FEATURES)
            self.assertEqual(dataset.values.shape, (2, 7))

    def test_scaler_uses_training_prefix_only(self) -> None:
        training = np.array([[0.0, 2.0], [2.0, 4.0]])
        scaler = TrainingPrefixScaler.fit(training)
        np.testing.assert_allclose(scaler.mean, np.array([1.0, 3.0]))
        np.testing.assert_allclose(scaler.transform(np.array([[101.0, 103.0]])), np.array([[100.0, 100.0]]))
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(training)), training)
