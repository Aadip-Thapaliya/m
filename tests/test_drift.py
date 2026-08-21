from __future__ import annotations

import unittest

import numpy as np

from ocg_ssm.adaptation.drift import PageHinkley
from ocg_ssm.adaptation.memory import RegimeMemory


class AdaptationTests(unittest.TestCase):
    def test_page_hinkley_detects_a_persistent_mean_shift(self) -> None:
        detector = PageHinkley(delta=0.01, threshold=1.5, min_instances=15)
        detected = [detector.update(value).detected for value in [0.05] * 40 + [1.2] * 40]
        self.assertTrue(any(detected[40:]))

    def test_regime_memory_finds_nearest_graph(self) -> None:
        memory = RegimeMemory(capacity=2)
        memory.add(np.zeros((2, 2)))
        memory.add(np.ones((2, 2)))
        match = memory.nearest(np.full((2, 2), 0.9))
        self.assertIsNotNone(match)
        self.assertEqual(match.index, 1)

    def test_regime_memory_respects_capacity(self) -> None:
        memory = RegimeMemory(capacity=2)
        for value in range(4):
            memory.add(np.full((2, 2), value))
        self.assertEqual(len(memory), 2)
