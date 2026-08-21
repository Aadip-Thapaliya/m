from __future__ import annotations

import unittest

import numpy as np

from ocg_ssm.data.synthetic import SCENARIOS, generate_dataset


class SyntheticDatasetTests(unittest.TestCase):
    def test_seed_reproduces_observations_and_graphs(self) -> None:
        first = generate_dataset(n_steps=120, n_variables=4, seed=17)
        second = generate_dataset(n_steps=120, n_variables=4, seed=17)
        np.testing.assert_array_equal(first.observations, second.observations)
        np.testing.assert_array_equal(first.adjacency, second.adjacency)

    def test_every_scenario_has_pointwise_ground_truth(self) -> None:
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                dataset = generate_dataset(scenario=scenario, n_steps=90, n_variables=5)
                self.assertEqual(dataset.observations.shape, (90, 5))
                self.assertEqual(dataset.adjacency.shape, (90, 5, 5))

    def test_abrupt_graph_changes_at_declared_point(self) -> None:
        dataset = generate_dataset(n_steps=120, n_variables=4, change_points=[45, 85])
        self.assertFalse(np.allclose(dataset.adjacency[44], dataset.adjacency[45]))

    def test_stationary_has_no_reported_change_points(self) -> None:
        dataset = generate_dataset(scenario="stationary", n_steps=90, n_variables=4)
        self.assertEqual(dataset.change_points, ())

    def test_abrupt_and_recurring_have_different_final_regimes(self) -> None:
        abrupt = generate_dataset(scenario="abrupt", n_steps=120, n_variables=4, seed=9)
        recurring = generate_dataset(scenario="recurring", n_steps=120, n_variables=4, seed=9)
        np.testing.assert_allclose(recurring.adjacency[0], recurring.adjacency[-1])
        self.assertFalse(np.allclose(abrupt.adjacency[0], abrupt.adjacency[-1]))

    def test_gradual_scenario_has_two_distinct_transitions(self) -> None:
        gradual = generate_dataset(
            scenario="gradual", n_steps=180, n_variables=4, change_points=[50, 110]
        )
        self.assertFalse(np.allclose(gradual.adjacency[49], gradual.adjacency[90]))
        self.assertFalse(np.allclose(gradual.adjacency[90], gradual.adjacency[-1]))
