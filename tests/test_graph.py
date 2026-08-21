from __future__ import annotations

import unittest

import numpy as np

from ocg_ssm.discovery.graph import contemporaneous_order_projection, graph_density, threshold_graph
from ocg_ssm.evaluation.metrics import graph_metrics


class GraphTests(unittest.TestCase):
    def test_threshold_removes_self_edges_by_default(self) -> None:
        graph = threshold_graph(np.array([[1.0, 0.3], [0.05, 1.0]]), 0.10)
        np.testing.assert_array_equal(graph, np.array([[0, 1], [0, 0]]))

    def test_reversed_directed_edge_costs_two_shd_edits(self) -> None:
        truth = np.array([[0.0, 0.0], [0.8, 0.0]])
        estimate = np.array([[0.0, 0.8], [0.0, 0.0]])
        scores = graph_metrics(truth, estimate, threshold=0.1)
        self.assertEqual(scores["shd"], 2)
        self.assertEqual(scores["f1"], 0.0)

    def test_exact_graph_has_perfect_f1(self) -> None:
        graph = np.array([[0.0, 0.0, 0.3], [0.4, 0.0, 0.0], [0.0, 0.2, 0.0]])
        self.assertEqual(graph_metrics(graph, graph)["f1"], 1.0)

    def test_graph_density_counts_only_off_diagonal(self) -> None:
        self.assertAlmostEqual(graph_density(np.array([[1, 1], [0, 1]])), 0.5)

    def test_contemporaneous_order_projection(self) -> None:
        graph = contemporaneous_order_projection(np.ones((3, 3)), [0, 1, 2])
        np.testing.assert_array_equal(graph, np.tril(np.ones((3, 3)), k=-1))
