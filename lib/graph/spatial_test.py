from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .spatial import (node_selection, neighborhood_selection, receptive_fields,
                      fill_features)


class SpatialTest(TestCase):
    def test_node_selection(self):
        points = np.array([[2, 4], [0, 1], [2.1, 2.5], [0.2, 3], [0.1, 4]])

        assert_equal(node_selection(points, size=6), [1, 3, 4, 2, 0, -1])
        assert_equal(node_selection(points, size=2), [1, 3])
        assert_equal(node_selection(points, size=3, stride=3), [1, 2, -1])
        assert_equal(node_selection(points, size=4, stride=2), [1, 4, 0, -1])
        assert_equal(
            node_selection(points, size=6, delta=3), [1, 2, 3, 0, 4, -1])

    def test_neighborhood_selection(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])

        adj = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
               [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
        adj = sp.coo_matrix(adj)

        assert_equal(
            neighborhood_selection(0, points, adj, size=5), [0, 2, 1, 4, 3])
        assert_equal(neighborhood_selection(0, points, adj, size=3), [0, 2, 1])
        assert_equal(
            neighborhood_selection(0, points, adj, size=7),
            [0, 2, 1, 4, 3, 5, 5])

        assert_equal(
            neighborhood_selection(1, points, adj, size=5), [1, 0, 2, 4, 3])

        assert_equal(neighborhood_selection(-1, points, adj, size=2), [5, 5])

    def test_receptive_fields(self):
        points = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ])

        adj = sp.coo_matrix([
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ])

        fields = receptive_fields(
            points, adj, node_size=3, neighborhood_size=4, node_stride=2)

        excepted = [[3, 0, 2, 1], [0, 2, 1, 4], [1, 0, 2, 4]]

        assert_equal(fields, excepted)

    def test_fill_features(self):
        receptive_fields = np.array([
            [3, 0, 2, 1],
            [0, 2, 1, 4],
            [1, 0, 2, 5],
        ])

        features = np.array([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
        ])

        block = fill_features(receptive_fields, features)

        expected = [
            [[3, 4, 5], [0, 1, 2], [2, 3, 4], [1, 2, 3]],
            [[0, 1, 2], [2, 3, 4], [1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [0, 1, 2], [2, 3, 4], [0, 0, 0]],
        ]

        assert_equal(block, expected)
