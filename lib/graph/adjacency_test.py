from __future__ import division

from unittest import TestCase

import numpy as np
from numpy import pi as PI
from numpy.testing import assert_equal
import scipy.sparse as sp

from .adjacency import (zero_one_scale_adj, invert_adj, points_to_l2_adj,
                        points_to_adj)


class GraphTest(TestCase):
    def test_zero_one_scale_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, 0.5, 0], [0.5, 0, 1], [0, 1, 0]]

        assert_equal(zero_one_scale_adj(adj).toarray(), expected)

        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, 0.75, 0], [0.75, 0, 1], [0, 1, 0]]

        assert_equal(
            zero_one_scale_adj(adj, scale_invariance=True).toarray(), expected)

    def test_invert_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, np.exp(-1 / 2), 0],
                    [np.exp(-1 / 2), 0, np.exp(-2 / 2)],
                    [0, np.exp(-2 / 2), 0]]

        assert_equal(invert_adj(adj, stddev=1).toarray(), expected)

        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, np.exp(-1 / 8), 0],
                    [np.exp(-1 / 8), 0, np.exp(-2 / 8)],
                    [0, np.exp(-2 / 8), 0]]

        assert_equal(invert_adj(adj, stddev=2).toarray(), expected)

    def test_points_to_l2_adj(self):
        points = np.array([[2, 2], [2, 4], [3, 2], [2, 1], [1, 2]])
        adj = sp.coo_matrix([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        adj_dist, adj_rad = points_to_l2_adj(adj, points)

        expected_dist = [[0, 4, 1, 1, 1], [4, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
        expected_rad = [[0, 0.5 * PI, PI, 1.5 * PI, 2 * PI],
                        [1.5 * PI, 0, 0, 0, 0], [2 * PI, 0, 0, 0, 0],
                        [0.5 * PI, 0, 0, 0, 0], [PI, 0, 0, 0, 0]]

        assert_equal(adj_dist.toarray(), expected_dist)
        assert_equal(adj_rad.toarray(), expected_rad)

    def test_points_to_adj(self):
        points = np.array([[2, 2], [2, 4], [3, 2], [2, 1], [1, 2]])
        adj = sp.coo_matrix([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        adj_dist, adj_rad = points_to_adj(adj, points, stddev=2)

        expected_dist = [[
            0, np.exp(-1 / 8), np.exp(-0.25 / 8), np.exp(-0.25 / 8),
            np.exp(-0.25 / 8)
        ], [np.exp(-1 / 8), 0, 0, 0, 0], [np.exp(-0.25 / 8), 0, 0, 0, 0],
                         [np.exp(-0.25 / 8), 0, 0, 0, 0],
                         [np.exp(-0.25 / 8), 0, 0, 0, 0]]
        expected_rad = [[0, 0.5 * PI, PI, 1.5 * PI, 2 * PI],
                        [1.5 * PI, 0, 0, 0, 0], [2 * PI, 0, 0, 0, 0],
                        [0.5 * PI, 0, 0, 0, 0], [PI, 0, 0, 0, 0]]

        assert_equal(adj_dist.toarray(), expected_dist)
        assert_equal(adj_rad.toarray(), expected_rad)
