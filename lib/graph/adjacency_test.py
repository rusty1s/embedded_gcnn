from __future__ import division

import tensorflow as tf
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .adjacency import (normalize_adj, invert_adj,
                        filter_highly_connected_nodes, grid_adj)


class GraphTest(tf.test.TestCase):
    def test_normalize_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, 0.5, 0], [0.5, 0, 1], [0, 1, 0]]

        assert_equal(normalize_adj(adj).toarray(), expected)

    def test_normalize_adj_scale_invariant(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, 0.75, 0], [0.75, 0, 1], [0, 1, 0]]

        assert_equal(
            normalize_adj(adj, scale_invariance=True).toarray(), expected)

    def test_invert_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, np.exp(-1 / 2), 0],
                    [np.exp(-1 / 2), 0, np.exp(-2 / 2)],
                    [0, np.exp(-2 / 2), 0]]

        assert_almost_equal(invert_adj(adj, stddev=1).toarray(), expected)

        expected = [[0, np.exp(-1 / 8), 0],
                    [np.exp(-1 / 8), 0, np.exp(-2 / 8)],
                    [0, np.exp(-2 / 8), 0]]

        assert_almost_equal(invert_adj(adj, stddev=2).toarray(), expected)

        assert_equal(invert_adj(2, stddev=1), np.exp(-1))

    def test_filter_highly_connected_nodes(self):
        adj = [[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0],
               [1, 0, 1, 0, 1], [1, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)

        perm = filter_highly_connected_nodes(adj, capacity=3)
        assert_equal(perm, [1, 2, 3, 4])

        perm = filter_highly_connected_nodes(adj, capacity=2)
        assert_equal(perm, [1, 4])

        # Test different weights.
        adj = [[0, 2, 3, 4, 5], [2, 0, 1, 0, 0], [3, 1, 0, 1, 0],
               [4, 0, 1, 0, 1], [5, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)

        perm = filter_highly_connected_nodes(adj, capacity=3)
        assert_equal(perm, [1, 2, 3, 4])

        perm = filter_highly_connected_nodes(adj, capacity=2)
        assert_equal(perm, [1, 4])

    def test_grid_adj_connectivity_4(self):
        adj = grid_adj((3, 2), connectivity=4)

        expected = [[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 0]]

        assert_equal(adj.toarray(), expected)

    def test_grid_adj_connectivity_8(self):
        adj = grid_adj((3, 2), connectivity=8)

        expected = [[0, 1, 1, 2, 0, 0], [1, 0, 2, 1, 0, 0], [1, 2, 0, 1, 1, 2],
                    [2, 1, 1, 0, 2, 1], [0, 0, 1, 2, 0, 1], [0, 0, 2, 1, 1, 0]]

        assert_equal(adj.toarray(), expected)
