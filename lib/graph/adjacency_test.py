from __future__ import division

import tensorflow as tf
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .adjacency import normalize_adj, invert_adj, grid_adj


class GraphTest(tf.test.TestCase):
    def test_normalize_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, 0.5, 0], [0.5, 0, 1], [0, 1, 0]]

        assert_equal(normalize_adj(adj).toarray(), expected)

    def test_invert_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, np.exp(-1 / 2), 0],
                    [np.exp(-1 / 2), 0, np.exp(-2 / 2)],
                    [0, np.exp(-2 / 2), 0]]

        assert_almost_equal(invert_adj(adj, sigma=1).toarray(), expected)

        expected = [[0, np.exp(-1 / 8), 0],
                    [np.exp(-1 / 8), 0, np.exp(-2 / 8)],
                    [0, np.exp(-2 / 8), 0]]

        assert_almost_equal(invert_adj(adj, sigma=2).toarray(), expected)

        assert_equal(invert_adj(2, sigma=1), np.exp(-1))

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
