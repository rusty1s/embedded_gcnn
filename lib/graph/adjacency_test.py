from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .adjacency import normalize_adj, invert_adj, grid_adj, embedded_adj


class GraphTest(TestCase):

    def test_normalize_adj(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        expected = [[0,   0.5, 0],
                    [0.5,   0, 1],
                    [0,     1, 0]]

        assert_equal(normalize_adj(A).toarray(), expected)

    def test_invert_adj(self):
        # Test sparse representation.
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        expected = [[0,            np.exp(-1/2),            0],
                    [np.exp(-1/2),            0, np.exp(-2/2)],
                    [0,            np.exp(-2/2),            0]]

        assert_almost_equal(invert_adj(A, sigma=1).toarray(), expected)

        expected = [[0,            np.exp(-1/8),            0],
                    [np.exp(-1/8),            0, np.exp(-2/8)],
                    [0,            np.exp(-2/8),            0]]

        assert_almost_equal(invert_adj(A, sigma=2).toarray(), expected)

        # Test dense representation.
        assert_equal(invert_adj(2, sigma=1), np.exp(-1))

    def test_grid_adj(self):
        A = grid_adj((3, 2), connectivity=4)

        expected = [[0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 1, 1, 0]]

        assert_equal(A.toarray(), expected)

        A = grid_adj((3, 2), connectivity=8)

        expected = [[0, 1, 1, 2, 0, 0],
                    [1, 0, 2, 1, 0, 0],
                    [1, 2, 0, 1, 1, 2],
                    [2, 1, 1, 0, 2, 1],
                    [0, 0, 1, 2, 0, 1],
                    [0, 0, 2, 1, 1, 0]]

        assert_equal(A.toarray(), expected)

    def test_embedded_adj(self):
        points = np.array([[1, 1], [3, 2], [4, -1]])
        neighbors = np.array([[0, 1], [0, 2], [1, 2]])

        expected = [[0,   5, 13],
                    [5,   0, 10],
                    [13, 10,  0]]

        assert_equal(embedded_adj(points, neighbors).toarray(), expected)
