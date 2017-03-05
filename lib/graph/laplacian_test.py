from __future__ import division

from unittest import TestCase
from math import sqrt

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .laplacian import laplacian, lmax, rescale_lap


class LaplacianTest(TestCase):
    def test_laplacian(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(np.array(adj))

        combinatorial = [[1, -1, 0], [-1, 3, -2], [0, -2, 2]]

        lap = laplacian(adj, normalized=False)
        assert_equal(lap.toarray(), combinatorial)

        normalized = [[1, -1 / sqrt(3), 0],
                      [-1 / sqrt(3), 1, -2 / sqrt(3 * 2)],
                      [0, -2 / sqrt(3 * 2), 1]]

        lap = laplacian(adj, normalized=True)
        assert_almost_equal(lap.toarray(), normalized)

    def test_lmax(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(np.array(adj, dtype=np.float32))

        lap = laplacian(adj, normalized=False)
        assert_almost_equal(lmax(lap, normalized=False), 4.7320508)

        lap = laplacian(adj, normalized=True)
        assert_equal(lmax(lap, normalized=True), 2)

    def test_rescale_lap(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(np.array(adj))

        lap = laplacian(adj, normalized=True)

        expected = [[0, -1 / sqrt(3), 0], [-1 / sqrt(3), 0, -2 / sqrt(3 * 2)],
                    [0, -2 / sqrt(3 * 2), 0]]

        assert_almost_equal(rescale_lap(lap, lmax=2).toarray(), expected)
