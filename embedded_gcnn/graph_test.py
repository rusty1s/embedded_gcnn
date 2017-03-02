from unittest import TestCase

import numpy as np
from numpy.testing import *
import scipy.sparse as sp

from .graph import normalize, gaussian, grid, embedded


class GraphTest(TestCase):

    def test_normalize(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        expected = [[0,   0.5, 0],
                    [0.5,   0, 1],
                    [0,     1, 0]]

        assert_equal(normalize(A).toarray(), expected)

    def test_gaussian(self):
        # Test sparse representation.
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        expected = [[0,            np.exp(-1/2),            0],
                    [np.exp(-1/2),            0, np.exp(-2/2)],
                    [0,            np.exp(-2/2),            0]]

        assert_equal(gaussian(A, sigma=1).toarray(), expected)

        expected = [[0,            np.exp(-1/8),            0],
                    [np.exp(-1/8),            0, np.exp(-2/8)],
                    [0,            np.exp(-2/8),            0]]

        assert_equal(gaussian(A, sigma=2).toarray(), expected)

        # Test dense representation.
        assert_equal(gaussian(2, sigma=1), np.exp(-1))

    def test_grid(self):
        A = grid((3, 2), connectivity=4)

        expected = [[0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 1, 1, 0]]

        assert_equal(A.toarray(), expected)

        A = grid((3, 2), connectivity=8)

        expected = [[0, 1, 1, 2, 0, 0],
                    [1, 0, 2, 1, 0, 0],
                    [1, 2, 0, 1, 1, 2],
                    [2, 1, 1, 0, 2, 1],
                    [0, 0, 1, 2, 0, 1],
                    [0, 0, 2, 1, 1, 0]]

        assert_equal(A.toarray(), expected)

    def test_embedded(self):
        points = np.array([[1, 1], [3, 2], [4, -1]])
        neighbors = np.array([[0, 1], [0, 2], [1, 2]])

        expected = [[0,   5, 13],
                    [5,   0, 10],
                    [13, 10,  0]]

        assert_equal(embedded(points, neighbors).toarray(), expected)
