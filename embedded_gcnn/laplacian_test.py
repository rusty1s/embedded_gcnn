from __future__ import division

from unittest import TestCase
from math import sqrt

import numpy as np
from numpy.testing import *
import scipy.sparse as sp


from .laplacian import (laplacian, _lmax, _rescale, chebyshev)


class LaplacianTest(TestCase):

    def test_laplacian(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        combinatorial = [[1,  -1,  0],
                         [-1,  3, -2],
                         [0,  -2,  2]]

        L = laplacian(A, normalized=False)
        assert_equal(L.toarray(), combinatorial)

        normalized = [[1,            -1/sqrt(3),            0],
                      [-1/sqrt(3),            1, -2/sqrt(3*2)],
                      [0,          -2/sqrt(3*2),            1]]

        L = laplacian(A, normalized=True)
        assert_almost_equal(L.toarray(), normalized)

    def test_lmax(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A, dtype=np.float32))

        L = laplacian(A, normalized=False)
        assert_almost_equal(_lmax(L, normalized=False), 4.7320509)

        L = laplacian(A, normalized=True)
        assert_equal(_lmax(L, normalized=True), 2)

    def test_rescale(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        L = laplacian(A, normalized=True)

        expected = [[0,            -1/sqrt(3),            0],
                    [-1/sqrt(3),            0, -2/sqrt(3*2)],
                    [0,          -2/sqrt(3*2),            0]]

        assert_almost_equal(_rescale(L, 2).toarray(), expected)

    def test_chebyshev(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))
        L = laplacian(A, normalized=True)
        r_L = _rescale(L, 2)

        X = np.array([[1], [2], [3]])

        Xt = chebyshev(L, X, k=2, normalized=True)

        assert_equal(Xt[0], X)
        assert_almost_equal(Xt[1], r_L.dot(X))
        assert_almost_equal(Xt[2], 2 * r_L.dot(r_L).dot(X) - X)
