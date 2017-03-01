from unittest import TestCase

import numpy as np
from numpy.testing import *
import scipy.sparse as sp

from .graph import max_weight, normalize, gaussian


class GraphTest(TestCase):

    def test_max_weight(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        assert_equal(max_weight(A), 2)

    def test_normalize(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        expected = [[0,   0.5, 0],
                    [0.5,   0, 1],
                    [0,     1, 0]]

        assert_equal(normalize(A, max_value=2).toarray(), expected)

    def test_gaussian(self):
        A = [[0, 1, 0],
             [1, 0, 2],
             [0, 2, 0]]
        A = sp.coo_matrix(np.array(A))

        expected = [[0,            np.exp(-1/2),            0],
                    [np.exp(-1/2),            0, np.exp(-4/2)],
                    [0,            np.exp(-4/2),            0]]

        assert_equal(gaussian(A, sigma=1).toarray(), expected)

        expected = [[0,            np.exp(-1/8),            0],
                    [np.exp(-1/8),            0, np.exp(-4/8)],
                    [0,            np.exp(-4/8),            0]]

        assert_equal(gaussian(A, sigma=2).toarray(), expected)
