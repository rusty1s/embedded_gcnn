from unittest import TestCase

import numpy as np
from numpy.testing import *
import scipy.sparse as sp

from .graph import normalize, gaussian


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
