from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .preprocess import preprocess_adj


class PreprocessTest(TestCase):
    def test_preprocess_adj(self):
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        adj = sp.coo_matrix(adj)

        adj_new = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
        adj_new = sp.coo_matrix(adj_new)
        degree = [[2, 0, 0], [0, 3, 0], [0, 0, 2]]
        degree = [[np.power(2, -0.5), 0, 0], [0, np.power(3, -0.5), 0],
                  [0, 0, np.power(2, -0.5)]]
        degree = sp.coo_matrix(degree)

        assert_equal(
            preprocess_adj(adj).toarray(),
            degree.dot(adj_new).dot(degree).toarray())

    def test_preprocess_adj_with_isolated_node(self):
        adj = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        adj = sp.coo_matrix(adj)

        adj_new = [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
        adj_new = sp.coo_matrix(adj_new)
        degree = [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
        degree = [[np.power(2, -0.5), 0, 0, 0], [0, np.power(3, -0.5), 0, 0],
                  [0, 0, np.power(2, -0.5), 0], [0, 0, 0, 0]]
        degree = sp.coo_matrix(degree)

        assert_equal(
            preprocess_adj(adj).toarray(),
            degree.dot(adj_new).dot(degree).toarray())
