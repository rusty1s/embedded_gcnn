from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .coarsening import _cluster_adj


class CoarseningTest(TestCase):
    def test_cluster_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)
        rid = np.array([0, 1, 2, 3])
        expected = [0, 0, 1, 1]
        assert_equal(_cluster_adj(adj, rid), expected)

        rid = np.array([3, 2, 1, 0])
        expected = [1, 1, 0, 0]
        assert_equal(_cluster_adj(adj, rid), expected)

        adj = [[0, 3, 2, 0, 0], [3, 0, 0, 2, 0], [2, 0, 0, 3, 0],
               [0, 2, 3, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        rid = np.array([0, 1, 2, 3, 4])
        expected = [0, 0, 1, 1, 2]
        assert_equal(_cluster_adj(adj, rid), expected)

        rid = np.array([4, 3, 2, 1, 0])
        expected = [1, 2, 1, 0, 0]
        assert_equal(_cluster_adj(adj, rid), expected)
