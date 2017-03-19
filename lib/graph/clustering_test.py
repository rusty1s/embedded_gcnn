from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .clustering import normalized_cut


class ClusteringTest(TestCase):
    def test_normalized_cut_without_singletons(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)
        rid = np.array([0, 1, 2, 3])
        assert_equal(normalized_cut(adj, rid), [0, 0, 1, 1])

        rid = np.array([3, 2, 1, 0])
        assert_equal(normalized_cut(adj, rid), [1, 1, 0, 0])

    def test_normalized_cut_with_singletons(self):
        adj = [[0, 3, 0, 2, 0], [3, 0, 2, 0, 0], [0, 2, 0, 3, 0],
               [2, 0, 3, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        rid = np.array([0, 1, 2, 3, 4])
        assert_equal(normalized_cut(adj, rid), [0, 0, 1, 1, 2])

        rid = np.array([4, 3, 2, 1, 0])
        assert_equal(normalized_cut(adj, rid), [2, 1, 1, 0, 0])

        rid = np.array([1, 0, 4, 2, 3])
        assert_equal(normalized_cut(adj, rid), [0, 0, 2, 1, 1])
