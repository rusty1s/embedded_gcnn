from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .distortion import perm_adj


class DistortionTest(TestCase):
    def test_perm_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)
        perm = np.array([2, 1, 3, 0])

        expected = [[0, 1, 0, 2], [1, 0, 2, 0], [0, 2, 0, 1], [2, 0, 1, 0]]

        assert_equal(perm_adj(adj, perm).toarray(), expected)

        # Add fake nodes
        perm = np.array([3, 2, 0, 4, 1])

        expected = [[0, 0, 0, 1, 2], [0, 0, 0, 0, 0], [0, 0, 0, 2, 1],
                    [1, 0, 2, 0, 0], [2, 0, 1, 0, 0]]

        assert_equal(perm_adj(adj, perm).toarray(), expected)

        # Test random perm
        adj_new = perm_adj(adj)
        assert_equal(adj_new.shape, [4, 4])
        assert_equal(np.array(adj_new.sum(1)).flatten(), [3, 3, 3, 3])
        assert_equal(np.array(adj_new.sum(0)).flatten(), [3, 3, 3, 3])

