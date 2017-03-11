from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .distortion import perm_adj, perm_features


class DistortionTest(TestCase):
    def test_perm_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)
        perm = np.array([2, 1, 3, 0])

        expected = [[0, 1, 0, 2], [1, 0, 2, 0], [0, 2, 0, 1], [2, 0, 1, 0]]

        assert_equal(perm_adj(adj, perm).toarray(), expected)

        # Add fake nodes.
        perm = np.array([3, 2, 0, 4, 1])

        expected = [[0, 0, 0, 1, 2], [0, 0, 0, 0, 0], [0, 0, 0, 2, 1],
                    [1, 0, 2, 0, 0], [2, 0, 1, 0, 0]]

        assert_equal(perm_adj(adj, perm).toarray(), expected)

        # Test random permutation.
        adj_new = perm_adj(adj)
        assert_equal(adj_new.shape, [4, 4])
        assert_equal(np.array(adj_new.sum(1)).flatten(), [3, 3, 3, 3])
        assert_equal(np.array(adj_new.sum(0)).flatten(), [3, 3, 3, 3])

    def test_perm_features(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        perm = np.array([2, 1, 3, 0])

        expected = [[5, 6], [3, 4], [7, 8], [1, 2]]

        assert_equal(perm_features(features, perm), expected)

        # Add fake nodes.
        perm = np.array([3, 2, 0, 4, 1])

        expected = [[7, 8], [5, 6], [1, 2], [0, 0], [3, 4]]

        assert_equal(perm_features(features, perm), expected)

        # Test random permutation.
        features_new = perm_features(features)
        assert_equal(features_new.shape, [4, 2])
        assert_equal(features_new.sum(), 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8)
