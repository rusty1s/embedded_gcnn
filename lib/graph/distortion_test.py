from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .distortion import (pad_adj, pad_features, perm_adj, perm_features,
                         perm_batch_of_features)


class DistortionTest(TestCase):
    def test_pad_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]]
        assert_equal(pad_adj(adj, (4, 4)).toarray(), expected)

        expected = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        assert_equal(pad_adj(adj, (3, 3)).toarray(), expected)

    def test_pad_features(self):
        features = np.array([[0, 1], [2, 3], [4, 5]])

        expected = [[0, 1], [2, 3], [4, 5], [0, 0], [0, 0]]
        assert_equal(pad_features(features, 5), expected)

        expected = [[0, 1], [2, 3], [4, 5]]
        assert_equal(pad_features(features, 3), expected)

    def test_perm_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)
        perm = np.array([2, 1, 3, 0])

        expected = [[0, 0, 2, 1], [0, 0, 1, 2], [2, 1, 0, 0], [1, 2, 0, 0]]

        assert_equal(perm_adj(adj, perm).toarray(), expected)

        # Add fake nodes.
        perm = np.array([3, 2, 0, 4, 1])

        expected = [[0, 2, 0, 0, 1], [2, 0, 1, 0, 0], [0, 1, 0, 0, 2],
                    [0, 0, 0, 0, 0], [1, 0, 2, 0, 0]]

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

    def test_perm_batch_of_features(self):
        features_1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        features_2 = np.array([[11, 12], [13, 14], [15, 16], [17, 18]])
        features = np.array([features_1, features_2])
        perm = np.array([2, 1, 3, 0])

        expected = [[[5, 6], [3, 4], [7, 8], [1, 2]], [[15, 16], [13, 14],
                                                       [17, 18], [11, 12]]]
        assert_equal(perm_batch_of_features(features, perm), expected)

        # Test random permutation.
        features_new = perm_batch_of_features(features)
        assert_equal(features_new.shape, [2, 4, 2])
        assert_equal(np.array(features_new[0].sum()).flatten(), [36])
        assert_equal(np.array(features_new[1].sum()).flatten(), [116])
