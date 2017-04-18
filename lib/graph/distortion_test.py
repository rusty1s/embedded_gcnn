from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .distortion import perm_adj, perm_features, pad_adj, pad_features


class DistortionTest(TestCase):
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

        # Slice nodes.
        perm = np.array([1, 2, 0])
        expected = [[0, 0, 2], [0, 0, 1], [2, 1, 0]]
        assert_equal(perm_adj(adj, perm).toarray(), expected)

        perm = np.array([3, 0])
        expected = [[0, 0], [0, 0]]
        assert_equal(perm_adj(adj, perm).toarray(), expected)

        # Slice nodes and add fake nodes.
        perm = np.array([3, 1, 4, 0])
        expected = [[0, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 0], [0, 2, 0, 0]]
        assert_equal(perm_adj(adj, perm).toarray(), expected)

        perm = np.array([1, 4, 2])
        expected = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert_equal(perm_adj(adj, perm).toarray(), expected)

    def test_perm_features(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        perm = np.array([2, 1, 3, 0])
        expected = [[5, 6], [3, 4], [7, 8], [1, 2]]
        assert_equal(perm_features(features, perm), expected)

        # Add fake nodes.
        perm = np.array([3, 2, 0, 4, 1])
        expected = [[7, 8], [5, 6], [1, 2], [0, 0], [3, 4]]
        assert_equal(perm_features(features, perm), expected)

        # Slice nodes.
        perm = np.array([2, 1])
        expected = [[5, 6], [3, 4]]
        assert_equal(perm_features(features, perm), expected)

        # Slice nodes and add fake nodes.
        perm = np.array([3, 4, 0])
        expected = [[7, 8], [0, 0], [1, 2]]
        assert_equal(perm_features(features, perm), expected)

    def test_pad_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)

        expected = [[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]]
        assert_equal(pad_adj(adj, 4).toarray(), expected)

        expected = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        assert_equal(pad_adj(adj, 3).toarray(), expected)

    def test_pad_features(self):
        features = np.array([[0, 1], [2, 3], [4, 5]])

        expected = [[0, 1], [2, 3], [4, 5], [0, 0], [0, 0]]
        assert_equal(pad_features(features, 5), expected)

        expected = [[0, 1], [2, 3], [4, 5]]
        assert_equal(pad_features(features, 3), expected)
