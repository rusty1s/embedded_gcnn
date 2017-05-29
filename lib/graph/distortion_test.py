from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .distortion import (perm_adj, perm_features, filter_adj, filter_features,
                         gray_color_threshold, degree_threshold,
                         area_threshold)


class DistortionTest(TestCase):
    def test_perm_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)

        perm = np.array([2, 1, 3, 0])
        expected = [[0, 0, 2, 1], [0, 0, 1, 2], [2, 1, 0, 0], [1, 2, 0, 0]]
        assert_equal(perm_adj(adj, perm).toarray(), expected)

        # Add fake nodes.
        perm = np.array([3, 2, 0, 4, 1, 5])
        expected = [[0, 2, 0, 0, 1, 0], [2, 0, 1, 0, 0, 0], [0, 1, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        assert_equal(perm_adj(adj, perm).toarray(), expected)

    def test_perm_features(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        perm = np.array([2, 1, 3, 0])
        expected = [[5, 6], [3, 4], [7, 8], [1, 2]]
        assert_equal(perm_features(features, perm), expected)

        # Add fake nodes.
        perm = np.array([3, 2, 0, 4, 1, 5])
        expected = [[7, 8], [5, 6], [1, 2], [0, 0], [3, 4], [0, 0]]
        assert_equal(perm_features(features, perm), expected)

    def test_filter_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)

        nodes = np.array([0, 1, 3])
        expected = [[0, 2, 0], [2, 0, 1], [0, 1, 0]]
        assert_equal(filter_adj(adj, nodes).toarray(), expected)

    def test_filter_features(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        nodes = np.array([1, 3])
        expected = [[3, 4], [7, 8]]
        assert_equal(filter_features(features, nodes), expected)

    def test_gray_color_threshold(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        nodes = gray_color_threshold(None, features, 3)
        expected = [1, 2, 3]
        assert_equal(nodes, expected)

    def test_degree_threshold(self):
        adj = [[0, 1, 1, 1, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0],
               [1, 0, 1, 0, 1], [1, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)

        nodes = degree_threshold(adj, None, 3)
        expected = [1, 2, 3, 4]
        assert_equal(nodes, expected)

    def test_area_threshold(self):
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        nodes = area_threshold(None, features, 4, idx=1)
        expected = [0, 1]
        assert_equal(nodes, expected)
