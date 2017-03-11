from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .coarsening import _cluster_adj, _coarsen_adj, _compute_perms


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

    def test_coarsen_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)
        rid = np.array([0, 1, 2, 3])
        expected_adj = [[0, 2], [2, 0]]
        expected_cluster_map = [0, 0, 1, 1]
        adj, cluster_map = _coarsen_adj(adj, rid)
        assert_equal(adj.toarray(), expected_adj)
        assert_equal(cluster_map, expected_cluster_map)

        adj = [[0, 3, 2, 0, 0], [3, 0, 0, 2, 0], [2, 0, 0, 3, 0],
               [0, 2, 3, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        rid = np.array([0, 1, 2, 3, 4])
        expected_adj = [[0, 4, 0], [4, 0, 1], [0, 1, 0]]
        expected_cluster_map = [0, 0, 1, 1, 2]
        adj, cluster_map = _coarsen_adj(adj, rid)
        assert_equal(adj.toarray(), expected_adj)
        assert_equal(cluster_map, expected_cluster_map)

    def test_compute_perms(self):
        cluster_map_1 = np.array([4, 1, 1, 2, 2, 3, 0, 0, 3])
        cluster_map_2 = np.array([2, 1, 0, 1, 0])
        cluster_maps = [cluster_map_1, cluster_map_2]

        perms = _compute_perms(cluster_maps)

        assert_equal(len(perms), 3)
        assert_equal(perms[2], [0, 1, 2])
        assert_equal(perms[1], [2, 4, 1, 3, 0, 5])
        assert_equal(perms[0], [3, 4, 0, 9, 1, 2, 5, 8, 6, 7, 10, 11])
