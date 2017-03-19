from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .coarsening import coarsen_adj, _coarsen_clustered_adj, compute_perms


class CoarseningTest(TestCase):
    def test_coarsen_clustered_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)
        cluster_map = np.array([0, 0, 1, 1])

        expected_adj = [[0, 2], [2, 0]]
        adj_new = _coarsen_clustered_adj(cluster_map, adj)
        assert_equal(adj_new.toarray(), expected_adj)

        cluster_map = np.array([0, 1, 0, 1])

        expected_adj = [[0, 4], [4, 0]]
        adj_new = _coarsen_clustered_adj(cluster_map, adj)
        assert_equal(adj_new.toarray(), expected_adj)

        adj = [[0, 3, 0, 2, 0], [3, 0, 2, 0, 0], [0, 2, 0, 3, 0],
               [2, 0, 3, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        cluster_map = np.array([0, 0, 1, 1, 2])

        expected_adj = [[0, 4, 0], [4, 0, 1], [0, 1, 0]]
        adj_new = _coarsen_clustered_adj(cluster_map, adj)
        assert_equal(adj_new.toarray(), expected_adj)

        cluster_map = np.array([2, 1, 1, 0, 0])

        expected_adj = [[0, 3, 2], [3, 0, 3], [2, 3, 0]]
        adj_new = _coarsen_clustered_adj(cluster_map, adj)
        assert_equal(adj_new.toarray(), expected_adj)

    def test_compute_perms(self):
        cluster_map_1 = np.array([4, 1, 1, 2, 2, 3, 0, 0, 3])
        cluster_map_2 = np.array([2, 1, 0, 1, 0])
        cluster_maps = [cluster_map_1, cluster_map_2]

        perms = compute_perms(cluster_maps)

        assert_equal(len(perms), 3)
        assert_equal(perms[2], [0, 1, 2])
        assert_equal(perms[1], [2, 4, 1, 3, 0, 5])
        assert_equal(perms[0], [3, 4, 0, 9, 1, 2, 5, 8, 6, 7, 10, 11])

        cluster_map_1 = np.array([0, 0, 1, 1, 2])
        cluster_map_2 = np.array([1, 0, 0])
        cluster_maps = [cluster_map_1, cluster_map_2]

        perms = compute_perms(cluster_maps)

        assert_equal(len(perms), 3)
        assert_equal(perms[2], [0, 1])
        assert_equal(perms[1], [1, 2, 0, 3])
        assert_equal(perms[0], [2, 3, 4, 5, 0, 1, 6, 7])

    def test_coarsen_adj(self):
        adj = [[0, 3, 0, 2, 0], [3, 0, 2, 0, 0], [0, 2, 0, 3, 0],
               [2, 0, 3, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        rid = np.array([0, 1, 2, 3, 4])

        adjs, perm = coarsen_adj(adj, levels=1, rid=rid)

        assert_equal(perm, [0, 1, 2, 3, 4, 5])
        assert_equal(len(adjs), 2)
        assert_equal(adjs[0].shape, (6, 6))
        assert_equal(adjs[1].shape, (3, 3))

        expected_1 = [[0, 3, 0, 2, 0, 0], [3, 0, 2, 0, 0, 0],
                      [0, 2, 0, 3, 0, 0], [2, 0, 3, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]
        assert_equal(adjs[0].toarray(), expected_1)

        expected_2 = [[0, 4, 0], [4, 0, 1], [0, 1, 0]]
        assert_equal(adjs[1].toarray(), expected_2)

        adjs, perm = coarsen_adj(adj, levels=2, rid=rid)
        # After the first level, the rid swaps to [2, 0, 1] and thus we get
        # parents [0, 0, 1, 1, 2] and [1, 0, 0].

        assert_equal(perm, [2, 3, 4, 5, 0, 1, 6, 7])
        assert_equal(len(adjs), 3)
        assert_equal(adjs[0].shape, (8, 8))
        assert_equal(adjs[1].shape, (4, 4))
        assert_equal(adjs[2].shape, (2, 2))

        expected_1 = [[0, 3, 0, 0, 0, 2, 0, 0], [3, 0, 1, 0, 2, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 2, 0, 0, 0, 3, 0, 0], [2, 0, 0, 0, 3, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]

        assert_equal(adjs[0].toarray(), expected_1)

        expected_2 = [[0, 1, 4, 0], [1, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]]
        assert_equal(adjs[1].toarray(), expected_2)

        expected_3 = [[0, 4], [4, 0]]
        assert_equal(adjs[2].toarray(), expected_3)

        # Test random permutation.
        adjs, perm = coarsen_adj(adj, levels=2)
        assert_equal(len(adjs), 3)
        assert_equal(adjs[0].shape, (8, 8))
        assert_equal(adjs[1].shape, (4, 4))
        assert_equal(adjs[2].shape, (2, 2))

        assert_equal(perm.shape, [8])
        assert_equal(np.max(perm), 7)
