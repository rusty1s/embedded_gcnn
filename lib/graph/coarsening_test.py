from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp

from .coarsening import (coarsen_adj, _coarsen_adj, _coarsen_clustered_adj,
                         _compute_perms)
from .adjacency import points_to_adj
from .distortion import perm_adj


class CoarseningCopyTest(TestCase):
    def test_compute_perms(self):
        cluster_maps = [
            np.array([4, 1, 1, 2, 2, 3, 0, 0, 3]), np.array([2, 1, 0, 1, 0])
        ]

        expected = [[3, 4, 0, 9, 1, 2, 5, 8, 6, 7, 10, 11], [2, 4, 1, 3, 0, 5],
                    [0, 1, 2]]

        assert_equal(_compute_perms(cluster_maps), expected)

    def test_coarsen_clustered_adj(self):
        adj = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0],
               [1, 0, 1, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        points = np.array([[1, 1], [3, 1], [3, 3], [1, 3], [0, 8]])
        mass = np.array([4, 6, 5, 2, 10])
        cluster_map = np.array([2, 1, 1, 0, 0])

        adj, points, mass = _coarsen_clustered_adj(adj, points, mass,
                                                   cluster_map)

        assert_equal(adj.toarray(), [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        assert_equal(mass, [12, 11, 4])
        assert_equal(points, [[2 / 12, (10 * 8 + 2 * 3) / 12],
                              [(6 * 3 + 5 * 3) / 11,
                               (6 * 1 + 5 * 3) / 11], [1, 1]])

        adj = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
               [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        points = np.array([[1, 1], [3, 2], [3, 0], [4, 1], [8, 3]])
        mass = np.array([1, 1, 1, 1, 1])
        cluster_map = np.array([0, 1, 0, 1, 2])

        adj, points, mass = _coarsen_clustered_adj(adj, points, mass,
                                                   cluster_map)

        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        assert_equal(points, [[2, 0.5], [3.5, 1.5], [8, 3]])
        assert_equal(mass, [2, 2, 1])

        adj = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
               [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        points = np.array([[1, 1], [3, 2], [3, 0], [4, 1], [8, 3]])
        mass = np.array([2, 1, 1, 2, 4])
        cluster_map = np.array([1, 1, 2, 0, 0])

        adj, points, mass = _coarsen_clustered_adj(adj, points, mass,
                                                   cluster_map)

        assert_equal(adj.toarray(), [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        assert_equal(points, [[40 / 6, 14 / 6], [5 / 3, 4 / 3], [3, 0]])
        assert_equal(mass, [6, 3, 1])

    def test_coarsen_adj_private(self):
        adj = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
               [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        points = np.array([[1, 1], [3, 2], [3, 0], [4, 1], [8, 3]])
        mass = np.array([4, 2, 2, 2, 8])
        levels = 2
        stddev = 1
        scale_invariance = False
        rid = np.array([2, 0, 1, 4, 3])

        adjs_dist, adjs_rad, cluster_maps = _coarsen_adj(
            adj, points, mass, levels, scale_invariance, stddev, rid)

        assert_equal(len(adjs_dist), 3)
        assert_equal(len(adjs_rad), 3)
        assert_equal(len(cluster_maps), 2)

        adj_dist_0, adj_rad_0 = points_to_adj(adj, points, scale_invariance,
                                              stddev)
        assert_equal(adjs_dist[0].toarray(), adj_dist_0.toarray())
        assert_equal(adjs_rad[0].toarray(), adj_rad_0.toarray())

        assert_equal(cluster_maps[0], [0, 1, 0, 1, 2])

        adj = sp.coo_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        points = np.array([[10 / 6, 4 / 6], [3.5, 1.5], [8, 3]])
        mass = np.array([6, 4, 8])

        adj_dist_1, adj_rad_1 = points_to_adj(adj, points, scale_invariance,
                                              stddev)
        assert_equal(adjs_dist[1].toarray(), adj_dist_1.toarray())
        assert_equal(adjs_rad[1].toarray(), adj_rad_1.toarray())

        # Second round rid by degree: [2, 0, 1]
        assert_equal(cluster_maps[1], [1, 0, 0])

        adj = sp.coo_matrix([[0, 1], [1, 0]])
        points = np.array([[(4 * 3.5 + 8 * 8) / 12, (4 * 1.5 + 8 * 3) / 12],
                           [10 / 6, 4 / 6]])
        mass = np.array([12, 6])

        adj_dist_2, adj_rad_2 = points_to_adj(adj, points, scale_invariance,
                                              stddev)
        assert_equal(adjs_dist[2].toarray(), adj_dist_2.toarray())
        assert_equal(adjs_rad[2].toarray(), adj_rad_2.toarray())

    def test_coarsen_adj(self):
        adj = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
               [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        points = np.array([[1, 1], [3, 2], [3, 0], [4, 1], [8, 3]])
        mass = np.array([4, 2, 2, 2, 8])
        levels = 2
        stddev = 1
        scale_invariance = False
        rid = np.array([2, 0, 1, 4, 3])

        adjs_dist, adjs_rad, perm = coarsen_adj(
            adj, points, mass, levels, scale_invariance, stddev, rid)

        assert_equal(len(adjs_dist), 3)
        assert_equal(len(adjs_rad), 3)

        perm_2 = np.array([0, 1])
        perm_1 = np.array([1, 2, 0, 3])
        perm_0 = np.array([1, 3, 4, 5, 0, 2, 6, 7])

        assert_equal(perm, perm_0)

        adj_dist_0, adj_rad_0 = points_to_adj(adj, points, scale_invariance,
                                              stddev)
        adj_dist_0 = perm_adj(adj_dist_0, perm_0)
        adj_rad_0 = perm_adj(adj_rad_0, perm_0)

        assert_equal(adjs_dist[0].toarray(), adj_dist_0.toarray())
        assert_equal(adjs_rad[0].toarray(), adj_rad_0.toarray())

        adj = sp.coo_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        points = np.array([[10 / 6, 4 / 6], [3.5, 1.5], [8, 3]])
        mass = np.array([6, 4, 8])

        adj_dist_1, adj_rad_1 = points_to_adj(adj, points, scale_invariance,
                                              stddev)
        adj_dist_1 = perm_adj(adj_dist_1, perm_1)
        adj_rad_1 = perm_adj(adj_rad_1, perm_1)

        assert_equal(adjs_dist[1].toarray(), adj_dist_1.toarray())
        assert_equal(adjs_rad[1].toarray(), adj_rad_1.toarray())

        adj = sp.coo_matrix([[0, 1], [1, 0]])
        points = np.array([[(4 * 3.5 + 8 * 8) / 12, (4 * 1.5 + 8 * 3) / 12],
                           [10 / 6, 4 / 6]])
        mass = np.array([12, 6])

        adj_dist_2, adj_rad_2 = points_to_adj(adj, points, scale_invariance,
                                              stddev)
        adj_dist_2 = perm_adj(adj_dist_2, perm_2)
        adj_rad_2 = perm_adj(adj_rad_2, perm_2)

        assert_equal(adjs_dist[2].toarray(), adj_dist_2.toarray())
        assert_equal(adjs_rad[2].toarray(), adj_rad_2.toarray())
