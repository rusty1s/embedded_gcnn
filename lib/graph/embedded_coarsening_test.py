from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .embedded_coarsening import (_coarsen_clustered_embedded_adj,
                                  coarsen_embedded_adj)
from .adjacency import normalize_adj, invert_adj


class EmbeddedCoarseningCopyTest(TestCase):
    def test_coarsen_clustered_embedded_adj(self):
        points = np.array([[1, 1], [3, 2], [3, 0], [4, 1], [8, 3]], np.float32)
        mass = np.array([1, 1, 1, 1, 1], np.float32)
        adj = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
               [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        cluster_map = np.array([0, 1, 0, 1, 2])

        points_new, mass_new, adj_new = _coarsen_clustered_embedded_adj(
            cluster_map, points, mass, adj)

        expected_points = [[2, 0.5], [3.5, 1.5], [8, 3]]
        expected_mass = [2, 2, 1]
        expected_adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

        assert_equal(points_new, expected_points)
        assert_equal(mass_new, expected_mass)
        assert_equal(adj_new.toarray(), expected_adj)

        mass = np.array([2, 1, 1, 2, 4], np.float32)
        cluster_map = np.array([1, 1, 2, 0, 0])

        points_new, mass_new, adj_new = _coarsen_clustered_embedded_adj(
            cluster_map, points, mass, adj)

        expected_points = [[40 / 6, 14 / 6], [5 / 3, 4 / 3], [3, 0]]
        expected_mass = [6, 3, 1]
        expected_adj = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

        assert_almost_equal(points_new, expected_points, decimal=6)
        assert_equal(mass_new, expected_mass)
        assert_equal(adj_new.toarray(), expected_adj)

    def test_coarsen_embedded_adj(self):
        points = np.array([[1, 1], [3, 2], [3, 0], [4, 1], [8, 3]], np.float32)
        adj = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 0],
               [0, 1, 1, 0, 1], [0, 0, 0, 1, 0]]
        adj = sp.coo_matrix(adj)
        mass = np.array([1, 1, 1, 1, 1], np.float32)
        stddev = 1
        rid = np.array([0, 1, 2, 3, 4])

        adjs_dist, adjs_rad, perm = coarsen_embedded_adj(
            points, mass, adj, levels=1, stddev=stddev, rid=rid)

        assert_equal(perm, [0, 1, 2, 3, 4, 5])
        assert_equal(len(adjs_dist), 2)
        assert_equal(adjs_dist[0].shape, (6, 6))
        assert_equal(adjs_dist[1].shape, (3, 3))
        assert_equal(len(adjs_rad), 2)
        assert_equal(adjs_rad[0].shape, (6, 6))
        assert_equal(adjs_rad[1].shape, (3, 3))

        expected_dist_1 = [[0, 5, 5, 0, 0, 0], [5, 0, 0, 2, 0, 0],
                           [5, 0, 0, 2, 0, 0], [0, 2, 2, 0, 20, 0],
                           [0, 0, 0, 20, 0, 0], [0, 0, 0, 0, 0, 0]]
        expected_dist_1 = invert_adj(
            normalize_adj(sp.coo_matrix(expected_dist_1)),
            stddev=stddev).toarray()
        assert_almost_equal(adjs_dist[0].toarray(), expected_dist_1)

        expected_dist_2 = [[0, 3.25, 0], [3.25, 0, 26.5], [0, 26.5, 0]]
        expected_dist_2 = invert_adj(
            normalize_adj(sp.coo_matrix(expected_dist_2)),
            stddev=stddev).toarray()
        assert_almost_equal(adjs_dist[1].toarray(), expected_dist_2)

        expected_rad_1 = [
            [0, np.arctan2(2, 1), np.arctan2(2, -1), 0, 0, 0],
            [np.arctan2(2, 1) + np.pi, 0, 0, 0.75 * np.pi, 0, 0],
            [np.arctan2(2, -1) + np.pi, 0, 0, 0.25 * np.pi, 0, 0], [
                0, 1.75 * np.pi, 1.25 * np.pi, 0, np.arctan2(4, 2), 0
            ], [0, 0, 0, np.arctan2(4, 2) + np.pi, 0, 0], [0, 0, 0, 0, 0, 0]
        ]
        assert_almost_equal(adjs_rad[0].toarray(), expected_rad_1, decimal=6)

        expected_rad_2 = [
            [0, np.arctan2(1.5, -1), 0],
            [np.arctan2(1.5, -1) + np.pi, 0, np.arctan2(4.5, 2.5)],
            [0, np.arctan2(4.5, 2.5) + np.pi, 0]
        ]
        assert_almost_equal(adjs_rad[1].toarray(), expected_rad_2, decimal=6)
