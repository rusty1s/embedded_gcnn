from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .embedded_coarsening import _coarsen_clustered_embedded_adj


class EmbeddedCoarseningCopyTest(TestCase):
    def test_coarsen_clustered_adj(self):
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

        expected_points = [[40/6, 14/6], [5/3, 4/3], [3, 0]]
        expected_mass = [6, 3, 1]
        expected_adj = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

        assert_almost_equal(points_new, expected_points, decimal=6)
        assert_equal(mass_new, expected_mass)
        assert_equal(adj_new.toarray(), expected_adj)
