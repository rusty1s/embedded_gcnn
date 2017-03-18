from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .points import points_to_embedded


class PointsTest(TestCase):
    def test_points_to_embedded_grdi_4(self):
        points = np.array([[2, 2], [2, 3], [3, 2], [2, 1], [1, 2]])
        adj = sp.coo_matrix([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        adj_dist, adj_rad = points_to_embedded(points, adj)

        expected_dist = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
        expected_rad = [[0, 2 * np.pi, 0.5 * np.pi, np.pi, 1.5 * np.pi],
                        [np.pi, 0, 0, 0, 0], [1.5 * np.pi, 0, 0, 0, 0],
                        [2 * np.pi, 0, 0, 0, 0], [0.5 * np.pi, 0, 0, 0, 0]]

        assert_equal(adj_dist.toarray(), expected_dist)
        assert_almost_equal(adj_rad.toarray(), expected_rad, decimal=6)

    def test_points_to_embedded_grdi_8(self):
        points = np.array([[2, 2], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1],
                           [1, 1], [1, 2], [1, 3]])
        adj = sp.coo_matrix(
            [[0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0]])
        adj_dist, adj_rad = points_to_embedded(points, adj)

        expected_dist = [
            [0, 1, 2, 1, 2, 1, 2, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        expected_rad = [[
            0, 2 * np.pi, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi,
            1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi
        ], [np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1.25 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1.5 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1.75 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.25 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.5 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0.75 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0]]

        assert_equal(adj_dist.toarray(), expected_dist)
        assert_almost_equal(adj_rad.toarray(), expected_rad, decimal=6)
