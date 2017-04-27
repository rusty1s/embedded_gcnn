from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .embedding import (grid_points, points_to_embedded_adj,
                        partition_embedded_adj)
from .adjacency import grid_adj


class EmbeddingTest(TestCase):
    def test_grid_points(self):
        expected = [[0, 2], [1, 2], [0, 1], [1, 1], [0, 0], [1, 0]]
        assert_equal(grid_points((3, 2)), expected)

    def test_points_to_embedded_grid_4(self):
        points = np.array([[2, 2], [2, 3], [3, 2], [2, 1], [1, 2]])
        adj = sp.coo_matrix([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        adj_dist, adj_rad = points_to_embedded_adj(points, adj)

        expected_dist = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
        expected_rad = [[0, 2 * np.pi, 0.5 * np.pi, np.pi, 1.5 * np.pi],
                        [np.pi, 0, 0, 0, 0], [1.5 * np.pi, 0, 0, 0, 0],
                        [2 * np.pi, 0, 0, 0, 0], [0.5 * np.pi, 0, 0, 0, 0]]

        assert_equal(adj_dist.toarray(), expected_dist)
        assert_almost_equal(adj_rad.toarray(), expected_rad, decimal=6)

    def test_points_to_embedded_grid_8(self):
        points = np.array([[2, 2], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1],
                           [1, 1], [1, 2], [1, 3]])
        adj = sp.coo_matrix(
            [[0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0]])
        adj_dist, adj_rad = points_to_embedded_adj(points, adj)

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

    def test_partition_embedded_adj_connectivity_4(self):
        points = grid_points((3, 2))
        adj = grid_adj((3, 2), connectivity=4)
        adj_dist, adj_rad = points_to_embedded_adj(points, adj)

        adjs = partition_embedded_adj(
            adj_dist, adj_rad, num_partitions=4, offset=0.25 * np.pi)

        expected_adj_top = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        expected_adj_right = [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]]
        expected_adj_bottom = [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        expected_adj_left = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]

        assert_equal(len(adjs), 4)
        assert_equal(adjs[0].nnz + adjs[1].nnz + adjs[2].nnz + adjs[3].nnz, 14)
        for adj in adjs:
            assert_equal(adjs[0].toarray(), expected_adj_top)
            assert_equal(adjs[1].toarray(), expected_adj_right)
            assert_equal(adjs[2].toarray(), expected_adj_bottom)
            assert_equal(adjs[3].toarray(), expected_adj_left)

    def test_partition_embedded_adj_connectivity_8(self):
        points = grid_points((3, 2))
        adj = grid_adj((3, 2), connectivity=8)
        adj_dist, adj_rad = points_to_embedded_adj(points, adj)

        adjs = partition_embedded_adj(
            adj_dist, adj_rad, num_partitions=4, offset=0.375 * np.pi)

        expected_adj_top = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [1, 2, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 2, 0, 0], [0, 0, 0, 1, 0, 0]]
        expected_adj_right = [[0, 1, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 2], [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]]
        expected_adj_bottom = [[0, 0, 1, 0, 0, 0], [0, 0, 2, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 1],
                               [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        expected_adj_left = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [2, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 1, 0]]

        assert_equal(len(adjs), 4)
        assert_equal(adjs[0].nnz + adjs[1].nnz + adjs[2].nnz + adjs[3].nnz, 22)
        for adj in adjs:
            assert_equal(adjs[0].toarray(), expected_adj_top)
            assert_equal(adjs[1].toarray(), expected_adj_right)
            assert_equal(adjs[2].toarray(), expected_adj_bottom)
            assert_equal(adjs[3].toarray(), expected_adj_left)
