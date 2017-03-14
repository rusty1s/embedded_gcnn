from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from .embedding import grid_embedded_adj


class EmbeddingTest(TestCase):
    def test_grid_embedded_adj(self):
        adj_dist, adj_rad = grid_embedded_adj((3, 2), connectivity=4)

        expected_dist = [[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0],
                         [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1],
                         [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 1, 0]]
        expected_rad = [[0, 0.5 * np.pi, np.pi, 0, 0, 0],
                        [1.5 * np.pi, 0, 0, np.pi, 0, 0],
                        [2 * np.pi, 0, 0, 0.5 * np.pi, np.pi, 0],
                        [0, 2 * np.pi, 1.5 * np.pi, 0, 0, np.pi],
                        [0, 0, 2 * np.pi, 0, 0, 0.5 * np.pi],
                        [0, 0, 0, 2 * np.pi, 1.5 * np.pi, 0]]

        assert_equal(adj_dist.toarray(), expected_dist)
        assert_almost_equal(adj_rad.toarray(), expected_rad, decimal=6)

        adj_dist, adj_rad = grid_embedded_adj((3, 2), connectivity=8)

        expected_dist = [[0, 1, 1, 2, 0, 0], [1, 0, 2, 1, 0, 0],
                         [1, 2, 0, 1, 1, 2], [2, 1, 1, 0, 2, 1],
                         [0, 0, 1, 2, 0, 1], [0, 0, 2, 1, 1, 0]]
        expected_rad = [
            [0, 0.5 * np.pi, np.pi, 0.75 * np.pi, 0, 0],
            [1.5 * np.pi, 0, 1.25 * np.pi, np.pi, 0, 0],
            [2 * np.pi, 0.25 * np.pi, 0, 0.5 * np.pi, np.pi, 0.75 * np.pi],
            [1.75 * np.pi, 2 * np.pi, 1.5 * np.pi, 0, 1.25 * np.pi, np.pi],
            [0, 0, 2 * np.pi, 0.25 * np.pi, 0, 0.5 * np.pi],
            [0, 0, 1.75 * np.pi, 2 * np.pi, 1.5 * np.pi, 0]
        ]

        assert_equal(adj_dist.toarray(), expected_dist)
        assert_almost_equal(adj_rad.toarray(), expected_rad, decimal=6)
