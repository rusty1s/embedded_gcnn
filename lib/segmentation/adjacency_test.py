from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .adjacency import segmentation_adjacency


class AdjacencyTest(TestCase):
    def test_segmentation_adjacency_simple(self):
        segmentation = np.array([[0, 0, 1, 1], [2, 2, 3, 3]])
        adj, points, mass = segmentation_adjacency(segmentation)

        assert_equal(adj.toarray(), [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1],
                                     [0, 1, 1, 0]])
        assert_equal(points, [[0, 0.5], [0, 2.5], [1, 0.5], [1, 2.5]])
        assert_equal(mass, [2, 2, 2, 2])

    def test_segmentation_adjacency_complex(self):
        segmentation = np.array([[0, 1, 1, 4, 6, 6], [0, 0, 1, 4, 6, 7],
                                 [0, 3, 1, 5, 5, 7], [0, 2, 2, 2, 5, 7],
                                 [8, 8, 2, 5, 5, 9], [8, 8, 8, 9, 9, 9]])
        adj, points, mass = segmentation_adjacency(segmentation)

        expected_adj = [
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0], [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
        ]

        expected_points = [[7 / 5, 1 / 5], [3 / 4, 7 / 4], [13 / 4, 8 / 4],
                           [2, 1], [1 / 2, 6 / 2], [15 / 5, 18 / 5],
                           [1 / 3, 13 / 3], [6 / 3, 15 / 3], [23 / 5, 4 / 5],
                           [19 / 4, 17 / 4]]

        assert_equal(adj.toarray(), expected_adj)
        assert_equal(points, expected_points)
        assert_equal(mass, [5, 4, 4, 1, 2, 5, 3, 3, 5, 4])
