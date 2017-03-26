from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from .adjacency import segmentation_adjacency


class AdjacencyTest(TestCase):
    def test_segmentation_adjacency(self):
        # Test ascending ordering.
        segmentation = np.array([[0, 0, 0], [1, 1, 1]])
        points, adj, mass = segmentation_adjacency(segmentation)

        assert_equal(points, [[1, 0], [1, 1]])
        assert_equal(adj.toarray(), [[0, 1], [1, 0]])
        assert_equal(mass, [3, 3])

        # Test any ordering.
        segmentation = np.array([[1, 1, 1], [0, 0, 0]])
        points, adj, mass = segmentation_adjacency(segmentation)

        assert_equal(mass, [3, 3])
        assert_equal(points, [[1, 1], [1, 0]])
        assert_equal(adj.toarray(), [[0, 1], [1, 0]])

    def test_segmentation_adjacency_connectivity(self):
        segmentation = np.array([[0, 0, 0, 2], [0, 1, 1, 3], [1, 4, 4, 3]])
        points, adj, mass = segmentation_adjacency(
            segmentation, connectivity=1)

        assert_equal(mass, [4, 3, 1, 2, 2])
        assert_almost_equal(
            points,
            [[3 / 4, 1 / 4], [1, 4 / 3], [3, 0], [3, 3 / 2], [3 / 2, 2]])
        assert_equal(adj.toarray(),
                     [[0, 1, 1, 0, 0], [1, 0, 0, 1, 1], [1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 1], [0, 1, 0, 1, 0]])

        points, adj, mass = segmentation_adjacency(segmentation)
        assert_equal(mass, [4, 3, 1, 2, 2])
        assert_almost_equal(
            points,
            [[3 / 4, 1 / 4], [1, 4 / 3], [3, 0], [3, 3 / 2], [3 / 2, 2]])
        assert_equal(adj.toarray(),
                     [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 0],
                      [1, 1, 1, 0, 1], [1, 1, 0, 1, 0]])
