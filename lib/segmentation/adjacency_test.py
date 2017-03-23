from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .adjacency import segmentation_adjacency


class AdjacencyTest(TestCase):
    def test_segmentation_adjacency(self):
        segmentation = np.array([[0, 0, 0], [1, 1, 1]])
        points, adj, mass = segmentation_adjacency(segmentation)

        assert_equal(points, [[0, 1], [1, 1]])
        assert_equal(adj.toarray(), [[0, 1], [1, 0]])
        assert_equal(mass, [3, 3])

        segmentation = np.array([[1, 1, 1], [0, 0, 0]])
        points, adj, mass = segmentation_adjacency(segmentation)

        assert_equal(points, [[1, 1], [0, 1]])
        assert_equal(adj.toarray(), [[0, 1], [1, 0]])
        assert_equal(mass, [3, 3])

        # TODO: More tests
