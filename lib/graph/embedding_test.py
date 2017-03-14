from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .embedding import embedded_adj


class EmbeddingTest(TestCase):
    def test_embedded_adj(self):
        points = np.array([[1, 1], [3, 2], [4, -1]])
        neighbors = np.array([[0, 1], [0, 2], [1, 2]])

        expected = [[0, 5, 13], [5, 0, 10], [13, 10, 0]]

        assert_equal(embedded_adj(points, neighbors).toarray(), expected)
