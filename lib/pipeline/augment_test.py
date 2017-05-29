from unittest import TestCase

import numpy as np
from numpy import pi as PI
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .augment import flip_left_right_adj, adjust_brightness, adjust_contrast


class AugmentTest(TestCase):
    def test_flip_left_right_adj(self):
        adj = [[0, 0.25 * PI, 0.75 * PI], [1.25 * PI, 0, 0], [1.75 * PI, 0, 0]]
        adj = sp.coo_matrix(adj)

        output = flip_left_right_adj(adj)

        expected = [[0, 1.75 * PI, 1.25 * PI], [0.75 * PI, 0, 0],
                    [0.25 * PI, 0, 0]]
        assert_equal(output.toarray(), expected)

    def test_adjust_brightness(self):
        features = np.array([[0.2, 0.4, 0.6, 1], [0.3, 0.2, 0.5, 2]])
        output = adjust_brightness(features, 0, 3, delta=0.5)

        assert_almost_equal(output, [[0.7, 0.9, 1, 1], [0.8, 0.7, 1, 2]])

    def test_adjust_contrast(self):
        features = np.array([[0.2, 0.4, 0.6, 1], [0.3, 0.2, 0.5, 2]])
        output = adjust_contrast(features, 0, 3, delta=-0.5)

        assert_almost_equal(output,
                            [[0.225, 0.35, 0.575, 1], [0.275, 0.25, 0.525, 2]])
