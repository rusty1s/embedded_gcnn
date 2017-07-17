from unittest import TestCase

import numpy as np
from numpy import pi as PI
from numpy.testing import assert_equal, assert_almost_equal
import scipy.sparse as sp

from .augment import (flip_left_right_adj, random_flip_left_right_adjs,
                      adjust_brightness, random_brightness, adjust_contrast,
                      random_contrast, augment_batch)


class AugmentTest(TestCase):
    def test_flip_left_right_adj(self):
        adj = [[0, 0.25 * PI, 0.75 * PI], [1.25 * PI, 0, 0], [1.75 * PI, 0, 0]]
        adj = sp.coo_matrix(adj)

        output = flip_left_right_adj(adj)

        expected = [[0, 1.75 * PI, 1.25 * PI], [0.75 * PI, 0, 0],
                    [0.25 * PI, 0, 0]]
        assert_equal(output.toarray(), expected)

        # Flip left right is mutable, so we create a new adjacency matrix.
        adj = [[0, 0.25 * PI, 0.75 * PI], [1.25 * PI, 0, 0], [1.75 * PI, 0, 0]]
        adj = sp.coo_matrix(adj)

        assert_equal(
            random_flip_left_right_adjs([adj], True)[0].toarray(), expected)
        assert_equal(
            random_flip_left_right_adjs([adj], False)[0].toarray(),
            adj.toarray())

        random = random_flip_left_right_adjs([adj])[0].toarray()
        adj = [[0, 0.25 * PI, 0.75 * PI], [1.25 * PI, 0, 0], [1.75 * PI, 0, 0]]

        self.assertTrue(
            np.array_equal(random, adj) or np.array_equal(random, expected))

    def test_adjust_brightness(self):
        features = np.array([[0.2, 0.4, 0.6, 1], [0.3, 0.2, 0.5, 2]])
        output = adjust_brightness(features, 0, 3, delta=0.5)

        assert_almost_equal(output, [[0.7, 0.9, 1, 1], [0.8, 0.7, 1, 2]])

        self.assertGreaterEqual(
            random_brightness(features, 0, 3, 0.5).min(), 0)
        self.assertLessEqual(random_brightness(features, 0, 3, 0.5).min(), 1)

    def test_adjust_contrast(self):
        features = np.array([[0.2, 0.4, 0.6, 1], [0.3, 0.2, 0.5, 2]])
        output = adjust_contrast(features, 0, 3, delta=-0.5)

        assert_almost_equal(output, [[0.225, 0.35, 0.575, 1],
                                     [0.275, 0.25, 0.525, 2]])

        self.assertGreaterEqual(random_contrast(features, 0, 3, 0.5).min(), 0)
        self.assertLessEqual(random_contrast(features, 0, 3, 0.5).min(), 1)

    def test_augment_batch(self):
        features = np.array([[0.2, 0.4], [0.3, 0.2], [0.8, 0.9]])
        adj_dist = [[0, 1, 2], [1, 0, 0], [2, 0, 0]]
        adj_dist = sp.coo_matrix(adj_dist)
        adj_rad = [[0, 0.25 * PI, 0.75 * PI], [1.25 * PI, 0, 0],
                   [1.75 * PI, 0, 0]]
        adj_rad = sp.coo_matrix(adj_rad)
        label = np.array([0, 0, 1])

        batch = augment_batch([(features, [adj_dist], [adj_rad], label)])

        expected = [[0, 1.75 * PI, 1.25 * PI], [0.75 * PI, 0, 0],
                    [0.25 * PI, 0, 0]]
        adj_rad = [[0, 0.25 * PI, 0.75 * PI], [1.25 * PI, 0, 0],
                   [1.75 * PI, 0, 0]]

        self.assertGreaterEqual(batch[0][0].min(), 0)
        self.assertLessEqual(batch[0][0].max(), 1)
        assert_equal(batch[0][1][0].toarray(), adj_dist.toarray())

        self.assertTrue(
            np.array_equal(batch[0][2][0].toarray(), expected) or
            np.array_equal(batch[0][2][0].toarray(), adj_rad))

        assert_equal(batch[0][3], label)
