from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from .augment import (flip_left_right_image, random_flip_left_right_image,
                      adjust_brightness, random_brightness, adjust_contrast,
                      random_contrast)


class AugmentTest(TestCase):
    def test_flip_left_right_image(self):
        image = np.array([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ])

        expected = [
            [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]],
            [[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]],
        ]

        assert_equal(flip_left_right_image(image), expected)

        random = random_flip_left_right_image(image)
        self.assertTrue(
            np.array_equal(random, image) or np.array_equal(random, expected))

    def test_adjust_brightness(self):
        image = np.array([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ])

        expected = [
            [[0.6, 0.7, 0.8], [0.9, 1.0, 1.0]],
            [[1.0, 1.0, 0.9], [0.8, 0.7, 0.6]],
        ]

        assert_equal(adjust_brightness(image, 0.5), expected)

        self.assertGreaterEqual(random_brightness(image, 0.5).min(), 0)
        self.assertLessEqual(random_brightness(image, 0.5).max(), 1)

    def test_adjust_contrast(self):
        image = np.array([
            [[0.1, 0.4, 0.3], [0.4, 0.5, 0.6]],
            [[0.6, 0.5, 0.4], [0.3, 0.2, 0.5]],
        ])

        expected = [
            [[0, 0.4, 0.225], [0.425, 0.55, 0.675]],
            [[0.725, 0.55, 0.375], [0.275, 0.1, 0.525]],
        ]

        assert_almost_equal(adjust_contrast(image, 0.5), expected)

        self.assertGreaterEqual(random_contrast(image, 0.5).min(), 0)
        self.assertLessEqual(random_contrast(image, 0.5).max(), 1)
