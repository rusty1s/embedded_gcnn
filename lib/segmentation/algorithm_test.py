from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .algorithm import slic, quickshift
from ..datasets.mnist import MNIST


class AlgorithmTest(TestCase):
    def test_slic(self):
        data = MNIST('data/mnist')
        image = data.test.next_batch(1, shuffle=False)[0][0]
        segmentation = slic(
            image, num_segments=100, compactness=5, max_iterations=10, sigma=0)

        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))

    def test_quickshift(self):
        data = MNIST('data/mnist')
        image = data.test.next_batch(1, shuffle=False)[0][0]
        segmentation = quickshift(
            image, ratio=1, kernel_size=2, max_dist=2, sigma=0)

        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))
