from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .algorithm import (slic, slic_fixed, quickshift, quickshift_fixed,
                        felzenszwalb, felzenszwalb_fixed)
from ..datasets import MNIST, Cifar10

mnist = MNIST('data/mnist')
cifar_10 = Cifar10('data/cifar_10')


class AlgorithmTest(TestCase):
    def test_slic(self):
        # Test grayscaled image.
        image = mnist.test.next_batch(1, shuffle=False)[0][0]
        segmentation = slic(
            image, num_segments=100, compactness=5, max_iterations=10, sigma=0)

        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))

        alg = slic_fixed(num_segments=100, compactness=5, max_iterations=10)
        assert_equal(alg(image), segmentation)

        # Test colorized image.
        image = cifar_10.test.next_batch(1, shuffle=False)[0][0]
        segmentation = alg(image)
        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))

    def test_quickshift(self):
        # Test grayscaled image.
        image = mnist.test.next_batch(1, shuffle=False)[0][0]
        segmentation = quickshift(
            image, ratio=1, kernel_size=2, max_dist=2, sigma=0)

        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))

        alg = quickshift_fixed(ratio=1, kernel_size=2, max_dist=2)
        assert_equal(alg(image), segmentation)

        # Test colorized image.
        image = cifar_10.test.next_batch(1, shuffle=False)[0][0]
        segmentation = alg(image)
        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))

    def test_felzenszwalb(self):
        # Test grayscaled image.
        image = mnist.test.next_batch(1, shuffle=False)[0][0]
        segmentation = felzenszwalb(image, scale=1, min_size=1)

        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))

        alg = felzenszwalb_fixed(scale=1, min_size=1)
        assert_equal(alg(image), segmentation)

        # Test colorized image.
        image = cifar_10.test.next_batch(1, shuffle=False)[0][0]
        segmentation = alg(image)
        idx = np.unique(segmentation)
        assert_equal(idx, np.arange(segmentation.max() + 1))
