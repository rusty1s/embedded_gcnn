from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .feature_extraction import feature_extraction_minimal


class FeatureExtractionTest(TestCase):
    def test_feature_extraction_minimal(self):
        segmentation = np.array([[0, 1, 1], [0, 1, 1]])
        image = np.array([[[0, 0, 0], [1, 1, 1], [1, 1, 1]],
                          [[0, 0, 0], [1, 1, 1], [1, 1, 1]]], np.float32)

        features = feature_extraction_minimal(segmentation, image)

        expected = [[2, 2, 1, 0], [4, 2, 2, 1]]

        assert_equal(features.shape, (2, 4))
        assert_equal(features, expected)
