from __future__ import division

from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from .feature_extraction import extract_features, extract_features_fixed


class FeatureExtractionTest(TestCase):
    def test_extract_features(self):
        segmentation = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])

        gray = np.array([[[0.1], [0.5], [0.1], [0.4]], [[0.3], [0.8], [0.1],
                                                        [0.0]]])

        features = extract_features(segmentation, gray, [0, 2, 3])
        expected = np.array([[1.7 / 4, 4, 2, 2], [0.6 / 4, 4, 2, 2]])
        assert_almost_equal(features, expected)

        rgb = np.array([[[0.1, 0.2, 0.4], [0.5, 0.3, 0.6], [0.1, 0.2, 1.0],
                         [0.4, 0.9, 1.0]], [[0.3, 0.4, 0.2], [0.8, 0.4, 0.2],
                                            [0.1, 0.4, 0.6], [0.0, 0.0, 1.0]]])

        features = extract_features(segmentation, rgb, [0, 2, 3])
        expected = [[1.7 / 4, 1.3 / 4, 1.4 / 4, 4, 2, 2],
                    [0.6 / 4, 1.5 / 4, 3.6 / 4, 4, 2, 2]]
        assert_almost_equal(features, expected)

    def test_extract_features_fixed(self):
        segmentation = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])

        gray = np.array([[[0.1], [0.5], [0.1], [0.4]], [[0.3], [0.8], [0.1],
                                                        [0.0]]])

        extract = extract_features_fixed([0, 2, 3])
        features = extract(segmentation, gray)
        expected = [[1.7 / 4, 4, 2, 2], [0.6 / 4, 4, 2, 2]]
        assert_almost_equal(features, expected)
