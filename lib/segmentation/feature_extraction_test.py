from unittest import TestCase

import numpy as np

from .feature_extraction import feature_extraction


class FeatureExtractionTest(TestCase):
    def test_feature_extraction(self):
        segmentation = np.array([[0, 1, 1], [0, 1, 1]])
        image = np.array([[[0, 0, 0], [1, 1, 1], [1, 1, 1]],
                          [[0, 0, 0], [1, 1, 1], [1, 1, 1]]], np.float32)

        feature_extraction(segmentation, image)
