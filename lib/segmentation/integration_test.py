from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from .adjacency import segmentation_adjacency
from .feature_extraction import feature_extraction_minimal


class SegmentationIntegrationTest(TestCase):
    def test_adjacency_and_feature_extraction(self):
        segmentation = np.array([[0, 0, 0, 2], [0, 1, 1, 3], [1, 4, 4, 3]])
        image = np.array([[0, 0, 0, 2], [0, 1, 1, 3], [1, 4, 4, 3]])
        image = np.reshape(image, (3, 4, 1))

        points, adj, mass = segmentation_adjacency(segmentation)
        assert_equal(mass, [4, 3, 1, 2, 2])
        assert_almost_equal(
            points,
            [[3 / 4, 1 / 4], [1, 4 / 3], [3, 0], [3, 3 / 2], [3 / 2, 2]])
        assert_equal(adj.toarray(),
                     [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 0],
                      [1, 1, 1, 0, 1], [1, 1, 0, 1, 0]])

        features = feature_extraction_minimal(segmentation, image)
        assert_equal(features.shape, (5, 4))
        expected_features_1 = [4, 2, 3, 0]
        expected_features_2 = [3, 2, 3, 1]
        expected_features_3 = [1, 1, 1, 2]
        expected_features_4 = [2, 2, 1, 3]
        expected_features_5 = [2, 1, 2, 4]

        assert_equal(features[0], expected_features_1)
        assert_equal(features[1], expected_features_2)
        assert_equal(features[2], expected_features_3)
        assert_equal(features[3], expected_features_4)
        assert_equal(features[4], expected_features_5)
