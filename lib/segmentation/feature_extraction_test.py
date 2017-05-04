from unittest import TestCase

import numpy as np
from skimage.measure import regionprops
from numpy.testing import assert_equal, assert_almost_equal

from .feature_extraction_new import FeatureExtraction


def _convert(props, key):
    p = np.array([prop[key] for prop in props])
    if (p.ndim > 0):
        p = p.T
    return p


class FeatureExtractionTest(TestCase):
    def test_feature_extraction_minimal(self):
        segmentation = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])

        features = FeatureExtraction(segmentation)
        props = regionprops(segmentation + 1)

        # Test private helper
        assert_equal(features._min_y, [0, 0])
        assert_equal(features._max_y, [2, 2])
        assert_equal(features._min_x, [0, 2])
        assert_equal(features._max_x, [2, 4])

        assert_equal(features._M_00, [4, 4])
        assert_equal(features._M_01, [2, 2])
        assert_equal(features._M_10, [2, 10])
        assert_equal(features._M_11, [1, 5])
        assert_equal(features._M_02, [2, 2])
        assert_equal(features._M_20, [2, 26])
        assert_equal(features._M_12, [1, 5])
        assert_equal(features._M_21, [1, 13])
        assert_equal(features._M_03, [2, 2])
        assert_equal(features._M_30, [2, 70])

        assert_equal(features._centroid_y, _convert(props, 'centroid')[0])
        assert_equal(features._centroid_x, _convert(props, 'centroid')[1])

        # Test central moments.
        moments_central = _convert(props, 'moments_central')
        assert_equal(features.mu_11, moments_central[1, 1])
        assert_equal(features.mu_02, moments_central[2, 0])
        assert_equal(features.mu_20, moments_central[0, 2])
        assert_equal(features.mu_12, moments_central[2, 1])
        assert_equal(features.mu_21, moments_central[1, 2])
        assert_equal(features.mu_03, moments_central[3, 0])
        assert_equal(features.mu_30, moments_central[0, 3])

        inertia_tensor = _convert(props, 'inertia_tensor')
        assert_equal(features.inertia_tensor_02, inertia_tensor[1, 1])
        assert_equal(features.inertia_tensor_20, inertia_tensor[0, 0])
        assert_equal(features.inertia_tensor_11, inertia_tensor[1, 0])
        assert_equal(features.inertia_tensor_11, inertia_tensor[0, 1])

        eigvals = _convert(props, 'inertia_tensor_eigvals')
        assert_equal(features.inertia_tensor_eigvals_1, eigvals[0])
        assert_equal(features.inertia_tensor_eigvals_2, eigvals[1])

        # Tests normalized moments.
        moments_normalized = _convert(props, 'moments_normalized')
        assert_equal(features.nu_11, moments_normalized[1, 1])
        assert_equal(features.nu_02, moments_normalized[2, 0])
        assert_equal(features.nu_20, moments_normalized[0, 2])
        assert_equal(features.nu_12, moments_normalized[2, 1])
        assert_equal(features.nu_21, moments_normalized[1, 2])
        assert_equal(features.nu_03, moments_normalized[3, 0])
        assert_equal(features.nu_30, moments_normalized[0, 3])

        # Tests hu moments.
        moments_hu = _convert(props, 'moments_hu')
        assert_equal(features.hu_1, moments_hu[0])
        assert_equal(features.hu_2, moments_hu[1])
        assert_equal(features.hu_3, moments_hu[2])
        assert_equal(features.hu_4, moments_hu[3])
        assert_equal(features.hu_5, moments_hu[4])

    def test_feature_extraction(self):
        segmentation = np.array([[0, 1, 1, 4, 6, 6], [0, 0, 1, 4, 6, 7],
                                 [0, 3, 1, 5, 5, 7], [0, 2, 2, 2, 5, 7],
                                 [8, 8, 2, 5, 5, 9], [8, 8, 8, 9, 9, 9]])

        features = FeatureExtraction(segmentation)
        props = regionprops(segmentation + 1)

        moments_central = _convert(props, 'moments_central')
        assert_almost_equal(features.mu_11, moments_central[1, 1])
        assert_almost_equal(features.mu_02, moments_central[2, 0])
        assert_almost_equal(features.mu_20, moments_central[0, 2])
        assert_almost_equal(features.mu_12, moments_central[2, 1])
        assert_almost_equal(features.mu_21, moments_central[1, 2])
        assert_almost_equal(features.mu_03, moments_central[3, 0])
        assert_almost_equal(features.mu_30, moments_central[0, 3])

        inertia_tensor = _convert(props, 'inertia_tensor')
        assert_almost_equal(features.inertia_tensor_02, inertia_tensor[1, 1])
        assert_almost_equal(features.inertia_tensor_20, inertia_tensor[0, 0])
        nu_11 = -1 * inertia_tensor[0, 1]
        assert_almost_equal(features.inertia_tensor_11, nu_11)
        nu_11 = -1 * inertia_tensor[1, 0]
        assert_almost_equal(features.inertia_tensor_11, nu_11)

        eigvals = _convert(props, 'inertia_tensor_eigvals')
        assert_almost_equal(features.inertia_tensor_eigvals_1, eigvals[0])
        assert_almost_equal(features.inertia_tensor_eigvals_2, eigvals[1])

        moments_normalized = _convert(props, 'moments_normalized')
        assert_almost_equal(features.nu_11, moments_normalized[1, 1])
        assert_almost_equal(features.nu_02, moments_normalized[2, 0])
        assert_almost_equal(features.nu_20, moments_normalized[0, 2])
        assert_almost_equal(features.nu_12, moments_normalized[2, 1])
        assert_almost_equal(features.nu_21, moments_normalized[1, 2])
        assert_almost_equal(features.nu_03, moments_normalized[3, 0])
        assert_almost_equal(features.nu_30, moments_normalized[0, 3])

        moments_hu = _convert(props, 'moments_hu')
        assert_almost_equal(features.hu_1, moments_hu[0])
        assert_almost_equal(features.hu_2, moments_hu[1])
        assert_almost_equal(features.hu_3, moments_hu[2])
        assert_almost_equal(features.hu_4, moments_hu[3])
        assert_almost_equal(features.hu_5, moments_hu[4])
