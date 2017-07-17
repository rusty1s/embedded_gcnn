from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .form_feature_selection import FormFeatureSelection
from ..datasets import MNIST
from .algorithm import slic_fixed

data = MNIST('data/mnist')
slic = slic_fixed(num_segments=100, compactness=5, max_iterations=10)


class FormFeatureSelectionTest(TestCase):
    def test_form_feature_selection(self):
        selector = FormFeatureSelection(data.train, slic, num_examples=10)

        self.assertEqual(selector.features.shape, (655, 38))
        self.assertEqual(selector.num_features, 38)
        self.assertEqual(len(selector.selected_features), 38)
        assert_equal(selector.selected_feature_indices, np.arange(38))

        # Remove low variance: no features removed, because we use standard
        # scaler as default.
        selector.remove_low_variance(0.1)
        self.assertEqual(selector.selected_feature_indices.shape[0], 38)

        # Univariate feature selection.
        selector.select_univariate(12)
        self.assertEqual(selector.selected_feature_indices.shape[0], 12)

        # Recursive feature eliminiation.
        selector.eliminate_recursive(9)
        self.assertEqual(selector.selected_feature_indices.shape[0], 9)
