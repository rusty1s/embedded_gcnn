from __future__ import division
from __future__ import print_function

import sys
from six.moves import xrange

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression

from .form_feature_extraction import FormFeatureExtraction


def _print_status(percentage):
    sys.stdout.write('\r>> Collecting features {:.2f}%'.format(percentage))
    sys.stdout.flush()


class FormFeatureSelection(object):
    def __init__(self, dataset, segmentation_algorithm, num_examples=None):

        if num_examples is None:
            num_examples = dataset.num_examples

        images, labels = dataset.next_batch(num_examples, shuffle=False)

        self._features = []
        self._labels = []

        for i in xrange(num_examples):
            segmentation = segmentation_algorithm(images[i])

            features = FormFeatureExtraction(segmentation).get_features()
            label = np.where(labels[i] == 1)[0][0]
            label = label.repeat(features.shape[0])

            self._features.append(features)
            self._labels.append(label)

            _print_status(100 * i / num_examples)

        _print_status(100)
        print()

        self._features = np.concatenate(self._features)
        self._idx = np.arange(self._features.shape[1])
        self._labels = np.concatenate(self._labels)

    @property
    def features(self):
        return self._features

    @property
    def num_features(self):
        return self._features.shape[1]

    @property
    def selected_feature_indices(self):
        return self._idx

    @property
    def selected_features(self):
        return [FormFeatureExtraction.methods[i] for i in self._idx]

    def remove_low_variance(self, threshold):
        sel = VarianceThreshold(threshold)
        self._features = sel.fit_transform(self._features)
        self._idx = self._idx[np.where(sel.get_support())[0]]

    def select_univariate(self, k):
        sel = SelectKBest(chi2, k)
        self._features = sel.fit_transform(self._features, self._labels)
        self._idx = self._idx[np.where(sel.get_support())[0]]

    def eliminiate_recursive(self, k):
        sel = RFE(LogisticRegression(), k)
        self._features = sel.fit_transform(self._features, self._labels)
        self._idx = self._idx[np.where(sel.get_support())[0]]
