from unittest import TestCase

from numpy.testing import assert_equal

from .dataset import PreprocessedDataset
from ..datasets.mnist import MNIST
from ..datasets.dataset import Dataset
from ..segmentation.algorithm import slic_fixed
from ..segmentation.feature_extraction import extract_features_fixed

dataset = MNIST('data/mnist').train
# Reduce dataset to 100 examples.
dataset = Dataset(dataset._images[:50], dataset._labels[:50])

segmentation_algorithm = slic_fixed(
    num_segments=100, compactness=5, max_iterations=10, sigma=0)
feature_extraction_algorithm = extract_features_fixed([0, 1, 2])

dataset = PreprocessedDataset(dataset, segmentation_algorithm,
                              feature_extraction_algorithm, 4)


class PreprocessedDatasetTest(TestCase):
    def test_init(self):
        self.assertEqual(dataset.num_examples, 50)

        features, adjs_dist, adjs_rad, label = dataset._data[0]
        self.assertEqual(features.shape, (80, 4))
        self.assertEqual(len(adjs_dist), 5)
        self.assertEqual(len(adjs_rad), 5)
        assert_equal(label, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    def test_next_batch(self):
        batch = dataset.next_batch(10, shuffle=False)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), 4)
        self.assertEqual(batch[0][0].shape, (80, 4))
        self.assertEqual(len(batch[0][1]), 5)
        self.assertEqual(len(batch[0][2]), 5)
        assert_equal(batch[0][3], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        batch = dataset.next_batch(dataset.num_examples - 10, shuffle=False)

        batch = dataset.next_batch(dataset.num_examples, shuffle=False)
        self.assertEqual(len(batch), 50)
        self.assertEqual(len(batch[0]), 4)
        self.assertEqual(batch[0][0].shape, (80, 4))
        self.assertEqual(len(batch[0][1]), 5)
        self.assertEqual(len(batch[0][2]), 5)
        assert_equal(batch[0][3], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    def test_next_batch_shuffle(self):
        batch = dataset.next_batch(10, shuffle=True)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), 4)

        batch = dataset.next_batch(dataset.num_examples - 10, shuffle=True)

        batch = dataset.next_batch(dataset.num_examples, shuffle=True)
        self.assertEqual(len(batch), 50)
        self.assertEqual(len(batch[0]), 4)
