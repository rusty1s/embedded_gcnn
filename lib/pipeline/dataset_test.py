from unittest import TestCase

from numpy.testing import assert_equal

from .dataset import PreprocessedDataset
from ..datasets.mnist import MNIST
from ..segmentation.algorithm import slic_fixed
from ..segmentation.feature_extraction import extract_features_fixed
from ..pipeline import preprocess_pipeline_fixed

# Load MNIST dataset and reduce size to 50 examples.
mnist = MNIST('data/mnist').train
mnist._images = mnist._images[:50]
mnist._labels = mnist._labels[:50]

segmentation_algorithm = slic_fixed(
    num_segments=100, compactness=5, max_iterations=10, sigma=0)
feature_extraction_algorithm = extract_features_fixed([0, 1, 2])
preprocess_algorithm = preprocess_pipeline_fixed(
    segmentation_algorithm, feature_extraction_algorithm, levels=4)


class PreprocessedDatasetTest(TestCase):
    def test_init(self):
        dataset = PreprocessedDataset(mnist, preprocess_algorithm)
        features, adjs_dist, adjs_rad, label = dataset._data[0]

        self.assertEqual(dataset.num_examples, 50)
        self.assertEqual(features.shape, (80, 4))
        self.assertEqual(len(adjs_dist), 5)
        self.assertEqual(len(adjs_rad), 5)
        assert_equal(label, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    def test_next_batch(self):
        dataset = PreprocessedDataset(mnist, preprocess_algorithm)

        batch = dataset.next_batch(10, shuffle=False)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), 4)
        self.assertEqual(batch[0][0].shape, (80, 4))
        self.assertEqual(len(batch[0][1]), 5)
        self.assertEqual(len(batch[0][2]), 5)
        assert_equal(batch[0][3], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

        dataset.next_batch(dataset.num_examples - 10, shuffle=False)

        batch = dataset.next_batch(dataset.num_examples, shuffle=False)
        self.assertEqual(len(batch), 50)
        self.assertEqual(len(batch[0]), 4)
        self.assertEqual(batch[0][0].shape, (80, 4))
        self.assertEqual(len(batch[0][1]), 5)
        self.assertEqual(len(batch[0][2]), 5)
        assert_equal(batch[0][3], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    def test_next_batch_shuffle(self):
        dataset = PreprocessedDataset(mnist, preprocess_algorithm)

        batch = dataset.next_batch(10, shuffle=True)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), 4)

        dataset.next_batch(dataset.num_examples - 10, shuffle=True)

        batch = dataset.next_batch(dataset.num_examples, shuffle=True)
        self.assertEqual(len(batch), 50)
        self.assertEqual(len(batch[0]), 4)
