from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .dataset import Datasets, Dataset


class DatasetTest(TestCase):
    def test_init_dataset(self):
        images = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(images, labels)

        self.assertEqual(dataset.epochs_completed, 0)
        self.assertEqual(dataset._index_in_epoch, 0)
        self.assertEqual(dataset.num_examples, 5)

    def test_next_batch(self):
        images = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(images, labels)

        batch = dataset.next_batch(4, shuffle=False)
        assert_equal(batch[0], [1, 2, 3, 4])
        assert_equal(batch[1], [0, 1, 0, 1])

        batch = dataset.next_batch(4, shuffle=False)
        assert_equal(batch[0], [5, 1, 2, 3])
        assert_equal(batch[1], [0, 0, 1, 0])

        batch = dataset.next_batch(2, shuffle=False)
        assert_equal(batch[0], [4, 5])
        assert_equal(batch[1], [1, 0])

        batch = dataset.next_batch(5, shuffle=False)
        assert_equal(batch[0], [1, 2, 3, 4, 5])
        assert_equal(batch[1], [0, 1, 0, 1, 0])

    def test_next_batch_shuffle(self):
        images = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(images, labels)

        batch_1 = dataset.next_batch(4, shuffle=True)
        self.assertEqual(batch_1[0].shape, (4, ))
        self.assertEqual(batch_1[1].shape, (4, ))

        batch_2 = dataset.next_batch(1, shuffle=True)
        self.assertEqual(batch_2[0].shape, (1, ))
        self.assertEqual(batch_2[1].shape, (1, ))

        images = np.concatenate((batch_1[0], batch_2[0]), axis=0)
        labels = np.concatenate((batch_1[1], batch_2[1]), axis=0)

        assert_equal(np.sort(images), [1, 2, 3, 4, 5])
        assert_equal(np.sort(labels), [0, 0, 0, 1, 1])

        images, labels = dataset.next_batch(5, shuffle=True)
        assert_equal(np.sort(images), [1, 2, 3, 4, 5])
        assert_equal(np.sort(labels), [0, 0, 0, 1, 1])

    def test_init_datasets(self):
        images = [1, 2, 3, 4, 5]
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(images, labels)
        datasets = Datasets(dataset, dataset, dataset)

        self.assertRaises(NotImplementedError, getattr, datasets, 'classes')
        self.assertRaises(NotImplementedError, getattr, datasets, 'width')
        self.assertRaises(NotImplementedError, getattr, datasets, 'height')
        self.assertRaises(NotImplementedError, getattr, datasets,
                          'num_channels')

        batch = datasets.train.next_batch(5, shuffle=False)
        assert_equal(images, batch[0])
        assert_equal(labels, batch[1])
        batch = datasets.val.next_batch(5, shuffle=False)
        assert_equal(images, batch[0])
        assert_equal(labels, batch[1])
        batch = datasets.test.next_batch(5, shuffle=False)
        assert_equal(images, batch[0])
        assert_equal(labels, batch[1])
