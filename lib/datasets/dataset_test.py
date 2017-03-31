from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal

from .dataset import Dataset


class DatasetTest(TestCase):
    def test_init_dataset(self):
        data = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(data, labels)

        self.assertEqual(dataset.epochs_completed, 0)
        assert_equal(dataset._data, data)
        assert_equal(dataset._labels, labels)
        self.assertIsNone(dataset._postprocess)
        self.assertEqual(dataset._index_in_epoch, 0)

    def test_next_batch_numpy(self):
        data = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(data, labels)

        data_batch, labels_batch = dataset.next_batch(4, shuffle=False)
        assert_equal(data_batch, [1, 2, 3, 4])
        assert_equal(labels_batch, [0, 1, 0, 1])

        data_batch, labels_batch = dataset.next_batch(4, shuffle=False)
        assert_equal(data_batch, [5, 1, 2, 3])
        assert_equal(labels_batch, [0, 0, 1, 0])

        data_batch, labels_batch = dataset.next_batch(2, shuffle=False)
        assert_equal(data_batch, [4, 5])
        assert_equal(labels_batch, [1, 0])

        data_batch, labels_batch = dataset.next_batch(5, shuffle=False)
        assert_equal(data_batch, [1, 2, 3, 4, 5])
        assert_equal(labels_batch, [0, 1, 0, 1, 0])

    def test_next_batch_list(self):
        data = [1, 2, 3, 4, 5]
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(data, labels)

        data_batch, labels_batch = dataset.next_batch(4, shuffle=False)
        self.assertEqual(data_batch, [1, 2, 3, 4])
        assert_equal(labels_batch, [0, 1, 0, 1])

        data_batch, labels_batch = dataset.next_batch(4, shuffle=False)
        self.assertEqual(data_batch, [5, 1, 2, 3])
        assert_equal(labels_batch, [0, 0, 1, 0])

        data_batch, labels_batch = dataset.next_batch(2, shuffle=False)
        self.assertEqual(data_batch, [4, 5])
        assert_equal(labels_batch, [1, 0])

        data_batch, labels_batch = dataset.next_batch(5, shuffle=False)
        self.assertEqual(data_batch, [1, 2, 3, 4, 5])
        assert_equal(labels_batch, [0, 1, 0, 1, 0])

    def test_next_batch_shuffle_numpy(self):
        data = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(data, labels)

        data_batch_1, labels_batch_1 = dataset.next_batch(4, shuffle=True)
        self.assertEqual(data_batch_1.shape, (4,))
        self.assertEqual(labels_batch_1.shape, (4,))

        data_batch_2, labels_batch_2 = dataset.next_batch(1, shuffle=True)
        self.assertEqual(data_batch_2.shape, (1,))
        self.assertEqual(labels_batch_2.shape, (1,))

        data_batch = np.concatenate((data_batch_1, data_batch_2), axis=0)
        labels_batch = np.concatenate((labels_batch_1, labels_batch_2), axis=0)

        assert_equal(np.sort(data_batch), [1, 2, 3, 4, 5])
        assert_equal(np.sort(labels_batch), [0, 0, 0, 1, 1])

    def test_next_batch_shuffle_list(self):
        data = [1, 2, 3, 4, 5]
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(data, labels)

        data_batch_1, labels_batch_1 = dataset.next_batch(4, shuffle=True)
        self.assertEqual(len(data_batch_1), 4)
        self.assertEqual(labels_batch_1.shape, (4,))

        data_batch_2, labels_batch_2 = dataset.next_batch(1, shuffle=True)
        self.assertEqual(len(data_batch_2), 1)
        self.assertEqual(labels_batch_2.shape, (1,))

        data_batch = data_batch_1 + data_batch_2
        labels_batch = np.concatenate((labels_batch_1, labels_batch_2), axis=0)

        self.assertEqual(sorted(data_batch), [1, 2, 3, 4, 5])
        assert_equal(np.sort(labels_batch), [0, 0, 0, 1, 1])

    def test_next_batch_postprocess(self):
        data = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(data, labels)
        dataset._postprocess = lambda x: x + 1

        data_batch, labels_batch = dataset.next_batch(4, shuffle=False)
        assert_equal(data_batch, [2, 3, 4, 5])
        assert_equal(labels_batch, [0, 1, 0, 1])
