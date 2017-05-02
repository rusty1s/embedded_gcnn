from unittest import TestCase

import numpy as np

from .mnist import MNIST

data = MNIST('data/mnist', val_size=10000)


class MNISTTest(TestCase):
    def test_init(self):
        self.assertEqual(data.train.num_examples, 50000)
        self.assertEqual(data.val.num_examples, 10000)
        self.assertEqual(data.test.num_examples, 10000)

    def test_shapes(self):
        images, labels = data.train.next_batch(32, shuffle=False)
        self.assertEqual(images.shape, (32, 28, 28, 1))
        self.assertEqual(labels.shape, (32, 10))
        data.train.next_batch(data.train.num_examples - 32, shuffle=False)

        images, labels = data.val.next_batch(32, shuffle=False)
        self.assertEqual(images.shape, (32, 28, 28, 1))
        self.assertEqual(labels.shape, (32, 10))
        data.val.next_batch(data.val.num_examples - 32, shuffle=False)

        images, labels = data.test.next_batch(32, shuffle=False)
        self.assertEqual(images.shape, (32, 28, 28, 1))
        self.assertEqual(labels.shape, (32, 10))
        data.test.next_batch(data.test.num_examples - 32, shuffle=False)

    def test_images(self):
        images, _ = data.train.next_batch(
            data.train.num_examples, shuffle=False)

        self.assertEqual(images.dtype, np.float32)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

        images, _ = data.val.next_batch(data.val.num_examples, shuffle=False)

        self.assertEqual(images.dtype, np.float32)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

        images, _ = data.test.next_batch(data.test.num_examples, shuffle=False)

        self.assertEqual(images.dtype, np.float32)
        self.assertLessEqual(images.max(), 1)
        self.assertGreaterEqual(images.min(), 0)

    def test_labels(self):
        _, labels = data.train.next_batch(
            data.train.num_examples, shuffle=False)

        self.assertEqual(labels.dtype, np.uint8)

    def test_label_functions(self):
        self.assertEqual(data.labels,
                         ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertEqual(data.num_labels, 10)

        _, labels = data.test.next_batch(5, shuffle=False)
        self.assertEqual(labels.dtype, np.uint8)

        self.assertEqual(data.label_name(labels[0]), ['7'])
        self.assertEqual(data.label_name(labels[1]), ['2'])
        self.assertEqual(data.label_name(labels[2]), ['1'])
        self.assertEqual(data.label_name(labels[3]), ['0'])
        self.assertEqual(data.label_name(labels[4]), ['4'])

        data.test.next_batch(data.test.num_examples - 5, shuffle=False)
