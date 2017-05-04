from unittest import TestCase

import numpy as np

from .cifar_10 import Cifar10

data = Cifar10('data/cifar_10', val_size=10000)


class Cifar10Test(TestCase):
    def test_init(self):
        self.assertEqual(data.train.num_examples, 40000)
        self.assertEqual(data.val.num_examples, 10000)
        self.assertEqual(data.test.num_examples, 10000)

    def test_shapes(self):
        images, labels = data.train.next_batch(32, shuffle=False)
        self.assertEqual(images.shape, (32, 32, 32, 3))
        self.assertEqual(labels.shape, (32, 10))
        data.train.next_batch(data.train.num_examples - 32, shuffle=False)

        images, labels = data.val.next_batch(32, shuffle=False)
        self.assertEqual(images.shape, (32, 32, 32, 3))
        self.assertEqual(labels.shape, (32, 10))
        data.val.next_batch(data.val.num_examples - 32, shuffle=False)

        images, labels = data.test.next_batch(32, shuffle=False)
        self.assertEqual(images.shape, (32, 32, 32, 3))
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

        _, labels = data.val.next_batch(
            data.val.num_examples, shuffle=False)

        self.assertEqual(labels.dtype, np.uint8)

        _, labels = data.test.next_batch(
            data.test.num_examples, shuffle=False)

        self.assertEqual(labels.dtype, np.uint8)

    def test_class_functions(self):
        self.assertEqual(data.classes, [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ])
        self.assertEqual(data.num_classes, 10)

        _, labels = data.test.next_batch(5, shuffle=False)

        self.assertEqual(data.classnames(labels[0]), ['cat'])
        self.assertEqual(data.classnames(labels[1]), ['ship'])
        self.assertEqual(data.classnames(labels[2]), ['ship'])
        self.assertEqual(data.classnames(labels[3]), ['airplane'])
        self.assertEqual(data.classnames(labels[4]), ['frog'])

        data.test.next_batch(data.test.num_examples - 5, shuffle=False)
