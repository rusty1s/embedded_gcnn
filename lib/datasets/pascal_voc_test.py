from unittest import TestCase

import numpy as np

from .pascal_voc import PascalVOC

data = PascalVOC('test_data', val_size=4)


class PascalVOCTest(TestCase):
    def test_init(self):
        self.assertEqual(data.train.num_examples, 4)
        self.assertEqual(data.val.num_examples, 4)
        self.assertEqual(data.test.num_examples, 4)

    def test_shapes(self):
        images, labels = data.train.next_batch(4, shuffle=False)
        for image in images:
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[0], 0)
            self.assertEqual(image.shape[2], 3)
        self.assertEqual(labels.shape, (4, 20))

        images, labels = data.val.next_batch(4, shuffle=False)
        for image in images:
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[0], 0)
            self.assertEqual(image.shape[2], 3)
        self.assertEqual(labels.shape, (4, 20))

        images, labels = data.test.next_batch(4, shuffle=False)
        for image in images:
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[0], 0)
            self.assertEqual(image.shape[2], 3)
        self.assertEqual(labels.shape, (4, 20))

    def test_images(self):
        images, _ = data.train.next_batch(
            data.train.num_examples, shuffle=False)

        for image in images:
            self.assertEqual(image.dtype, np.float32)
            self.assertLessEqual(image.max(), 1)
            self.assertGreaterEqual(image.min(), 0)

        images, _ = data.val.next_batch(data.val.num_examples, shuffle=False)

        for image in images:
            self.assertEqual(image.dtype, np.float32)
            self.assertLessEqual(image.max(), 1)
            self.assertGreaterEqual(image.min(), 0)

        images, _ = data.test.next_batch(data.test.num_examples, shuffle=False)

        for image in images:
            self.assertEqual(image.dtype, np.float32)
            self.assertLessEqual(image.max(), 1)
            self.assertGreaterEqual(image.min(), 0)

    def test_labels(self):
        _, labels = data.train.next_batch(4, shuffle=False)
        self.assertEqual(labels.dtype, np.uint8)

        _, labels = data.val.next_batch(4, shuffle=False)
        self.assertEqual(labels.dtype, np.uint8)

        _, labels = data.test.next_batch(4, shuffle=False)
        self.assertEqual(labels.dtype, np.uint8)

    def test_class_functions(self):
        self.assertEqual(data.classes, [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa',
            'tvmonitor'
        ])
        self.assertEqual(data.num_classes, 20)

        _, labels = data.test.next_batch(4, shuffle=False)

        self.assertEqual(data.classnames(labels[0]), ['person'])
        self.assertEqual(data.classnames(labels[1]), ['person', 'aeroplane'])
        self.assertEqual(data.classnames(labels[2]), ['aeroplane'])
        self.assertEqual(data.classnames(labels[3]), ['tvmonitor'])

    def test_filter_labels(self):
        filtered_data = PascalVOC(
            'test_data',
            val_size=4,
            classes=['aeroplane'])

        images, labels = filtered_data.test.next_batch(4, shuffle=False)

        self.assertEqual(len(images), 4)
        self.assertEqual(labels.shape, (4, 1))

        for label in labels:
            self.assertEqual(label[0], 1)
            self.assertEqual(filtered_data.classnames(label), ['aeroplane'])
